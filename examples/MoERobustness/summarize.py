"""Verify retained run evidence and compute the predeclared paired summary."""

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from .data import SPEC, SPEC_PATH, digest
from .metrics import class_metrics, write_json

CELLS = {("dense", "clean"), ("dense", "proxy"), ("moe", "clean"), ("moe", "proxy")}


def paired_class4(arms):
    if set(arms) != CELLS:
        raise ValueError("The paired comparison requires all four arms")
    return (arms["dense", "clean"] - arms["dense", "proxy"]
            - (arms["moe", "clean"] - arms["moe", "proxy"]))


def verify_arm(directory, manifest):
    if manifest["status"] != "complete" or manifest.get("native_exit") != 0:
        raise ValueError(f"Incomplete cell at {directory.name}")
    if manifest["source"]["dirty"] or manifest["protocol_sha256"] != digest(SPEC_PATH.read_bytes()):
        raise ValueError("Source or protocol differs")
    files = manifest["files"]
    required = {"cohort.npz", "metrics.jsonl", "predictions.npz", "summary.json", "weights.npz",
                "engine/config.yaml"}
    if not required.issubset(files):
        raise ValueError("Missing a required hashed artifact")
    for name, expected in files.items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory.resolve()) or digest(file.read_bytes()) != expected:
            raise ValueError(f"Artifact hash mismatch at {directory.name}/{name}")
    metrics = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
        result = {key: predictions[key] for key in predictions.files}
    if not np.array_equal(result["test_ids"], np.arange(10000)):
        raise ValueError("Pilot test IDs are not the full official sequence")
    if (result["logits"].shape != (10000, 10) or not np.isfinite(result["logits"]).all()
            or not np.array_equal(result["logits"].argmax(axis=1), result["predicted"])):
        raise ValueError("Invalid logit or prediction arrays")
    recalculated = class_metrics(result["labels"], result["predicted"])
    if any(value != metrics["test"][key] for key, value in recalculated.items()):
        raise ValueError("Stored test metrics differ from the prediction arrays")
    epochs = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if [entry["epoch"] for entry in epochs] != list(range(1, SPEC["epochs"] + 1)):
        raise ValueError("Incomplete fixed-epoch ledger")
    if metrics["train"]["examples"] != 10 * SPEC["train_per_digit"]:
        raise ValueError("Unexpected train size")
    wanted = (SPEC["train_per_digit"] * SPEC["proxy_fraction_numerator"]
              // SPEC["proxy_fraction_denominator"]) if manifest["data_condition"] == "proxy" else 0
    if metrics["train"]["modified_input_count"] != wanted:
        raise ValueError("Unexpected number of modified inputs")
    route = result["expert"]
    if manifest["architecture"] == "moe":
        if route.shape != (10000,) or not ((route >= 0) & (route < SPEC["experts"])).all():
            raise ValueError("Incomplete held-out assignment records")
    elif route.shape != (10000,) or not (route == -1).all():
        raise ValueError("Unexpected assignment records for dense model")
    return metrics, result


def summarize_pair(arms):
    f1 = {arm: values[0]["test"]["f1"][4] for arm, values in arms.items()}
    delta = paired_class4(f1)
    clean_dense = arms["dense", "clean"][0]
    clean_moe, route_reference = arms["moe", "clean"]
    labels = route_reference["labels"]
    if any(not np.array_equal(predictions["labels"], labels) for _, predictions in arms.values()):
        raise ValueError("The test labels differ across arms")
    all_provenances = [row["provenance"] for row, _ in arms.values()]
    if any(record != all_provenances[0] for record in all_provenances):
        raise ValueError("Split or data hashes differ across arms")
    shared = clean_dense["model"]["shared_initial_weight_digests"]
    if any(row["model"]["shared_initial_weight_digests"] != shared for row, _ in arms.values()):
        raise ValueError("Shared initial weights differ across arms")
    for condition in ("clean", "proxy"):
        if (arms["dense", condition][0]["train"]["inputs_sha256"]
                != arms["moe", condition][0]["train"]["inputs_sha256"]):
            raise ValueError("Architectures saw different training inputs")

    diag = clean_moe["diagnostic"]["assignment_counts"]
    top = sorted(range(SPEC["experts"]), key=lambda index: (-diag[index], index))[:2]
    outside = ~np.isin(route_reference["expert"], top)
    outside_rates = {}
    if outside.any():
        outside_rates = {
            f"{arch}_{condition}": float(np.mean(predictions["predicted"][outside] != labels[outside]))
            for (arch, condition), (_, predictions) in arms.items()
        }
    route_counts = clean_moe["validation"]["route_counts"]["all"]
    collapse = max(route_counts) / sum(route_counts)
    dense_val = clean_dense["validation"]["macro_f1"]
    moe_val = clean_moe["validation"]["macro_f1"]
    reasons = []
    if min(dense_val, moe_val) < SPEC["minimum_clean_val_macro_f1"]:
        reasons.append("clean validation macro F1 below preregistered floor")
    if abs(dense_val - moe_val) > SPEC["max_clean_model_f1_gap"]:
        reasons.append("clean dense and MoE validation scores differ by over 3 F1 points")
    if collapse > SPEC["max_clean_router_fraction"]:
        reasons.append("single expert routed over 90% of clean validation examples")
    return {
        "paired_class4_f1_advantage_moe": delta,
        "clean_validation_macro_f1": {"dense": dense_val, "moe": moe_val},
        "clean_router_max_validation_load": collapse, "feasibility_failures": reasons,
        "fixed_clean_reference_top_two_experts": top,
        "outside_reference_group_size": int(outside.sum()),
        "outside_reference_error_rates": outside_rates,
        "arms": {
            f"{arch}_{condition}": {
                "test_f1_4": values[0]["test"]["f1"][4], "test_f1_9": values[0]["test"]["f1"][9],
                "test_macro_f1": values[0]["test"]["macro_f1"],
                "test_bystander_macro_f1": values[0]["test"]["bystander_macro_f1"],
                "test_4_as_9": values[0]["test"]["confusion"][4][9],
                "test_9_as_4": values[0]["test"]["confusion"][9][4],
                "diagnostic_assignments": values[0]["diagnostic"]["assignment_counts"],
            } for (arch, condition), values in arms.items()
        },
    }


def summarize(root, seeds):
    cells, invalid = {}, []
    for path in sorted(root.rglob("manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") != "pilot" or record.get("seed") not in seeds:
            continue
        key = (record["seed"], record["architecture"], record["data_condition"])
        if key in cells:
            raise ValueError(f"Ambiguous duplicate pilot outcome: {key}")
        cells[key] = (record, path.parent)
    expected = {(seed, *arm) for seed in seeds for arm in CELLS}
    missing = sorted(expected - cells.keys())
    pairs = {}
    pins = {(record.get("source", {}).get("commit"), record.get("source", {}).get("tree"),
             record.get("protocol_sha256")) for record, _ in cells.values()}
    if len(pins) > 1:
        raise ValueError("Pilot cells use mixed code and protocol pins")
    for seed in seeds:
        if any(cell[0] == seed for cell in missing):
            continue
        try:
            arms = {arm: verify_arm(directory, manifest)
                    for arm in CELLS for manifest, directory in [cells[(seed, *arm)]]}
            pairs[seed] = summarize_pair(arms)
        except (ValueError, OSError, KeyError) as error:
            invalid.append({"seed": seed, "reason": str(error)})
    deltas = [summary["paired_class4_f1_advantage_moe"] for summary in pairs.values()]
    median = statistics.median(deltas) if deltas else None
    complete = not missing and not invalid and len(pairs) == len(seeds)
    interpretable = complete and all(not summary["feasibility_failures"] for summary in pairs.values())
    matched_sign = bool(deltas) and (min(deltas) > 0 or max(deltas) < 0)
    return {
        "study": SPEC["study"], "expected_runs": len(expected), "observed_manifests": len(cells),
        "complete": complete, "passes_feasibility": interpretable, "missing": missing,
        "invalid": invalid, "pairs": pairs, "primary_paired_median": median,
        "primary_paired_range": [min(deltas), max(deltas)] if deltas else None,
        "mechanism_followup_threshold_met": (
            interpretable and len(seeds) == len(SPEC["seeds"]) and matched_sign
            and abs(median) >= SPEC["mechanism_threshold_f1"]
        ),
        "interpretation": "Exploratory matched comparisons within one fixed MNIST proxy cohort.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=SPEC["seeds"])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.runs, [args.seed] if args.seed else SPEC["seeds"])
    write_json(args.output, result)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())

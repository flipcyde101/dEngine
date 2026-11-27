"""Unlock the fresh evaluation set only for the complete verified matrix."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
import math
import time

import numpy as np
from scipy.stats import t
import torch

from ..data import array_digest
from ..metrics import class_metrics, evaluate, mean_cross_entropy, routing_counts
from ..run import environment, source_state, tensor_dataset
from .design import (ARMS, N_CASES, PROTOCOL_PATH, SPEC, case_id, derive_seed, file_digest,
                     interpretation, read_json, starts_for, write_json)
from .fit import ARCHITECTURES, weights_digest
from .verify import verify_start


def paired_loss(results, reference, specialist):
    def loss(model):
        return results[model + "-clean"]["f1"][4] - results[model + "-proxy"]["f1"][4]
    return 100 * (loss(reference) - loss(specialist))


def interval(values):
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Require two or more finite cohort means")
    middle, df = mean(values), len(values) - 1
    half = float(t.ppf(.975, df) * stdev(values) / math.sqrt(len(values)))
    return {"mean_pp": middle, "low_pp": middle - half, "high_pp": middle + half, "df": df}


def verify_provider(config, provider):
    for field in ("serviceAccount", "timeout", "queueTtl"):
        if provider.get(field) != config[field]:
            raise ValueError(f"Build request differs in {field}")
    if set(provider["tags"]) != set(config["tags"]):
        raise ValueError("Build tags differ from request")
    if any(provider["options"].get(key) != value for key, value in config["options"].items()):
        raise ValueError("Build machine or logging differs from request")
    if provider["options"].get("pool") or len(provider["steps"]) != len(config["steps"]):
        raise ValueError("Unexpected pool or extra build step")
    for expected, observed in zip(config["steps"], provider["steps"]):
        if any(expected[key] != observed.get(key) for key in ("id", "name", "entrypoint")):
            raise ValueError("Unexpected build step")
        args = [arg.replace("$$", "$") for arg in expected["args"]]
        if observed.get("args") != args:
            raise ValueError("Build arguments differ from request")


def verified_job(lab, job, source):
    key = job["name"]
    provider = read_json(lab / f"{key}-provider.json")
    config = lab / job["config"]
    if file_digest(config) != job["config_sha256"]:
        raise ValueError(f"Reviewed configuration fingerprint changed for {key}")
    verify_provider(read_json(config), provider)
    archive = lab / f"{key}-receipt"
    reassembly = read_json(archive / "reassembly.json")
    worker = read_json(archive / "outputs" / "worker.json")
    if (provider["status"] != "SUCCESS" or job["tag"] not in provider["tags"]
            or reassembly["native_exit_pairs"] != [[0, 0]]
            or not any("none true false 65532:65532 cap_drop_all" in line
                       for line in reassembly["isolation_lines"])
            or reassembly["sha256"] != file_digest(archive / "worker-evidence.tar.gz")):
        raise ValueError(f"Unverified provider or archive receipt for {key}")
    for field, expected in {
        "source": source, "kind": job["kind"], "cohort": job["cohort"], "cohort_sha256": job["cohort_sha256"],
        "slots": job["slots"], "verified": job["slots"], "native_exit": 0, "error": None,
        "version": "3.13.18", "platform": "x86_64", "torch": "2.10.0+cpu", "torchvision": "0.25.0+cpu",
    }.items():
        if worker.get(field) != expected:
            raise ValueError(f"Incorrect worker {field} for {key}")
    cohort = job["cohort"]
    starts = [starts_for(cohort)[slot - 1] for slot in job["slots"]]
    expected_labels = [case_id(cohort, start["slot"], arm) for start in starts for arm in start["order"]]
    if job["kind"] == "qualify":
        expected_labels = ["focused-tests", *expected_labels]
    if ([proc["label"] for proc in worker["processes"]] != expected_labels
            or any(proc["native_exit"] != 0 or proc["timed_out"] for proc in worker["processes"])):
        raise ValueError(f"Incomplete native process receipts for {key}")
    return provider["id"], archive / "outputs" / "cases", starts, worker


def qualification(lab, packet, source):
    jobs = [job for job in packet["jobs"] if job["kind"] == "qualify"]
    if {job["name"] for job in jobs} != {"qualify-a", "qualify-b"} or len(jobs) != 2:
        raise ValueError("Require the two planned qualification runs")
    observed, build_ids = [], []
    for job in jobs:
        provider, folder, starts, _ = verified_job(lab, job, source)
        records = verify_start(folder, job["cohort"], job["cohort_sha256"], starts[0], source)
        build_ids.append(provider)
        observed.append(records)
    if build_ids[0] == build_ids[1]:
        raise ValueError("Duplicate qualification receipt")
    differences, equal_weights = [], 0
    for arm in ARMS:
        left, right = (records[arm] for records in observed)
        for key in ("case", "cohort_sha256", "initial_shared", "initial_model", "batch_order"):
            if left[key] != right[key]:
                raise ValueError("Qualification replay has different inputs or initialization")
        equal_weights += left["final_model"] == right["final_model"]
        a, b = left["validation"], right["validation"]
        differences.extend((abs(a["macro_f1"] - b["macro_f1"]), abs(a["f1"][4] - b["f1"][4])))
    maximum = 100 * max(differences)
    if maximum > .5:
        raise ValueError("Qualification development F1 replay exceeds the 0.5 point tolerance")
    return {
        "passed": True, "build_ids": build_ids, "replayed_fits": len(ARMS), "equal_model_arrays": equal_weights,
        "max_replay_development_f1_difference_pp": maximum,
        "cpu_models": [records["dense-clean"]["runtime"]["cpu_model"] for records in observed],
        "source_commit": source["commit"]
    }


def verified_matrix(lab, sealed_dir):
    packet = read_json(lab / "packet.json")
    design = read_json(sealed_dir / "study.json")
    source = source_state("pilot")
    if (source["commit"] != packet["commit"] or source["tree"] != packet["tree"]
            or file_digest(sealed_dir / "study.json") != packet["study_sha256"]
            or file_digest(PROTOCOL_PATH) != design["protocol_sha256"]):
        raise ValueError("Study, run packet or evaluator source differs from the frozen design")
    checks = qualification(lab, packet, source)
    rows, builds, cohorts = {}, set(), {}
    for item in design["cohorts"]:
        path = sealed_dir / item["file"]
        if file_digest(path) != item["sha256"]:
            raise ValueError("A planned cohort changed")
        cohorts[item["cohort"]] = (item["sha256"], read_json(path))
    for job in packet["jobs"]:
        if job["kind"] != "study":
            continue
        cohort = job["cohort"]
        cohort_sha, _ = cohorts[cohort]
        if job["cohort_sha256"] != cohort_sha:
            raise ValueError("Provider job differs from its planned cohort")
        provider, folder, starts, _ = verified_job(lab, job, source)
        if provider in builds or provider in checks["build_ids"]:
            raise ValueError("Duplicated provider receipt")
        builds.add(provider)
        for start in starts:
            key = (cohort, start["slot"])
            if key in rows:
                raise ValueError("More than one result for a planned start")
            rows[key] = (folder, verify_start(folder, cohort, cohort_sha, start, source))
    expected = {(cohort, slot) for cohort in range(1, SPEC["cohorts"] + 1)
                for slot in range(1, SPEC["starts_per_cohort"] + 1)}
    if set(rows) != expected or sum(len(records) for _, records in rows.values()) != N_CASES:
        raise ValueError("Study matrix is incomplete")
    return design, source, cohorts, rows, builds, checks


def open_holdout(sealed_dir, design, cohorts):
    path = sealed_dir / "sealed-evaluation.npz"
    if file_digest(path) != design["evaluation_sha256"]:
        raise ValueError("Evaluation artifact differs from the original seal")
    with np.load(path, allow_pickle=False) as sealed:
        ids, images, labels = sealed["ids"], sealed["images"], sealed["labels"]
    if (len(ids) != len(set(ids)) or array_digest(ids) != design["evaluation_ids_sha256"]
            or images.shape != (len(ids), 28, 28) or images.dtype != np.uint8
            or labels.shape != ids.shape or labels.dtype != np.uint8
            or np.bincount(labels, minlength=10).tolist() != [SPEC["test_per_digit"]] * 10):
        raise ValueError("Malformed sealed evaluation data")
    members = set(ids.tolist())
    for _, cohort in cohorts.values():
        if members.intersection(cohort["train_ids"] + cohort["validation_ids"]):
            raise ValueError("A held-out image ID was used during training")
    return ids, images, labels


def score_case(arm, dataset, directory, row):
    model_class = ARCHITECTURES[arm.split("-")[0]]
    model = model_class(dataset)
    with np.load(directory / "weights.npz", allow_pickle=False) as packed:
        model.load_state_dict({key: torch.from_numpy(packed[key]) for key in packed.files}, strict=True)
    if weights_digest(model.state_dict()) != row["final_model"]:
        raise ValueError(f"Weights do not match the trained model for {row['case']}")
    labels, logits, expert = evaluate(model, dataset)
    predictions = logits.argmax(axis=1)
    metrics = class_metrics(labels, predictions)
    metrics["cross_entropy"] = mean_cross_entropy(labels, logits)
    metrics["route_counts"] = {
        "all": routing_counts(expert), "digit_4": routing_counts(expert[labels == 4]),
        "digit_9": routing_counts(expert[labels == 9]),
    }
    return metrics, predictions.astype(np.uint8), expert.astype(np.int8)


def feasibility(records):
    clean = {model: records[model + "-clean"]["validation"]["macro_f1"] for model in ARCHITECTURES}
    route_counts = records["moe-clean"]["validation"]["route_counts"]["all"]
    share = max(route_counts) / sum(route_counts)
    return {
        "clean_validation_macro_f1": clean,
        "clean_validation_max_expert_share": share,
        "flags": {
            "any_clean_model_below_floor": min(clean.values()) < SPEC["minimum_clean_validation_macro_f1"],
            "main_clean_comparison_gap": abs(clean["dense"] - clean["moe"]) > SPEC["max_clean_validation_macro_f1_gap"],
            "clean_router_concentration": share > SPEC["max_clean_router_share"],
        },
    }


def sensitivity(cohort_means, test_y, cases, guesses):
    rng = np.random.default_rng(derive_seed("resample"))
    bootstrap = np.asarray(cohort_means)[rng.integers(0, len(cohort_means), (10_000, len(cohort_means)))].mean(axis=1)
    actual_4 = test_y == 4
    position = {case: i for i, case in enumerate(cases)}
    weights = np.zeros(len(cases), dtype=np.float64)
    prefix = [key.removesuffix("-dense-clean") for key in cases if key.endswith("-dense-clean")]
    for key in prefix:
        for arm, sign in (("dense-clean", 1), ("dense-proxy", -1), ("moe-clean", -1), ("moe-proxy", 1)):
            weights[position[key + "-" + arm]] = 100 * sign / len(prefix)
    selects = guesses == 4
    true_positive = np.ascontiguousarray(selects[:, actual_4], dtype=np.float32)
    false_positive = np.ascontiguousarray(selects[:, ~actual_4], dtype=np.float32)
    count = int(actual_4.sum())
    all_weights = []
    # The same sampled images are applied to every checkpoint.
    for _ in range(256):
        group_counts = np.zeros(len(test_y), dtype=np.float32)
        for label in range(10):
            members = np.flatnonzero(test_y == label)
            group_counts[members] = np.bincount(rng.integers(0, len(members), len(members)), minlength=len(members))
        all_weights.append(group_counts)
    sampled = np.asarray(all_weights, dtype=np.float32)
    tp = true_positive @ sampled[:, actual_4].T
    fp = false_positive @ sampled[:, ~actual_4].T
    retest = weights @ (2 * tp / (count + tp + fp))
    return {
        "leave_one_cohort_out_mean_range_pp": [
            min(mean(cohort_means[:i] + cohort_means[i + 1:]) for i in range(len(cohort_means))),
            max(mean(cohort_means[:i] + cohort_means[i + 1:]) for i in range(len(cohort_means))),
        ],
        "cohort_bootstrap_95_percent_range_pp": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "common_evaluation_image_replay_95_percent_range_pp": np.percentile(retest, [2.5, 97.5]).tolist(),
        "cohort_bootstrap_repeats": len(bootstrap), "evaluation_resamples": len(retest),
    }


def score(lab, sealed_dir, out):
    design, source, cohorts, matrix, builds, checks = verified_matrix(lab, sealed_dir)
    print(f"FOLLOWUP_EVALUATION_AUTHORIZED_BY_MATRIX {len(matrix)} matches", flush=True)
    ids, images, labels = open_holdout(sealed_dir, design, cohorts)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dataset = tensor_dataset(images, labels)
    out.mkdir(parents=True, exist_ok=False)
    cases, predicted, routes, report = [], [], [], []
    started = time.monotonic()
    for (cohort, slot), (root, records) in sorted(matrix.items()):
        scores = {}
        for arm in ARMS:
            record = records[arm]
            key = record["case"]
            result, guesses, assignments = score_case(arm, dataset, root / key, record)
            scores[arm] = result
            cases.append(key)
            predicted.append(guesses)
            routes.append(assignments)
            print(f"FOLLOWUP_SCORED {len(cases)}/{N_CASES} {time.monotonic() - started:.3f}s {key}", flush=True)
        report.append({"cohort": cohort, "slot": slot, "scores": scores,
                       "main_effect_pp": paired_loss(scores, "dense", "moe"),
                       "capacity_control_effect_pp": paired_loss(scores, "wide", "moe"),
                       "feasibility": feasibility(records)})
    grouped = [mean(row["main_effect_pp"] for row in report if row["cohort"] == index)
               for index in sorted(cohorts)]
    wide_grouped = [mean(row["capacity_control_effect_pp"] for row in report if row["cohort"] == index)
                    for index in sorted(cohorts)]
    effect = interval(grouped)
    margin = SPEC["minimum_practical_difference_pp"]
    raw = out / "heldout-predictions.npz"
    stacked = np.asarray(predicted, dtype=np.uint8)
    with raw.open("xb") as handle:
        np.savez_compressed(handle, case_ids=np.asarray(cases), test_ids=ids, labels=labels,
                            predicted=stacked, expert=np.asarray(routes, dtype=np.int8))
    all_effects = [row["main_effect_pp"] for row in report]
    flagged = Counter(flag for row in report for flag, state in row["feasibility"]["flags"].items() if state)
    write_json(out / "study-results.json", {
        "study": SPEC["study"], "scored_fits": len(cases), "cohort_means_pp": grouped,
        "summary": effect, "primary_practical_margin_pp": margin,
        "practical_result": interpretation(effect["low_pp"], effect["high_pp"], margin),
        "capacity_control_descriptive": interval(wide_grouped), "capacity_control_cohort_means_pp": wide_grouped,
        "match_median_pp": median(all_effects), "match_range_pp": [min(all_effects), max(all_effects)],
        "feasibility_flags": dict(flagged), "sensitivity": sensitivity(grouped, labels, cases, stacked),
        "holdout_sha256": design["evaluation_sha256"], "predictions_sha256": file_digest(raw),
        "source": source, "scoring_runtime": environment(), "verified_study_builds": sorted(builds),
        "qualification_replay": checks,
        "matched_comparisons": report,
    })
    print(f"FOLLOWUP_SCORE_COMPLETE {len(cases)} fits", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", type=Path)
    parser.add_argument("sealed_design", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    if args.qualification_only:
        packet = read_json(args.lab / "packet.json")
        source = source_state("pilot")
        if source["commit"] != packet["commit"] or source["tree"] != packet["tree"]:
            raise ValueError("Qualification source differs from the reviewed run packet")
        report = qualification(args.lab, packet, source)
        write_json(args.out, report)
        print(report)
        return
    score(args.lab, args.sealed_design, args.out)


if __name__ == "__main__":
    main()

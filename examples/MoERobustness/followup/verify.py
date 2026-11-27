"""Verify paired checkpoints before the held-out evaluator can access them."""

import json
import re

from .design import ARMS, PROTOCOL_PATH, SPEC, case_id, file_digest, read_json


PARAMETERS = {"dense": 60458, "wide": 85742, "moe": 85678}
ARTIFACTS = {"weights.npz", "metrics.jsonl", "engine/config.yaml"}


def verify_pairing(records):
    if set(records) != set(ARMS):
        raise ValueError("All six matched arms are required")
    shared = records["dense-clean"]["initial_shared"]
    ordering = records["dense-clean"]["batch_order"]
    for record in records.values():
        if record["initial_shared"] != shared:
            raise ValueError("Shared initial weights differ")
        if record["batch_order"] != ordering:
            raise ValueError("Actual batch order differs")
    for model in PARAMETERS:
        if records[f"{model}-clean"]["initial_model"] != records[f"{model}-proxy"]["initial_model"]:
            raise ValueError(f"Paired initial weights differ for {model}")


def verify_start(cases, cohort, cohort_sha, start, source):
    result = {}
    for arm in ARMS:
        key = case_id(cohort, start["slot"], arm)
        folder = cases / key
        row = read_json(folder / "case.json")
        wanted = {
            "case": key, "arm": arm, "slot": start["slot"], "cohort": cohort,
            "cohort_sha256": cohort_sha, "protocol_sha256": file_digest(PROTOCOL_PATH),
            "model_seed": start["model_seed"], "order_seed": start["order_seed"], "native_exit": 0,
            "parameters": PARAMETERS[arm.split("-")[0]],
        }
        if any(row.get(name) != value for name, value in wanted.items()) or row["source"] != source:
            raise ValueError(f"Wrong identity or source for {key}")
        if source["dirty"] or row["runtime"]["python"] != "3.13.18" or row["runtime"]["torch"] != "2.10.0+cpu":
            raise ValueError(f"Unexpected source or runtime for {key}")
        if set(row["files"]) != ARTIFACTS or any(file_digest(folder / name) != sha for name, sha in row["files"].items()):
            raise ValueError(f"Altered or missing output artifact for {key}")
        if len(row["batch_order"]) != SPEC["epochs"] or any(
                item["examples"] != 10 * SPEC["train_per_digit"]
                or not re.fullmatch("[a-f0-9]{64}", item["order_sha256"]) for item in row["batch_order"]):
            raise ValueError(f"Incomplete training order for {key}")
        with (folder / "metrics.jsonl").open(encoding="utf-8") as stream:
            epochs = [json.loads(line) for line in stream]
        if ([epoch["epoch"] for epoch in epochs] != list(range(1, SPEC["epochs"] + 1))
                or row["validation"] != epochs[-1]["validation"]):
            raise ValueError(f"Incorrect development epoch ledger for {key}")
        result[arm] = row
    verify_pairing(result)
    return result

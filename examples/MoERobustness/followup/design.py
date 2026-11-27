"""Frozen scope, run identifiers and the prespecified interpretation."""

from hashlib import sha256
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
SPEC = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = ("dense-clean", "moe-proxy", "wide-clean", "dense-proxy", "moe-clean", "wide-proxy")
N_CASES = SPEC["cohorts"] * SPEC["starts_per_cohort"] * len(ARMS)


def digest(raw):
    return sha256(raw).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def canonical_json(record):
    return (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def derive_seed(domain, *indices):
    key = canonical_json([SPEC["master_seed"], domain, *indices])
    return int.from_bytes(sha256(key).digest()[:8], "big") % (2 ** 32)


def arm_order(index):
    offset = index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def case_id(cohort, slot, arm):
    if arm not in ARMS or not 1 <= cohort <= SPEC["cohorts"] or not 1 <= slot <= SPEC["starts_per_cohort"]:
        raise ValueError("Unknown study case")
    return f"c{cohort:02d}-s{slot}-{arm}"


def starts_for(cohort):
    return [{
        "slot": slot,
        "model_seed": derive_seed("model", cohort, slot),
        "order_seed": derive_seed("batches", cohort, slot),
        "order": list(arm_order((cohort - 1) * SPEC["starts_per_cohort"] + slot - 1)),
    } for slot in range(1, SPEC["starts_per_cohort"] + 1)]


def interpretation(low, high, margin):
    if not all(math.isfinite(value) for value in (low, high, margin)) or low > high or margin <= 0:
        raise ValueError("Invalid interval or margin")
    if low > margin:
        return "material_advantage"
    if high < -margin:
        return "material_disadvantage"
    if low > -margin and high < margin:
        return "small_average_difference"
    return "unresolved"

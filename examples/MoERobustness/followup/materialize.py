"""Standard-library input preparation for the connected worker step."""

import argparse
import gzip
import json
from pathlib import Path
import struct
import urllib.request

from .design import SPEC, digest, read_json, write_json


FILES = Path(__file__).resolve().parents[1] / "protocol.json"


def study_inputs(source=None):
    public = json.loads(FILES.read_text(encoding="utf-8"))["dataset"][:2]
    result = []
    for row, meta in enumerate(public):
        if meta["url"] != "https://ossci-datasets.s3.amazonaws.com/mnist/" + meta["name"]:
            raise ValueError("Unexpected dataset URL")
        if source is None:
            with urllib.request.urlopen(meta["url"], timeout=60) as response:
                packed = response.read(meta["bytes"] + 1)
        else:
            packed = (source / meta["name"]).read_bytes()
        if len(packed) != meta["bytes"] or digest(packed) != meta["sha256"]:
            raise ValueError(f"Unexpected public dataset bytes for {meta['name']}")
        raw = gzip.decompress(packed)
        header = struct.pack(">IIII", 2051, 60_000, 28, 28) if row == 0 else struct.pack(">II", 2049, 60_000)
        if not raw.startswith(header) or len(raw) != len(header) + 60_000 * (784 if row == 0 else 1):
            raise ValueError("Unexpected IDX dimensions")
        result.append(raw[len(header):])
    return result


def materialize(cohort, out, source=None):
    selected = cohort["train_ids"] + cohort["validation_ids"]
    if (len(cohort["train_ids"]) != 10 * SPEC["train_per_digit"]
            or len(cohort["validation_ids"]) != 10 * SPEC["validation_per_digit"]
            or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < 60_000 for i in selected)):
        raise ValueError("Malformed training or validation IDs")
    images, labels = study_inputs(source)
    subset = b"".join(images[i * 784:(i + 1) * 784] for i in selected)
    targets = bytes(labels[i] for i in selected)
    if digest(subset) != cohort["images_sha256"] or digest(targets) != cohort["labels_sha256"]:
        raise ValueError("Selected inputs do not match the reviewed fingerprints")
    out.mkdir(parents=True, exist_ok=False)
    for name, data in (("images.bin", subset), ("labels.bin", targets)):
        with (out / name).open("xb") as handle:
            handle.write(data)
    write_json(out / "cohort.json", cohort)
    print(f"FOLLOWUP_SUBSET {len(selected)} training and validation examples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    materialize(read_json(args.cohort), args.out, args.source)


if __name__ == "__main__":
    main()

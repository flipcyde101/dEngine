"""Execute and archive one bounded isolated Cloud Build workload."""

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch
import torchvision

from ..data import EXPECTED_COHORT, SPEC, fingerprints, make_cohort, read_mnist
from ..metrics import write_json
from ..summarize import summarize

ARMS = (("dense", "clean"), ("dense", "proxy"), ("moe", "clean"), ("moe", "proxy"))


def run_process(label, argv, timeout, seed=None):
    env = {**os.environ, "PYTHONHASHSEED": str(seed if seed is not None else 11)}
    print(f"DENGINE_START {label} {datetime.now(timezone.utc).isoformat()}", flush=True)
    start = time.monotonic()
    timed_out = False
    try:
        result = subprocess.run(argv, env=env, timeout=timeout, check=False)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = None
        timed_out = True
    duration = time.monotonic() - start
    print(f"DENGINE_PROCESS_EXIT {label} {code} timed_out={timed_out} {duration:.3f}s",
          flush=True)
    return {"label": label, "native_exit": code, "timed_out": timed_out,
            "elapsed_s": duration}


def cell(mode, seed, arch, condition, directory, dataset):
    return run_process(
        f"{mode}-{seed}-{arch}-{condition}-{directory.name}",
        [sys.executable, "-Bu", "-m", "examples.MoERobustness.run", "--mode", mode,
         "--arch", arch, "--data", condition, "--seed", str(seed), "--dataset-dir",
         str(dataset), "--output-dir", str(directory)],
        timeout=330 if mode == "pilot" else 120, seed=seed,
    )


def compressed_bundle(out):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gzf:
        with tarfile.open(fileobj=gzf, mode="w|") as archive:
            for path in sorted(out.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Unexpected symlink in experiment output")
                if not path.is_file():
                    continue
                info = archive.gettarinfo(str(path), arcname=path.relative_to(out).as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    return buffer.getvalue()


def print_bundle(out):
    bundle = compressed_bundle(out)
    encoded = base64.b64encode(bundle).decode("ascii")
    size = 10000
    parts = [encoded[i:i + size] for i in range(0, len(encoded), size)]
    meta = {"bytes": len(bundle), "sha256": hashlib.sha256(bundle).hexdigest(),
            "chunks": len(parts)}
    print("DENGINE_BUNDLE " + json.dumps(meta, sort_keys=True), flush=True)
    for index, part in enumerate(parts):
        print(f"DENGINE_CHUNK {index:05d}/{len(parts):05d} {part}", flush=True)
    print("DENGINE_BUNDLE_DONE " + meta["sha256"], flush=True)


def canary(out, dataset, status):
    train, labels, test, test_labels = read_mnist(dataset)
    cohort = make_cohort(train, labels)
    observed = np.flatnonzero(np.any(cohort.clean != cohort.proxy, axis=(1, 2)))
    if not np.array_equal(cohort.train_ids[observed], cohort.pairs[:, 0]):
        raise ValueError("Changed MNIST inputs differ from the frozen training cohort")
    if len(cohort.pairs) != 300 or test.shape != (10000, 28, 28):
        raise ValueError("Unexpected pilot data cardinalities")
    observed_cohort = fingerprints(cohort, test, test_labels)
    if observed_cohort != EXPECTED_COHORT:
        raise ValueError("The worker did not reconstruct the frozen cohort")
    status["mnist"] = {
        "train_size": len(cohort.train_ids), "validation_size": len(cohort.val_ids),
        "modified_training_inputs": len(observed), "test_size": len(test),
        "fingerprints": observed_cohort,
    }
    receipt = run_process(
        "focused-pytest",
        [
            sys.executable, "-B", "-m", "pytest", "-q", "-p",
            "no:cacheprovider", "tests/test_moe_robustness.py",
        ],
        180,
    )
    status["processes"].append(receipt)
    if receipt["native_exit"] != 0:
        return 124 if receipt["timed_out"] else receipt["native_exit"]

    for name in ("replay-a", "replay-b"):
        receipt = cell("smoke", 11, "moe", "proxy", out / name, dataset)
        status["processes"].append(receipt)
        if receipt["native_exit"] != 0:
            return 124 if receipt["timed_out"] else receipt["native_exit"]
    for file in ("predictions.npz", "weights.npz"):
        with (np.load(out / "replay-a" / file, allow_pickle=False) as left,
              np.load(out / "replay-b" / file, allow_pickle=False) as right):
            if left.files != right.files or any(not np.array_equal(left[k], right[k]) for k in left.files):
                raise ValueError(f"Same-worker synthetic replay differs for {file}")
    status["deterministic_synthetic_replay"] = "bitwise equal arrays"
    metrics = json.loads((out / "replay-a" / "summary.json").read_text(encoding="utf-8"))
    initial = metrics["initial_validation"]["cross_entropy"]
    final = metrics["validation"]["cross_entropy"]
    status["synthetic_learning"] = {"initial_cross_entropy": initial, "final_cross_entropy": final}
    if final >= initial:
        raise ValueError("Synthetic learning check did not reduce validation loss")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=("canary", "pilot"))
    parser.add_argument("--seed", type=int, choices=SPEC["seeds"])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.job == "pilot") != (args.seed is not None):
        parser.error("Provide a seed for a pilot job only")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    status = {
        "job": args.job, "seed": args.seed, "python": platform.python_version(),
        "machine": platform.machine(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "processes": [],
        "started_utc": datetime.now(timezone.utc).isoformat(), "error": None,
    }
    code = 1
    try:
        if sys.version_info[:2] != (3, 13) or platform.machine() != "x86_64":
            raise RuntimeError("Pilot requires Linux amd64 CPython 3.13")
        if (str(torch.__version__) != "2.10.0+cpu"
                or torchvision.__version__ != "0.25.0+cpu" or torch.cuda.is_available()):
            raise RuntimeError("Incorrect CPU torch or torchvision worker")
        if args.job == "canary":
            code = canary(out, args.dataset_dir, status)
        else:
            code = 0
            for index, (arch, condition) in enumerate(ARMS):
                print(f"DENGINE_PROGRESS {index}/4 cells complete, next {arch}/{condition}", flush=True)
                receipt = cell("pilot", args.seed, arch, condition,
                               out / f"seed-{args.seed}-{arch}-{condition}", args.dataset_dir)
                status["processes"].append(receipt)
                if receipt["native_exit"] != 0:
                    code = 124 if receipt["timed_out"] else receipt["native_exit"]
                    break
            print(f"DENGINE_PROGRESS {sum(r['native_exit'] == 0 for r in status['processes'])}"
                  "/4 cells complete, validating and archiving outputs", flush=True)
            outcome = summarize(out, [args.seed])
            write_json(out / "paired-summary.json", outcome)
            if code == 0 and not outcome["complete"]:
                code = 2
    except Exception as error:
        code = 1
        status["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        status.update({"native_exit": code, "finished_utc": datetime.now(timezone.utc).isoformat()})
        write_json(out / "worker.json", status)
        print_bundle(out)
    return code


if __name__ == "__main__":
    sys.exit(main())

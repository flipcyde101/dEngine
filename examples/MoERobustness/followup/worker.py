"""Run a bounded group of matches and return native exits and checkpoints."""

import argparse
from datetime import datetime, timezone
import platform
import sys

import torch
import torchvision

from _dengine_cli_paths import cli_input_path, cli_output_path

from ..lab.worker import WorkerTask, print_bundle, run_process
from ..run import source_state
from .design import case_id, file_digest, read_json, starts_for, write_json
from .verify import verify_start


def run_worker(args):
    args.output.mkdir(exist_ok=False, parents=True)
    cases = args.output / "cases"
    cases.mkdir()
    source = source_state("pilot")
    status = {
        "kind": args.kind, "slots": args.slots, "cohort_sha256": args.cohort_sha,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "version": platform.python_version(), "platform": platform.machine(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "processes": [], "verified": [], "error": None,
    }
    rc = 1
    try:
        if (sys.platform != "linux" or platform.machine() != "x86_64" or platform.python_version() != "3.13.18"
                or torch.__version__ != "2.10.0+cpu" or torchvision.__version__ != "0.25.0+cpu"
                or torch.cuda.is_available()):
            raise ValueError("Unexpected worker runtime")
        if file_digest(args.directory / "cohort.json") != args.cohort_sha:
            raise ValueError("Unexpected work request inputs")
        data = read_json(args.directory / "cohort.json")
        cohort = data["cohort"]
        status["cohort"] = cohort
        starts = {start["slot"]: start for start in starts_for(cohort)}
        if args.slots != sorted(set(args.slots)) or any(slot not in starts for slot in args.slots):
            raise ValueError("Invalid match slot")
        if args.kind == "qualify":
            check = run_process(
                "focused-tests", WorkerTask.TESTS,
                ["-q", "-p", "no:cacheprovider", "tests/test_moe_followup.py"], 180,
            )
            status["processes"].append(check)
            if check["native_exit"] != 0:
                raise RuntimeError("Worker checks failed")
        total = 6 * len(args.slots)
        for slot in args.slots:
            start = starts[slot]
            for arm in start["order"]:
                key = case_id(cohort, slot, arm)
                receipt = run_process(key, WorkerTask.FOLLOWUP,
                                      ["--", str(args.directory), args.cohort_sha, str(slot), arm, str(cases / key)],
                                      100, start["model_seed"])
                status["processes"].append(receipt)
                if receipt["native_exit"] != 0:
                    raise RuntimeError(f"Cell did not finish successfully: {key}")
                print(f"FOLLOWUP_CASE_FINISHED {key} {receipt['elapsed_s']:.3f}", flush=True)
            verify_start(cases, cohort, args.cohort_sha, start, source)
            status["verified"].append(slot)
            print(f"FOLLOWUP_MATCHES_VERIFIED {len(status['verified']) * 6}/{total}", flush=True)
        rc = 0
    except Exception as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        print(f"FOLLOWUP_FAILURE {type(error).__name__}: {error}", flush=True)
    finally:
        status.update(native_exit=rc, finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "worker.json", status)
        print_bundle(args.output)
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=cli_input_path)
    parser.add_argument("cohort_sha")
    parser.add_argument("output", type=cli_output_path)
    parser.add_argument("--kind", choices=("study", "qualify"), required=True)
    parser.add_argument("--slots", nargs="+", type=int, required=True)
    return run_worker(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())

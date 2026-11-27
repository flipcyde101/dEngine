"""Run one frozen cell through the dEngine centralized scenario."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import distributions
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from dengine import load_experiment_from_yamls
from dengine.bin.args_parser import SimulationArguments, VerbosityLevel
from dengine.bin.simulation import load_engine
from dengine.dataset import SupervisedDataset
from dengine.dataset.decorators import register_dataset
from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.partitioning.decorators import register_partitioning

from .data import SPEC, SPEC_PATH, array_digest, digest, fixture, make_cohort, read_mnist
from .metrics import PilotMetrics, evaluate, routing_counts, write_json
from .models import DenseDigitClassifier, SparseDigitClassifier
from .progress import Progress


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@lru_cache(maxsize=1)
def study_inputs(mode, directory):
    arrays = fixture() if mode == "smoke" else read_mnist(Path(directory))
    train, labels, test, test_labels = arrays
    # Copy once, then use the same unmodified source across the loaders.
    return (make_cohort(train, labels, 10 if mode == "smoke" else None,
                        5 if mode == "smoke" else None), test, test_labels)


def tensor_dataset(pixels, labels):
    images = torch.tensor(pixels, dtype=torch.float32).unsqueeze(1).div_(255)
    return SupervisedDataset(images, torch.tensor(labels, dtype=torch.long))


@register_dataset("moe_pilot_train")
def training_dataset(output_path, experiment_cfg, mode, condition) -> SupervisedDataset:
    cohort, _, _ = study_inputs(mode, output_path)
    pixels = cohort.clean if condition == "clean" else cohort.proxy
    return tensor_dataset(pixels, cohort.labels)


@register_dataset("moe_pilot_test")
def test_dataset(output_path, experiment_cfg, mode) -> SupervisedDataset:
    _, pixels, labels = study_inputs(mode, output_path)
    return tensor_dataset(pixels, labels)


@register_partitioning("moe_pilot_partition")
def partition(dataset, test, partitions, graph, train_size) -> TYPE_DATASET_PARTITIONING:
    nodes = list(graph.nodes)
    if len(nodes) != 1 or not 0 < train_size < len(dataset):
        raise ValueError("The frozen pilot requires one node and two nonempty splits")
    return {nodes[0]: {"train": torch.arange(train_size),
                       "validation": torch.arange(train_size, len(dataset))}}


def source_state(mode):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=ROOT, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, text=True,
        ).strip()

    status = git("status", "--porcelain", "--untracked-files=normal")
    if status and mode == "pilot":
        raise RuntimeError("Pilot requires a clean, pinned checkout")
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(status), "upstream_commit": SPEC["upstream_commit"]}


def environment():
    cpu = platform.processor() or platform.machine()
    if sys.platform == "linux":
        rows = Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
        cpu = next((row.split(":", 1)[1].strip()
                    for row in rows if row.startswith("model name")), cpu)
    return {
        "python": platform.python_version(), "os": platform.system(), "cpu_model": cpu,
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
    }


def peak_working_set():
    if sys.platform != "linux":
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def run_cell(args, out, progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(SPEC["torch_cpu_threads"])
    torch.use_deterministic_algorithms(True)
    cohort, test_pixels, test_labels = study_inputs(args.mode, args.dataset_dir)
    train_size = len(cohort.train_ids)
    epochs = 1 if args.mode == "smoke" else SPEC["epochs"]
    model_class = DenseDigitClassifier if args.arch == "dense" else SparseDigitClassifier
    trainer_args = {
        "training_batch_size": SPEC["batch_size"], "validation_batch_size": 256,
        "epochs": epochs, "optimizer": SPEC["optimizer"], "lr": SPEC["learning_rate"],
        "seed": args.seed, "verbose": False,
    }
    cfg = load_experiment_from_yamls(
        [HERE / "configs" / "pilot.yml"], name="engine",
        experiments_directory_root=str(out), seed=args.seed,
        overrides={
            "dataset": {"train": {"arguments": {"mode": args.mode, "condition": args.data}},
                        "test": {"arguments": {"mode": args.mode}}},
            "partitioning": {"arguments": {"train_size": train_size}},
            "client": {"training_engine": {"arguments": trainer_args},
                       "local_model": {"target": model_class.__name__}},
        },
    )
    sim_args = SimulationArguments(
        gpus=[], torch_num_threads=SPEC["torch_cpu_threads"], verbosity=VerbosityLevel.silent,
        seed=args.seed, resume_checkpoints=False, dataset_directory=args.dataset_dir,
        sanity_check=False, output_directory=out, dump_stdout=False,
    )
    engine = load_engine(sim_args, cfg, verbose=False)
    client = next(iter(engine.clients.values()))
    metrics, = client.callback._callbacks
    if not isinstance(metrics, PilotMetrics):
        raise RuntimeError("dEngine did not wire the expected metrics callback")
    metrics.progress = progress
    metrics.training_examples = train_size

    model = client.model
    initial = {key: array_digest(value.detach().numpy())
               for key, value in model.state_dict().items() if key.startswith(("features.", "head."))}
    progress.update(2, "training and validation")
    engine.run()

    progress.update(4, "writing and checking outputs", "archiving predictions, metrics and weights")
    pixels = cohort.clean if args.data == "clean" else cohort.proxy
    changed = np.any(cohort.clean[:train_size] != pixels[:train_size], axis=(1, 2))
    counts = routing_counts(evaluate(
        model, tensor_dataset(cohort.proxy[cohort.proxy_positions],
                              cohort.labels[cohort.proxy_positions]))[2])
    total_macs = 28 * 28 * 8 * 9 + 14 * 14 * 16 * 8 * 9 + 784 * 64 + 64 * 10 + 2 * 64 * 64
    report = {
        "train": {"examples": train_size, "modified_input_count": int(changed.sum()),
                  "inputs_sha256": array_digest(pixels[:train_size])},
        "validation": metrics.validation, "test": metrics.test,
        "diagnostic": {"assignment_counts": counts, "examples": len(cohort.pairs)},
        "model": {"parameter_count": sum(p.numel() for p in model.parameters()),
                  "active_macs_per_image": total_macs + (4 * 64 if args.arch == "moe" else 0),
                  "shared_initial_weight_digests": initial},
        "provenance": {
            "train_ids_sha256": array_digest(cohort.train_ids),
            "val_ids_sha256": array_digest(cohort.val_ids),
            "train_val_labels_sha256": array_digest(cohort.labels),
            "validation_pixels_sha256": array_digest(pixels[train_size:]),
            "test_pixels_sha256": array_digest(test_pixels),
            "test_labels_sha256": array_digest(test_labels),
            "proxy_pairs_sha256": array_digest(cohort.pairs),
        },
    }
    write_json(out / "summary.json", report)
    np.savez_compressed(out / "cohort.npz", train_ids=cohort.train_ids,
                        val_ids=cohort.val_ids, proxy_pairs=cohort.pairs)
    np.savez_compressed(out / "predictions.npz", **metrics.predictions)
    # Store named float arrays instead of a pickled model object.
    np.savez_compressed(out / "weights.npz", **{
        k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "pilot"))
    parser.add_argument("--arch", required=True, choices=("dense", "moe"))
    parser.add_argument("--data", required=True, choices=("clean", "proxy"))
    parser.add_argument("--seed", required=True, type=int, choices=SPEC["seeds"])
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.dataset_dir = args.dataset_dir.resolve()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT) and args.mode == "pilot":
        raise ValueError("Use an output directory outside the pilot checkout")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    facts = {
        "schema": 1, "study": SPEC["study"], "mode": args.mode, "architecture": args.arch,
        "data_condition": args.data, "seed": args.seed, "protocol_sha256": digest(SPEC_PATH.read_bytes()),
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
    }
    write_json(out / "manifest.json", facts)
    try:
        with Progress(f"{args.mode}/{args.seed}/{args.arch}/{args.data}") as progress:
            facts["source"] = source_state(args.mode)
            run_cell(args, out, progress)
            facts.update({
                "status": "complete", "environment": environment(), "native_exit": 0,
                "elapsed_s": time.monotonic() - started, "peak_rss_bytes": peak_working_set(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "files": {path.relative_to(out).as_posix(): digest(path.read_bytes())
                          for path in sorted(out.rglob("*"))
                          if path.is_file() and path.name != "manifest.json"},
            })
            write_json(out / "manifest.json", facts)
    except Exception as error:
        facts.update({"status": "failed", "error_type": type(error).__name__,
                      "elapsed_s": time.monotonic() - started,
                      "finished_utc": datetime.now(timezone.utc).isoformat()})
        write_json(out / "manifest.json", facts)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

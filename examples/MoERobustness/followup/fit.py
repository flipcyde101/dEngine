"""Fit one planned cell through dEngine using only its development subset."""

import argparse
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler

from _dengine_cli_paths import cli_input_path, cli_output_path, contained_file

from dengine import load_experiment_from_yamls
from dengine.bin.args_parser import SimulationArguments, VerbosityLevel
from dengine.bin.simulation import load_engine
from dengine.dataset import SupervisedDataset
from dengine.dataset.decorators import register_dataset
from dengine.models.decorators import register_model
from dengine.training_strategies.decorators import register_local_training

from ..data import SPEC as PILOT_SPEC, array_digest
from ..metrics import PilotMetrics, PilotTrainingEngine
from ..models import DenseDigitClassifier, DigitClassifier, SparseDigitClassifier
from ..run import environment, peak_working_set, source_state, tensor_dataset
from .cohort import proxy_pixels
from .design import (ARMS, PROTOCOL_PATH, SPEC, canonical_json, case_id, digest, file_digest,
                     read_json, starts_for, write_json)
from .verify import PARAMETERS


PILOT_YAML = Path(__file__).resolve().parents[1] / "configs" / "pilot.yml"


@register_model()
class WideDigitClassifier(DigitClassifier):
    """Dense control with one wide residual block and the original shared layers."""

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        width = SPEC["wide_inner_width"]
        self.block = nn.Sequential(nn.Linear(64, width), nn.ReLU(), nn.Linear(width, 64))

    predict = DenseDigitClassifier.predict


ARCHITECTURES = {"dense": DenseDigitClassifier, "wide": WideDigitClassifier, "moe": SparseDigitClassifier}


def weights_digest(state):
    return digest(canonical_json({key: array_digest(value.detach().cpu().numpy())
                                  for key, value in state.items()}))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


class RecordedSampler(Sampler):
    """Digest the indices actually drawn by the dEngine training loader."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.epochs = []

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        order = sha256()
        seen = set()
        for item in self.sampler:
            if item in seen:
                raise ValueError("Training sampler repeated an example")
            seen.add(item)
            order.update(int(item).to_bytes(8, "little"))
            yield item
        if seen != set(range(len(self.sampler))):
            raise ValueError("Training sampler missed an example")
        self.epochs.append({"examples": len(seen), "order_sha256": order.hexdigest()})


@register_local_training()
class RecordedTrainingEngine(PilotTrainingEngine):
    """The pilot learning algorithm with a measured data order."""

    def __init__(self, *args, order_seed, **kwargs):
        # PilotTrainingEngine adds 9127 to its RNG seed.
        super().__init__(*args, seed=order_seed - 9127, **kwargs)
        batch_sampler = self._ldr_train.batch_sampler
        self.recorded_sampler = RecordedSampler(batch_sampler.sampler)
        batch_sampler.sampler = self.recorded_sampler


@lru_cache(maxsize=1)
def subset(directory):
    if {file.name for file in directory.iterdir()} != {"cohort.json", "images.bin", "labels.bin"}:
        raise ValueError("Require only the reviewed development input files")
    meta = read_json(directory / "cohort.json")
    raw_images = contained_file(directory, "images.bin").read_bytes()
    raw_labels = contained_file(directory, "labels.bin").read_bytes()
    if digest(raw_images) != meta["images_sha256"] or digest(raw_labels) != meta["labels_sha256"]:
        raise ValueError("Development inputs differ from the frozen cohort")
    labels = np.frombuffer(raw_labels, dtype=np.uint8)
    images = np.frombuffer(raw_images, dtype=np.uint8).reshape(len(labels), 28, 28)
    train_size = len(meta["train_ids"])
    if (np.bincount(labels[:train_size], minlength=10).tolist() != [SPEC["train_per_digit"]] * 10
            or np.bincount(labels[train_size:], minlength=10).tolist() != [SPEC["validation_per_digit"]] * 10
            or len(meta["pairs"]) != SPEC["proxy_count"]):
        raise ValueError("Cohort size or label distribution differs from protocol")
    return images, labels, meta


@register_dataset("moe_followup_data")
def development_data(output_path, experiment_cfg, condition, validation_only=False) -> SupervisedDataset:
    images, labels, meta = subset(Path(output_path))
    train_size = len(meta["train_ids"])
    if validation_only:
        return tensor_dataset(images[train_size:], labels[train_size:])
    if condition == "proxy":
        ids = meta["train_ids"] + meta["validation_ids"]
        images = proxy_pixels(images, labels, ids, meta["pairs"], train_size)
        if digest(images.tobytes()) != meta["proxy_sha256"]:
            raise ValueError("Proxy bytes differ from the frozen plan")
    return tensor_dataset(images, labels)


def fit(directory, cohort_sha, slot, arm, out):
    if file_digest(directory / "cohort.json") != cohort_sha:
        raise ValueError("Cohort manifest fingerprint mismatch")
    _, _, cohort = subset(directory)
    starts = starts_for(cohort["cohort"])
    if cohort["starts"] != starts or not 1 <= slot <= len(starts):
        raise ValueError("Cohort has the wrong assigned starts")
    start = starts[slot - 1]
    label = case_id(cohort["cohort"], slot, arm)
    architecture, condition = arm.split("-")
    model_class = ARCHITECTURES[architecture]
    source = source_state("pilot")
    if SPEC["router_balance_weight"] != PILOT_SPEC["load_balance_weight"]:
        raise ValueError("Follow-up router loss does not match the pilot")
    torch.set_num_threads(PILOT_SPEC["torch_cpu_threads"])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise RuntimeError("Follow-up requires the pinned CPU runtime")

    cfg = load_experiment_from_yamls(
        [PILOT_YAML], name="engine", experiments_directory_root=str(out), seed=start["model_seed"],
        overrides={
            "dataset": {
                "train": {"target": "moe_followup_data", "arguments": {"condition": condition}},
                "test": {"target": "moe_followup_data", "arguments": {"condition": "clean", "validation_only": True}},
            },
            "partitioning": {"arguments": {"train_size": len(cohort["train_ids"])}},
            "client": {
                "training_engine": {"target": "RecordedTrainingEngine", "arguments": {
                    "training_batch_size": SPEC["batch_size"], "validation_batch_size": 256, "epochs": SPEC["epochs"],
                    "optimizer": "Adam", "lr": SPEC["adam_learning_rate"], "order_seed": start["order_seed"],
                    "verbose": False,
                }},
                "local_model": {"target": model_class.__name__},
            },
        },
    )
    args = SimulationArguments(
        gpus=[], torch_num_threads=PILOT_SPEC["torch_cpu_threads"], verbosity=VerbosityLevel.silent,
        seed=start["model_seed"], resume_checkpoints=False, dataset_directory=directory,
        sanity_check=False, output_directory=out, dump_stdout=False,
    )
    model_start = time.monotonic()
    engine = load_engine(args, cfg, verbose=False)
    client = next(iter(engine.clients.values()))
    trainer = client.local_training_engine
    metrics, = client.callback._callbacks
    if not isinstance(trainer, RecordedTrainingEngine) or not isinstance(metrics, PilotMetrics):
        raise RuntimeError("dEngine did not register the configured components")
    model = client.model
    params = parameter_count(model)
    if params != PARAMETERS[architecture]:
        raise ValueError("Unexpected model size")
    state = model.state_dict()
    shared = weights_digest({key: value for key, value in state.items() if key.startswith(("features.", "head."))})
    initial = weights_digest(state)
    del state
    engine.run()
    order = trainer.recorded_sampler.epochs
    if len(order) != SPEC["epochs"] or metrics.validation is None:
        raise ValueError("The measured training ledger is incomplete")
    final_state = model.state_dict()
    if not all(torch.isfinite(value).all() for value in final_state.values()):
        raise ValueError("Nonfinite model weights")
    weights = out / "weights.npz"
    with weights.open("xb") as stream:
        np.savez_compressed(stream, **{key: value.detach().cpu().numpy() for key, value in final_state.items()})
    files = {name: file_digest(out / name) for name in ("weights.npz", "metrics.jsonl", "engine/config.yaml")}
    readback = environment()
    write_json(out / "case.json", {
        "case": label, "source": source, "protocol_sha256": file_digest(PROTOCOL_PATH),
        "cohort_sha256": cohort_sha, "cohort": cohort["cohort"], "slot": slot,
        "model_seed": start["model_seed"], "order_seed": start["order_seed"], "arm": arm,
        "initial_shared": shared, "initial_model": initial, "batch_order": order,
        "final_model": weights_digest(final_state), "parameters": params,
        "validation": metrics.validation, "files": files, "elapsed_s": time.monotonic() - model_start,
        "runtime": {key: readback[key] for key in ("python", "os", "torch", "cpu_model", "threads")},
        "peak_memory_bytes": peak_working_set(), "native_exit": 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=cli_input_path)
    parser.add_argument("cohort_sha")
    parser.add_argument("slot", type=int)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("out", type=cli_output_path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    fit(args.directory, args.cohort_sha, args.slot, args.arm, args.out)


if __name__ == "__main__":
    main()

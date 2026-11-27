"""Record and verify dEngine run setup artifacts."""

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import yaml

from dengine.bin.args_parser import SimulationArguments
from dengine.config import ExperimentConfiguration
from dengine.config.constants import GRAPH_DIR_NAME, PARTITIONS_DIR_NAME, TRAIN_PARTITIONS_NAME
from dengine.dataset import SupervisedDataset
from dengine.graph import Graph
from dengine.partitioning import TYPE_DATASET_PARTITIONING


MANIFEST_NAME = "run-manifest.json"
SCHEMA_VERSION = 1
CONFIG_NAME = "config.yaml"
INDICES_NAME = f"{PARTITIONS_DIR_NAME}/indices.json"
RUNTIME_PACKAGES = (
    "dengine", "matplotlib", "networkx", "numpy", "omegaconf", "pandas", "pillow",
    "pydantic", "pyyaml", "scikit-learn", "torch", "torchvision"
)
PLOT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".svg"}
SETUP_FIELDS = ("seed", "configuration_sha256", "source", "runtime", "datasets", "graph_nodes", "graph_artifacts")
RUNTIME_FLAGS = ("torch_cuda_available", "torch_deterministic_algorithms", "torch_cudnn_deterministic",
                 "torch_cudnn_benchmark", "resume_checkpoints")


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _artifact_path(root: Path, name: str) -> Path:
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts) or "\\" in name or ":" in name:
        raise ValueError(f"Invalid artifact path: {name}")
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Artifact is missing or outside the run directory: {name}")
    return path


def _artifact_record(root: Path, name: str) -> dict:
    path = _artifact_path(root, name)
    return {"sha256": _file_digest(path), "bytes": path.stat().st_size}


def _configuration_identity(path: Path) -> tuple[int, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    seed = config.get("seed") if isinstance(config, dict) else None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"Invalid resolved configuration: {path}")
    config.pop("experiments_directory_root", None)
    canonical = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return seed, hashlib.sha256(canonical).hexdigest()


def _git_provenance(package_root: Path) -> tuple[str | None, bool | None]:
    repository = package_root.parent
    if not (repository / ".git").exists():
        return None, None

    def read_git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repository, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", timeout=3
        ).stdout.strip()

    try:
        return read_git("rev-parse", "HEAD"), bool(read_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None, None


def _component_sources(components: list[tuple[str, object]]) -> list[dict]:
    records = []
    seen = set()
    file_digests: dict[Path, str] = {}
    for target, implementation in components:
        key = target, id(implementation)
        if key in seen:
            continue
        seen.add(key)
        checksum = None
        try:
            source_file = inspect.getsourcefile(implementation)
            if source_file:
                path = Path(source_file).resolve()
                if path not in file_digests:
                    file_digests[path] = _file_digest(path)
                checksum = file_digests[path]
        except (TypeError, OSError):
            pass
        name = f"{implementation.__module__}.{implementation.__qualname__}"
        records.append({"target": target, "implementation": name, "sha256": checksum})
    return sorted(records, key=lambda value: (value["target"], value["implementation"], value["sha256"] or ""))


def _source_record(components: list[tuple[str, object]]) -> dict:
    package_root = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        name = path.relative_to(package_root).as_posix()
        hasher.update(f"{name}\0{_file_digest(path)}\n".encode("utf-8"))
    revision, dirty = _git_provenance(package_root)
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "package_sha256": hasher.hexdigest(),
        "components": _component_sources(components),
    }


def _runtime_record(args: SimulationArguments) -> dict:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_num_threads": torch.get_num_threads(),
        "requested_gpu_ids": args.gpus,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
        "resume_checkpoints": args.resume_checkpoints,
    }


def _tensor_record(values: torch.Tensor) -> dict | None:
    if not isinstance(values, torch.Tensor) or values.layout != torch.strided or values.is_quantized:
        return None
    data = values.detach().resolve_conj().resolve_neg().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256(data.view(torch.uint8).numpy()).hexdigest()
    return {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": digest}


def _dataset_record(dataset: SupervisedDataset) -> dict:
    return {
        "samples": len(dataset),
        "data": _tensor_record(dataset.data),
        "targets": _tensor_record(dataset.targets),
    }


def _write_json(path: Path, value: dict):
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_run_manifest(
    args: SimulationArguments,
    config: ExperimentConfiguration,
    graph: Graph,
    training_data: SupervisedDataset,
    test_data: SupervisedDataset,
    partitions: TYPE_DATASET_PARTITIONING,
    artifacts: list[Path],
    components: list[tuple[str, object]],
) -> Path:
    """Save the prepared inputs after dEngine finishes loading the engine."""
    output = config.output_directory
    root = output.resolve()
    indices_path = output / INDICES_NAME
    _write_json(indices_path, {
        node: {split: indices.tolist() for split, indices in splits.items()}
        for node, splits in partitions.items()
    })
    graph_artifacts = [
        path for path in config.graph_output_directory.rglob("*")
        if path.is_file() and path.suffix.lower() not in PLOT_EXTENSIONS
    ]
    graph_nodes = list(graph.nodes)
    if not graph_artifacts:
        topology = config.graph_output_directory / "topology.json"
        topology.parent.mkdir(parents=True, exist_ok=True)
        _write_json(topology, {"adjacency": [
            [str(node), [[str(peer), float(graph.get_weight(node, peer))] for peer in graph.neighbors(node)]]
            for node in graph_nodes
        ]})
        graph_artifacts = [topology]
    names = sorted(path.relative_to(output).as_posix() for path in [*artifacts, *graph_artifacts, indices_path])
    seed, config_digest = _configuration_identity(_artifact_path(root, CONFIG_NAME))
    record = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "configuration_sha256": config_digest,
        "source": _source_record(components),
        "runtime": _runtime_record(args),
        "datasets": {"train": _dataset_record(training_data), "test": _dataset_record(test_data)},
        "graph_nodes": [str(node) for node in graph_nodes],
        "graph_artifacts": sorted(path.relative_to(output).as_posix() for path in graph_artifacts),
        "artifacts": {name: _artifact_record(root, name) for name in names},
    }
    manifest = root / MANIFEST_NAME
    _write_json(manifest, record)
    return manifest


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _verify_source(source, manifest: Path):
    if (
        not isinstance(source, dict)
        or not {"git_revision", "git_dirty", "package_sha256", "components"} <= source.keys()
        or not _is_sha256(source["package_sha256"])
        or not isinstance(source["git_revision"], (str, type(None)))
        or not isinstance(source["git_dirty"], (bool, type(None)))
        or not isinstance(source["components"], list)
        or not source["components"]
    ):
        raise ValueError(f"Invalid source metadata in {manifest}")
    for entry in source["components"]:
        if (
            not isinstance(entry, dict) or not {"target", "implementation", "sha256"} <= entry.keys()
            or not all(isinstance(entry[key], str) and entry[key] for key in ("target", "implementation"))
            or (entry["sha256"] is not None and not _is_sha256(entry["sha256"]))
        ):
            raise ValueError(f"Invalid component source metadata in {manifest}")


def _verify_runtime(runtime, manifest: Path):
    if not isinstance(runtime, dict):
        raise ValueError(f"Invalid runtime metadata in {manifest}")
    versions = runtime.get("packages")
    threads = runtime.get("torch_num_threads")
    gpu_ids = runtime.get("requested_gpu_ids")
    if (
        not all(isinstance(runtime.get(key), str) and runtime[key]
                for key in ("python", "python_implementation", "system", "machine"))
        or not isinstance(versions, dict) or set(versions) != set(RUNTIME_PACKAGES)
        or not all(value is None or (isinstance(value, str) and value) for value in versions.values())
        or not isinstance(threads, int) or isinstance(threads, bool) or threads < 1
        or not isinstance(gpu_ids, list) or not all(isinstance(gpu, int) and not isinstance(gpu, bool) for gpu in gpu_ids)
        or "torch_cuda_version" not in runtime or not isinstance(runtime["torch_cuda_version"], (str, type(None)))
        or not all(isinstance(runtime.get(key), bool) for key in RUNTIME_FLAGS)
    ):
        raise ValueError(f"Invalid runtime metadata in {manifest}")


def _verify_metadata(record: dict, manifest: Path):
    if not all(field in record for field in SETUP_FIELDS):
        raise ValueError(f"Incomplete run metadata in {manifest}")
    _verify_source(record["source"], manifest)
    _verify_runtime(record["runtime"], manifest)
    if not isinstance(record["graph_nodes"], list) or not all(isinstance(node, str) for node in record["graph_nodes"]):
        raise ValueError(f"Invalid graph metadata in {manifest}")
    graph_artifacts = record["graph_artifacts"]
    if not isinstance(graph_artifacts, list) or not graph_artifacts or not all(
        isinstance(name, str) and name.startswith(f"{GRAPH_DIR_NAME}/") and name in record["artifacts"]
        for name in graph_artifacts
    ):
        raise ValueError(f"Missing graph topology artifacts in {manifest}")

    datasets = record["datasets"]
    if not isinstance(datasets, dict) or datasets.keys() != {"train", "test"}:
        raise ValueError(f"Incomplete dataset metadata in {manifest}")
    for split, dataset in datasets.items():
        if not isinstance(dataset, dict) or not {"samples", "data", "targets"} <= dataset.keys():
            raise ValueError(f"Incomplete {split} dataset metadata in {manifest}")
        for tensor in (dataset["data"], dataset["targets"]):
            if tensor is not None and (
                not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list)
                or not isinstance(tensor.get("dtype"), str) or not _is_sha256(tensor.get("sha256"))
            ):
                raise ValueError(f"Invalid {split} dataset fingerprint in {manifest}")


def verify_run_manifest(manifest: Path) -> dict:
    """Verify the recorded files relative to the manifest directory."""
    manifest = Path(manifest)
    root = manifest.parent.resolve()
    record = json.loads(_artifact_path(root, manifest.name).read_text(encoding="utf-8"))
    schema = record.get("schema_version") if isinstance(record, dict) else None
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise ValueError(f"Missing or invalid manifest schema version in {manifest}")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema version in {manifest}")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or not {CONFIG_NAME, TRAIN_PARTITIONS_NAME, INDICES_NAME} <= artifacts.keys():
        raise ValueError(f"Incomplete artifact list in {manifest}")
    _verify_metadata(record, manifest)
    for name, expected in artifacts.items():
        if _artifact_record(root, name) != expected:
            raise ValueError(f"Artifact fingerprint differs: {name}")
    seed, config_digest = _configuration_identity(root / CONFIG_NAME)
    if seed != record["seed"] or config_digest != record["configuration_sha256"]:
        raise ValueError(f"Configuration metadata differs in {manifest}")
    return record


def _replay_artifacts(record: dict) -> dict:
    return {name: info for name, info in record["artifacts"].items() if name != CONFIG_NAME}


def compare_run_manifests(original: Path, replay: Path) -> None:
    """Verify both manifests and check that the replay prepared the same inputs."""
    left, right = verify_run_manifest(original), verify_run_manifest(replay)
    differences = [field for field in SETUP_FIELDS if left[field] != right[field]]
    if differences:
        raise ValueError(f"Replay differs in {', '.join(differences)}")
    if any(component["sha256"] is None for component in left["source"]["components"]):
        raise ValueError("Replay comparison requires source fingerprints for the loaded components")
    for dataset in left["datasets"].values():
        if dataset["data"] is None or dataset["targets"] is None:
            raise ValueError("Replay comparison requires fingerprints for both datasets")
    if _replay_artifacts(left) != _replay_artifacts(right):
        raise ValueError("Replay differs in graph or partition artifacts")


def main():
    from _dengine_cli_paths import cli_input_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=cli_input_path)
    parser.add_argument("--compare", type=cli_input_path, help="Verify and compare a replay manifest")
    args = parser.parse_args()
    try:
        if args.compare:
            compare_run_manifests(args.manifest, args.compare)
            print("Replay setup matches the original manifest.")
        else:
            verify_run_manifest(args.manifest)
            print("Run manifest artifacts verified.")
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

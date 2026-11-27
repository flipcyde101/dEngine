import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn

from dengine import load_experiment_from_yamls
from dengine.bin.args_parser import SimulationArguments, VerbosityLevel
from dengine.bin.simulation import _cli, load_engine
from dengine.dataset import SupervisedDataset
from dengine.dataset.decorators import register_dataset
from dengine.graph.decorators import register_graph
from dengine.graph.nx_graph import NXGraph
from dengine.models.decorators import BUILTIN_MODELS, ModuleBase, register_model


@register_dataset("manifest_test_images")
def manifest_images(output_path, split, **kwargs) -> SupervisedDataset:
    offset_file = Path(output_path) / "offset.txt"
    offset = float(offset_file.read_text()) if offset_file.exists() else 0
    pixels = torch.arange(48, dtype=torch.float32).reshape(12, 1, 2, 2) / 48
    data = pixels + offset + (0.25 if split == "test" else 0)
    return SupervisedDataset(data, torch.arange(12) % 2)


@register_model("ManifestTestClassifier")
class ManifestTestClassifier(ModuleBase):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    def forward(self, images):
        return self.layers(images)


@register_graph()
class ManifestGraphWithoutArtifacts(NXGraph):
    def dump(self):
        pass


@pytest.fixture
def run_experiment(tmp_path, monkeypatch):
    config = tmp_path / "tiny.yml"
    config.write_text(yaml.safe_dump({
        "name": "manifest-demo",
        "seed": 417,
        "dataset": {
            "train": {"target": "manifest_test_images", "arguments": {"split": "train"}},
            "test": {"target": "manifest_test_images", "arguments": {"split": "test"}},
        },
        "graph": {"target": "NXGraph", "arguments": {"nx_class": "complete_graph", "n": 1}},
        "partitioning": {"target": "random_iid", "arguments": {"validation_percentage": 0.25}},
        "scenario": {"target": "CentralizedScenarioEngine"},
        "client": {
            "target": "CentralizedClient",
            "local_model": {"target": "ManifestTestClassifier"},
            "training_engine": {
                "target": "EngineWithoutEarlyStopping",
                "arguments": {"training_batch_size": 4, "validation_batch_size": 4, "epochs": 1,
                              "optimizer": "SGD", "lr": 0.01, "verbose": False},
            },
        },
    }), encoding="utf-8")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PYTHONHASHSEED", "417")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    existing_figures = set(plt.get_fignums())

    def run(name, resolved_config=None, record=True):
        output = tmp_path / name
        command = [
            "dengine-simulate", "--config", str(resolved_config or config),
            "--output_directory", str(output), "--dataset_directory", str(datasets), "--seed", "19",
            "--gpu", "-1",
        ]
        if record:
            command.append("--run_manifest")
        monkeypatch.setattr(sys, "argv", command)
        _cli()
        return output / "manifest-demo"

    yield run, datasets
    for figure in set(plt.get_fignums()) - existing_figures:
        plt.close(figure)


def test_simulation_writes_a_manifest_for_its_effective_inputs(run_experiment):
    run, _ = run_experiment
    output = run("original")
    manifest = output / "run-manifest.json"
    assert manifest.is_file()
    record = json.loads(manifest.read_text(encoding="utf-8"))

    assert record["schema_version"] == 1
    assert record["seed"] == 417
    assert record["runtime"]["torch_num_threads"] == 1
    assert record["graph_nodes"] == ["0"]
    assert record["datasets"]["train"]["data"]["shape"] == [12, 1, 2, 2]
    assert record["datasets"]["train"]["data"]["sha256"] != record["datasets"]["test"]["data"]["sha256"]
    assert len(record["source"]["package_sha256"]) == 64
    assert {"ManifestTestClassifier", "manifest_test_images"} <= {
        component["target"] for component in record["source"]["components"] if component["sha256"]
    }

    expected_files = {"config.yaml", "networkx/graph.edgelist", "partitioning/train_partitions.csv",
                      "partitioning/0.random_iid.csv", "partitioning/indices.json"}
    assert expected_files <= record["artifacts"].keys()
    for name in expected_files:
        content = (output / name).read_bytes()
        assert record["artifacts"][name]["sha256"] == hashlib.sha256(content).hexdigest()

    splits = json.loads((output / "partitioning" / "indices.json").read_bytes())
    train = splits["0"]["train"]
    validation = splits["0"]["validation"]
    assert len(train) == 9 and len(validation) == 3
    assert sorted(train + validation) == list(range(12))


def test_replay_from_resolved_configuration_matches(run_experiment):
    run, _ = run_experiment
    original = run("original")
    replay = run("replay", original / "config.yaml")
    first = original / "run-manifest.json"
    second = replay / "run-manifest.json"
    assert first.is_file() and second.is_file()
    from dengine.run_manifest import compare_run_manifests

    assert (original / "config.yaml").read_bytes() != (replay / "config.yaml").read_bytes()
    compare_run_manifests(first, second)


def test_replay_detects_changed_data_with_unchanged_labels_and_configuration(run_experiment):
    run, datasets = run_experiment
    original = run("original")
    (datasets / "offset.txt").write_text("0.5", encoding="utf-8")
    replay = run("replay", original / "config.yaml")
    first = original / "run-manifest.json"
    second = replay / "run-manifest.json"
    assert first.is_file() and second.is_file()
    from dengine.run_manifest import compare_run_manifests

    with pytest.raises(ValueError, match="datasets"):
        compare_run_manifests(first, second)


def test_replay_detects_changed_registered_model_source(run_experiment, tmp_path, monkeypatch):
    run, _ = run_experiment
    model_path = tmp_path / "external_model.py"

    def register_external_model(scale):
        source = (
            "from torch import nn\n"
            "from dengine.models.decorators import ModuleBase\n"
            "class ExternalClassifier(ModuleBase):\n"
            "    def __init__(self, dataset, **kwargs):\n"
            "        super().__init__(dataset, **kwargs)\n"
            "        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))\n"
            "    def forward(self, inputs):\n"
            f"        return self.layers(inputs) * {scale}\n"
        )
        model_path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("manifest_external_model", model_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        spec.loader.exec_module(module)
        monkeypatch.setitem(BUILTIN_MODELS, "ManifestTestClassifier", module.ExternalClassifier)

    register_external_model(1)
    original = run("original")
    register_external_model(0.25)
    replay = run("replay", original / "config.yaml")
    from dengine.run_manifest import compare_run_manifests

    with pytest.raises(ValueError, match="source"):
        compare_run_manifests(original / "run-manifest.json", replay / "run-manifest.json")


def test_verification_detects_changed_split_indices(run_experiment):
    run, _ = run_experiment
    output = run("original")
    manifest = output / "run-manifest.json"
    assert manifest.is_file()
    from dengine.run_manifest import verify_run_manifest

    (output / "partitioning" / "indices.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="partitioning/indices.json"):
        verify_run_manifest(manifest)


def test_replay_detects_different_split_order_with_valid_artifact_hashes(run_experiment):
    run, _ = run_experiment
    original = run("original")
    replay = run("replay", original / "config.yaml")
    from dengine.run_manifest import compare_run_manifests

    path = replay / "partitioning" / "indices.json"
    splits = json.loads(path.read_bytes())
    splits["0"]["train"].reverse()
    path.write_text(json.dumps(splits), encoding="utf-8")
    content = path.read_bytes()
    manifest = replay / "run-manifest.json"
    record = json.loads(manifest.read_bytes())
    record["artifacts"]["partitioning/indices.json"] = {
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
    }
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="partition"):
        compare_run_manifests(original / "run-manifest.json", manifest)


def test_verification_rejects_incomplete_dataset_metadata(run_experiment):
    run, _ = run_experiment
    manifest = run("original") / "run-manifest.json"
    from dengine.run_manifest import verify_run_manifest

    record = json.loads(manifest.read_bytes())
    del record["datasets"]["train"]["data"]
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset"):
        verify_run_manifest(manifest)


@pytest.mark.parametrize("field", ["source", "runtime"])
def test_verification_rejects_empty_provenance_metadata(run_experiment, field):
    run, _ = run_experiment
    manifest = run("original") / "run-manifest.json"
    from dengine.run_manifest import verify_run_manifest

    record = json.loads(manifest.read_bytes())
    record[field] = {}
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        verify_run_manifest(manifest)


def test_verification_requires_saved_topology_even_for_an_empty_edge_list(run_experiment):
    run, _ = run_experiment
    output = run("original")
    manifest = output / "run-manifest.json"
    from dengine.run_manifest import verify_run_manifest

    record = verify_run_manifest(manifest)
    assert record["artifacts"]["networkx/graph.edgelist"]["bytes"] == 0
    del record["artifacts"]["networkx/graph.edgelist"]
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="graph"):
        verify_run_manifest(manifest)


def test_graph_without_dumped_topology_records_its_adjacency(run_experiment, tmp_path):
    run, _ = run_experiment
    config = tmp_path / "tiny.yml"
    content = yaml.safe_load(config.read_text(encoding="utf-8"))
    content["graph"]["target"] = "ManifestGraphWithoutArtifacts"
    config.write_text(yaml.safe_dump(content), encoding="utf-8")
    output = run("original")
    from dengine.run_manifest import verify_run_manifest

    record = verify_run_manifest(output / "run-manifest.json")
    assert record["graph_artifacts"] == ["networkx/topology.json"]
    assert json.loads((output / "networkx" / "topology.json").read_bytes()) == {"adjacency": [["0", []]]}


def test_verification_rejects_artifacts_outside_the_run(run_experiment):
    run, _ = run_experiment
    output = run("original")
    manifest = output / "run-manifest.json"
    from dengine.run_manifest import verify_run_manifest

    outside = output.parent / "unrelated.txt"
    outside.write_bytes(b"outside")
    record = json.loads(manifest.read_bytes())
    record["artifacts"]["../unrelated.txt"] = {
        "sha256": hashlib.sha256(b"outside").hexdigest(), "bytes": 7,
    }
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact path"):
        verify_run_manifest(manifest)


def test_artifact_names_use_portable_separators(run_experiment):
    run, _ = run_experiment
    output = run("original")
    manifest = output / "run-manifest.json"
    from dengine.run_manifest import verify_run_manifest

    alias = r"partitioning\indices.json"
    if not (output / alias).exists():
        (output / alias).write_bytes((output / "partitioning" / "indices.json").read_bytes())
    record = json.loads(manifest.read_bytes())
    record["artifacts"][alias] = record["artifacts"]["partitioning/indices.json"]
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact path"):
        verify_run_manifest(manifest)


def test_manifest_reports_a_missing_artifact(run_experiment):
    run, _ = run_experiment
    output = run("original")
    from dengine.run_manifest import verify_run_manifest

    (output / "partitioning" / "indices.json").unlink()
    with pytest.raises(ValueError, match="missing.*partitioning/indices.json"):
        verify_run_manifest(output / "run-manifest.json")


def test_manifest_rejects_an_artifact_link_to_another_directory(run_experiment):
    run, _ = run_experiment
    output = run("original")
    from dengine.run_manifest import verify_run_manifest

    artifact = output / "partitioning" / "indices.json"
    outside = output.parent / "indices-external.json"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(outside)
    except OSError as error:
        if getattr(error, "winerror", None) != 1314:
            raise
        pytest.skip("Windows symlink permission is unavailable")
    try:
        with pytest.raises(ValueError, match="outside.*partitioning/indices.json"):
            verify_run_manifest(output / "run-manifest.json")
    finally:
        artifact.unlink()


def test_manifest_recorded_run_cannot_overwrite_an_existing_run(run_experiment):
    run, _ = run_experiment
    output = run("original")
    config = (output / "config.yaml").read_bytes()
    with pytest.raises(FileExistsError, match="fresh output directory"):
        run("original")
    assert (output / "config.yaml").read_bytes() == config


def test_existing_simulation_without_manifest_retains_saved_artifacts(run_experiment):
    run, _ = run_experiment
    output = run("legacy", record=False)
    run("legacy", record=False)
    assert (output / "config.yaml").is_file()
    assert (output / "partitioning" / "train_partitions.csv").is_file()
    assert not (output / "run-manifest.json").exists()


def test_programmatic_engine_load_requires_opt_in_to_record_a_manifest(run_experiment, tmp_path):
    _, datasets = run_experiment
    args = SimulationArguments(
        gpus=[-1], torch_num_threads=1, verbosity=VerbosityLevel.silent, seed=417,
        resume_checkpoints=False, dataset_directory=datasets, sanity_check=False,
        output_directory=tmp_path / "api", dump_stdout=False,
    )
    config = load_experiment_from_yamls(
        [tmp_path / "tiny.yml"], experiments_directory_root=str(args.output_directory), seed=args.seed,
    )
    load_engine(args, config, verbose=False)
    output = args.output_directory / "manifest-demo"
    assert (output / "config.yaml").is_file()
    assert not (output / "run-manifest.json").exists()

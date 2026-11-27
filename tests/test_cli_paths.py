from argparse import ArgumentTypeError
from io import BytesIO
import json
from os import pathsep
import tarfile
from types import SimpleNamespace

import pytest

from _dengine_cli_paths import cli_input_path, cli_output_path


def test_workspace_defaults_to_working_directory(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("DENGINE_INPUT_ROOTS", raising=False)
    monkeypatch.delenv("DENGINE_OUTPUT_ROOTS", raising=False)

    assert cli_input_path("input.json") == workspace / "input.json"
    assert cli_output_path("report") == workspace / "report"
    with pytest.raises(ArgumentTypeError, match="outside"):
        cli_input_path(tmp_path / "unrelated.json")
    with pytest.raises(ArgumentTypeError, match="outside"):
        cli_output_path(workspace)


def test_read_and_write_workspaces_are_independent(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    exports = tmp_path / "exports"
    for directory in (inputs, outputs, exports):
        directory.mkdir()
    monkeypatch.setenv("DENGINE_INPUT_ROOTS", pathsep.join(map(str, (inputs, exports))))
    monkeypatch.setenv("DENGINE_OUTPUT_ROOTS", str(outputs))

    assert cli_input_path(inputs) == inputs
    assert cli_input_path(exports / "scores.json") == exports / "scores.json"
    assert cli_output_path(outputs / "report") == outputs / "report"
    with pytest.raises(ArgumentTypeError, match="outside"):
        cli_output_path(inputs / "report")
    with pytest.raises(ArgumentTypeError, match="outside"):
        cli_input_path(outputs / "report")


def test_worker_archive_contains_portable_regular_entries(tmp_path):
    from examples.MoERobustness.lab.worker import compressed_bundle

    nested = tmp_path / "case-1"
    nested.mkdir()
    (nested / "worker.json").write_bytes(b"{}")
    packed = compressed_bundle(tmp_path)
    with tarfile.open(fileobj=BytesIO(packed), mode="r:gz") as archive:
        assert archive.getnames() == ["case-1/worker.json"]
        assert archive.getmembers()[0].isreg()
        with archive.extractfile("case-1/worker.json") as member:
            assert member.read() == b"{}"


def test_workspace_worker_launch_uses_fixed_module(monkeypatch, tmp_path):
    from examples.MoERobustness.lab import worker

    observed = []

    def record_call(argv, **kwargs):
        observed.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", record_call)
    destination = tmp_path / "pilot outputs"
    source = tmp_path / "public inputs"
    assert worker.cell("smoke", 11, "moe", "proxy", destination, source)["native_exit"] == 0
    argv, settings = observed[0]
    assert argv == [worker.sys.executable, "-Bu", "-m", "examples.MoERobustness.lab.invoke", "pilot"]
    assert json.loads(settings["input"])[-2:] == [f"--dataset-dir={source}", f"--output-dir={destination}"]
    assert settings["shell"] is False
    assert settings["check"] is False
    with pytest.raises(TypeError, match="Invalid worker task"):
        worker.run_process("unrecognized-task", None, [], 120)
    assert len(observed) == 1

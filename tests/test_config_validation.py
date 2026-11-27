import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples" / "HelloWorld" / "config.yml"


def test_module_validates_from_a_custom_project_without_importing_it(tmp_path):
    project = tmp_path / "custom_project"
    project.mkdir()
    (project / "__init__.py").write_text(
        "raise RuntimeError('Local project code was imported')\n", encoding="utf-8"
    )
    output = tmp_path / "must-not-write"
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY), DENGINE_INPUT_ROOTS=str(REPOSITORY))
    result = subprocess.run(
        [sys.executable, "-B", "-m", "dengine.bin.validate_config",
         "--config", str(EXAMPLE), "--output_directory", str(output)],
        cwd=project, env=environment, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Configuration schema is valid" in result.stdout
    assert not output.exists()
    assert sorted(item.name for item in project.iterdir()) == ["__init__.py"]


@pytest.fixture(autouse=True)
def config_workspaces(tmp_path, monkeypatch):
    monkeypatch.setenv("DENGINE_INPUT_ROOTS", os.pathsep.join((str(REPOSITORY), str(tmp_path))))


def test_layers_and_overrides_use_the_simulation_schema(tmp_path, capsys):
    from dengine.bin.validate_config import main

    overlay = tmp_path / "overlay.yml"
    overlay.write_text("seed: invalid\n", encoding="utf-8")
    output = tmp_path / "must-not-write"
    args = ["--config", str(EXAMPLE), str(overlay), "--output_directory", str(output)]
    assert main(args) == 1
    assert "seed" in capsys.readouterr().err

    assert main([*args, "--set", "seed=92"]) == 0
    assert "Configuration schema is valid" in capsys.readouterr().out
    assert not output.exists()


def test_schema_errors_report_the_field_without_echoing_the_input(tmp_path, capsys):
    from dengine.bin.validate_config import main

    overlay = tmp_path / "invalid-seed.yml"
    overlay.write_text("seed: PRIVATE_VALUE_FOR_TEST\n", encoding="utf-8")
    assert main(["--config", str(EXAMPLE), str(overlay)]) == 1
    result = capsys.readouterr()
    assert "seed" in result.err
    assert "PRIVATE_VALUE_FOR_TEST" not in result.err + result.out


def test_missing_config_reports_the_requested_file(tmp_path, capsys):
    from dengine.bin.validate_config import main

    missing = tmp_path / "missing.yml"
    assert main(["--config", str(missing)]) == 1
    assert "missing.yml" in capsys.readouterr().err


def test_installed_config_name_still_works_with_an_external_input_root(tmp_path, monkeypatch, capsys):
    from dengine.bin import args_parser
    from dengine.bin.validate_config import main

    workspace = tmp_path / "inputs"
    workspace.mkdir()
    monkeypatch.setenv("DENGINE_INPUT_ROOTS", str(workspace))
    monkeypatch.setattr(args_parser, "get_config", lambda name: EXAMPLE if name == "core.yml" else workspace)
    assert main(["--config", "core.yml"]) == 0
    assert "Configuration schema is valid" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("content", "diagnostic"),
    [("seed: [PRIVATE_VALUE_FOR_TEST\n", "line"),
     ("- PRIVATE_VALUE_FOR_TEST\n", "mapping"),
     ("7: PRIVATE_VALUE_FOR_TEST\n", "string keys")],
)
def test_invalid_yaml_has_a_safe_file_diagnostic(tmp_path, capsys, content, diagnostic):
    from dengine.bin.validate_config import main

    broken = tmp_path / "broken.yml"
    broken.write_text(content, encoding="utf-8")
    assert main(["--config", str(broken)]) == 1
    message = capsys.readouterr().err
    assert "broken.yml" in message
    assert diagnostic in message
    assert "PRIVATE_VALUE_FOR_TEST" not in message


def test_environment_values_are_checked_without_echoing_them(tmp_path, monkeypatch, capsys):
    from dengine.bin.validate_config import main

    environment_key = "DENGINE_CONFIG_VALIDATION_TEST_SEED"
    overlay = tmp_path / "environment.yml"
    overlay.write_text(f"seed: ${environment_key}\n", encoding="utf-8")
    args = ["--config", str(EXAMPLE), str(overlay)]

    monkeypatch.delenv(environment_key, raising=False)
    assert main(args) == 1
    assert environment_key in capsys.readouterr().err

    monkeypatch.setenv(environment_key, "'PRIVATE_VALUE_FOR_TEST")
    assert main(args) == 1
    failure = capsys.readouterr().err
    assert environment_key in failure
    assert "PRIVATE_VALUE_FOR_TEST" not in failure

    monkeypatch.setenv(environment_key, "122")
    assert main(args) == 0
    assert "Configuration schema is valid" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("args", "diagnostic"),
    [([], "--config"),
     (["--config", str(EXAMPLE), "--set", "not-a-key-value"], "KEY=VALUE")],
)
def test_invalid_invocation_is_a_usage_error(capsys, args, diagnostic):
    from dengine.bin.validate_config import main

    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert diagnostic in capsys.readouterr().err

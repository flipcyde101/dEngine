"""Limit command-line file access to directories chosen by the operator."""

from argparse import ArgumentTypeError
from os import environ, pathsep
from os.path import abspath
from pathlib import Path


def _allowed_path(value, variable):
    configured = environ.get(variable)
    entries = configured.split(pathsep) if configured is not None else [str(Path.cwd())]
    roots = []
    canonical_roots = []
    for entry in entries:
        root = Path(entry)
        if not entry or not root.is_absolute():
            raise ArgumentTypeError(f"{variable} must list absolute workspace directories")
        root = Path(abspath(root))
        canonical = root.resolve()
        if not canonical.is_dir() or canonical == canonical.parent:
            raise ArgumentTypeError(f"{variable} must list existing workspace directories below a filesystem root")
        roots.append(root)
        canonical_roots.append(canonical)
    candidate = Path(abspath(value))
    if not any(candidate.is_relative_to(root) for root in [*roots, *canonical_roots]):
        raise ArgumentTypeError(f"Path is outside the approved workspace ({variable})")
    try:
        path = candidate.resolve()
    except (OSError, ValueError) as error:
        raise ArgumentTypeError("Could not resolve the supplied workspace path") from error
    permits_root = variable == "DENGINE_INPUT_ROOTS"
    if not any(path.is_relative_to(root) and (permits_root or path != root) for root in canonical_roots):
        raise ArgumentTypeError(f"Path is outside the approved workspace ({variable})")
    return path


def cli_input_path(value):
    """Resolve a CLI input within a configured read workspace."""
    return _allowed_path(value, "DENGINE_INPUT_ROOTS")


def cli_output_path(value):
    """Resolve a CLI output below a configured write workspace."""
    candidate = Path(value)
    resolved = _allowed_path(candidate, "DENGINE_OUTPUT_ROOTS")
    if candidate.is_symlink():
        raise ArgumentTypeError("Output path must not be a symbolic link")
    return resolved


def contained_file(directory, name):
    """Return a regular file in a known directory without following it outside."""
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Required file is missing or outside its directory: {name}")
    return path

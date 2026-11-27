"""Read one bounded worker request from stdin and start its fixed entry point."""

import argparse
import json
import sys

MAX_REQUEST_BYTES = 32_768


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("pilot", "followup"))
    task = parser.parse_args().task
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError
        request = json.loads(raw)
        if (not isinstance(request, list) or len(request) > 16
                or not all(isinstance(item, str) for item in request)):
            raise ValueError
    except (UnicodeError, ValueError):
        parser.error("Expected one bounded JSON worker request")
    if task == "pilot":
        from ..run import main as run_pilot
        return run_pilot(request)
    from ..followup.fit import main as fit_cell
    return fit_cell(request)


if __name__ == "__main__":
    raise SystemExit(main())

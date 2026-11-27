"""Check the experiment configuration schema without preparing a run."""

import sys
from argparse import ArgumentParser, ArgumentTypeError

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError
from yaml import YAMLError

from dengine.bin.args_parser import resolve_config_files
from dengine.config.utils import ConfigurationInputError, load_experiment_from_yamls


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True, metavar="FILE",
                        help="YAML files or installed config names, in load order")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a YAML field using a dotted key. Repeat for multiple fields.")
    parser.add_argument("--seed", type=int, default=0, help="Default seed for schema checks")
    parser.add_argument("--output_directory", default="./logs",
                        help="Experiment output root used as a configuration value")
    args = parser.parse_args(argv)

    for assignment in args.overrides:
        key, equals, _ = assignment.partition("=")
        if not equals or not key.strip():
            parser.error("--set requires KEY=VALUE")

    try:
        files = resolve_config_files(args.config)
    except (ValueError, ArgumentTypeError) as error:
        print(error, file=sys.stderr)
        return 1

    try:
        overrides = dict(OmegaConf.from_dotlist(args.overrides))
        load_experiment_from_yamls(
            files,
            overrides=overrides,
            experiments_directory_root=args.output_directory,
            seed=args.seed,
        )
    except ConfigurationInputError as error:
        print(error, file=sys.stderr)
        return 1
    except ValidationError as error:
        print("Configuration schema errors:", file=sys.stderr)
        for issue in error.errors(include_input=False, include_context=False, include_url=False):
            location = ".".join(str(part) for part in issue["loc"])
            print(f"  {location}: {issue['msg']}", file=sys.stderr)
        return 1
    except OSError as error:
        path = f" {error.filename}" if error.filename else ""
        print(f"Could not read configuration file{path}.", file=sys.stderr)
        return 1
    except UnicodeDecodeError:
        print("Could not decode a configuration file. Check its text encoding.", file=sys.stderr)
        return 1
    except (OmegaConfBaseException, YAMLError, ValueError):
        print("Could not combine configuration values. Check the YAML and --set overrides.",
              file=sys.stderr)
        return 1

    print("Configuration schema is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# dEngine 🍋 — A Decentralized Learning Framework
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DecentralizedLearning/dEngine/blob/main/examples/MNIST/dEngine-MNIST.ipynb)

**dEngine** is a Python simulation framework for research on **decentralized, federated, and gossip-style machine learning**. It lets you define a population of clients, a communication topology, a data partitioning strategy, and a training/aggregation scenario in a single YAML file, then run the whole simulation on one machine (CPU or GPU) with PyTorch.

It's built for researchers who want to iterate quickly on collaborative-learning algorithms (federated averaging, decentralized gossip averaging, and beyond) without re-implementing the simulation plumbing (network graphs, dataset partitioning, checkpointing, metrics) for every project.

## Research tools in this fork

This is a research fork of [dEngine](https://github.com/DecentralizedLearning/dEngine), a Python framework for decentralized learning. The fork adds four capabilities for recording, checking, and analyzing experiments.

| Capability | What it does |
| --- | --- |
| Run manifests | Saves the effective configuration, seed, runtime versions, and fingerprints of inputs used to prepare a run. |
| Replay and verification | Replays the synthetic example from its saved configuration and compares recorded setup fingerprints. |
| Configuration validation | Checks a YAML configuration before datasets load or a simulation starts. |
| Paired class reports | Turns saved study scores into per-digit F1 loss comparisons with descriptive uncertainty intervals. |

### Try it from a source checkout

Use Python 3.13 in a virtual environment, then run these commands from the repository root.

```sh
python -m pip install -e .
dengine-validate --config examples/HelloWorld/config.yml
python -m examples.RunManifest.run --config examples/RunManifest/config.yml --output_directory logs/manifest-demo/first --gpu -1 --run_manifest
```

The first dEngine command checks the config. The second runs one communication round with two clients and synthetic data, then leaves a manifest in the run directory. Use a fresh output path for each run. See [how to replay and compare the setup](#record-and-verify-a-run-setup).

For the MNIST study, see the [paired report instructions](examples/MoERobustness/followup/README.md#paired-class-comparisons). The reporter reads saved study scores and writes JSON and Markdown to a fresh output directory. The class comparisons are exploratory. Digit 4 remains the study's primary outcome.

> [!NOTE]
> Manifest comparison verifies the recorded setup and available fingerprints. Matching manifests do not guarantee identical model results across machines.

## Why dEngine?

- **Config-driven experiments.** Every experiment — graph topology, dataset, partitioning, client behavior, aggregation rule — is declared in YAML and validated with [Pydantic](https://docs.pydantic.dev/) via [OmegaConf](https://omegaconf.readthedocs.io/). Swap a topology or a training strategy by pointing to a different config, no code changes required.
- **Pluggable scenarios.** Ships with reference implementations of **centralized**, **federated (FedAvg)**, **decentralized gossip averaging**, etc., plus an event-driven scenario API (`SyncEngine`, `Depends`, custom `Event`s) for building your own.
- **Realistic communication graphs.** Static and dynamic topologies out of the box (Erdős–Rényi, Barabási–Albert, complete graph, star, or your own via [NetworkX](https://networkx.org/)), including support for time-varying contact graphs.
- **Flexible data partitioning.** Built-in IID and non-IID partitioning strategies (including hub/non-hub variants) for splitting a dataset across clients.
- **Batteries included.** Reference datasets (MNIST, EMNIST, CIFAR-10/100) and models (MLP, small CNNs, ResNet), plus callbacks for checkpointing, logging, and metrics/analysis utilities (confusion matrices, partitioning plots, experiment reports).
- **Composable via `BUILTINS`.** Mix and match built-in configs (`dengine.BUILTINS.CORE...`) with your own custom config files and CLI overrides.

## Installation

dEngine requires **Python ≥ 3.13**.

```bash
pip install git+https://github.com/DecentralizedLearning/dEngine.git
```

This installs the `dengine` package along with its dependencies (PyTorch, torchvision, NetworkX, OmegaConf, Pydantic, scikit-learn, pandas, matplotlib, and others — see `pyproject.toml`).

## Quick start

The [`examples/`](./examples) directory contains runnable projects. The simplest is `examples/MNIST`, which trains a small CNN on MNIST over a Barabási–Albert graph using decentralized averaging.

A minimal experiment config looks like this:

```yaml
scenario:
  target: This_Will_Be_Overwritten
  arguments:
    max_communication_rounds: 200

graph:
  target: CustomNetworking

dataset:
  train:
    target: custom_mnist_train
  test:
    target: custom_mnist_test

partitioning:
  target: custom_partitioning
  arguments:
    validation_percentage: .2

client:
  target: CustomClient
  training_engine:
    target: CustomTrainingEngine
    arguments:
      epochs: 5
      lr: 0.001
      optimizer: SGD
      training_batch_size: 128
      validation_batch_size: 128
  local_model:
    target: CustomNet

callbacks:
- target: CustomCallback
  arguments:
    every_n_local_epochs: 5
```

You can also compose an experiment entirely from dEngine's built-in configs in code:

```python
import dengine
from dengine import load_experiment_from_yamls
from dengine.bin.simulation import run_simulation, SimulationArguments, VerbosityLevel
from dengine.config.utils import convert_to_nested_dict

args = SimulationArguments(
    gpus=[0],
    dataset_directory="datasets/",
    output_directory="logs/",
    dump_stdout=False,
    resume_checkpoints=False,
    sanity_check=False,
    seed=123,
    torch_num_threads=1,
    verbosity=VerbosityLevel.info,
)

cfg = load_experiment_from_yamls(
    [
        dengine.BUILTINS.CORE.GRAPH.BA_SMALL,
        dengine.BUILTINS.CORE.PARTITIONING.IID,
        dengine.BUILTINS.CORE.DATASETS.MNIST,
        dengine.BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
    ],
    experiments_directory_root=str(args.output_directory),
    overrides=convert_to_nested_dict({"client.training_engine.arguments.epochs": 10}),
    seed=args.seed,
)

run_simulation(args, cfg)
```

Once installed, dEngine also exposes a console entry point:

```bash
dengine-simulate --config <your_config.yml> --gpu 0 --output_directory logs/
```

Run `dengine-simulate --help` (or `python main.py --help` from an example) to see all CLI options. Use `--sanity_check` to load the engine and datasets and skip training.

### Check a configuration before a run

Use `dengine-validate` to check a configuration without loading datasets, importing custom experiment components, or creating run files.

```sh
dengine-validate --config examples/HelloWorld/config.yml --set seed=92
```

From a source checkout, you can also use `python -m dengine.bin.validate_config`. List YAML files in load order and repeat `--set KEY=VALUE` to apply dotted overrides. Validation uses the simulator's YAML loader and checks the required fields and types in the combined configuration. Exit status `0` means the schema passed, `1` means the configuration failed, and `2` signals invalid command usage. Component availability and component-specific arguments are checked during engine setup.

### Record and verify a run setup

Pass `--run_manifest` to `dengine-simulate` or set `SimulationArguments.run_manifest=True` in Python. Choose a fresh output directory for the experiment name. After the engine loads successfully, dEngine writes `run-manifest.json` beside the resolved `config.yaml`. The manifest describes the prepared setup, while training and its outputs follow the normal engine path.

The manifest records effective configuration and seed, dEngine code and runtime versions, and source fingerprints for the components the engine loads. It also records training and test tensor fingerprints, graph node order and topology, and checksums for saved graph and partition files. The existing partition CSVs describe class counts. The new `partitioning/indices.json` records each client's ordered train and validation indices. The manifest uses paths relative to the run directory.

From the repository root, run this two-client example with synthetic images and one communication round. It uses no dataset downloads.

```sh
python -m examples.RunManifest.run --config examples/RunManifest/config.yml --output_directory logs/manifest-demo/original --gpu -1 --run_manifest
python -m dengine.run_manifest logs/manifest-demo/original/tiny-manifest/run-manifest.json
python -m examples.RunManifest.run --config logs/manifest-demo/original/tiny-manifest/config.yaml --output_directory logs/manifest-demo/replay --gpu -1 --run_manifest
python -m dengine.run_manifest logs/manifest-demo/original/tiny-manifest/run-manifest.json --compare logs/manifest-demo/replay/tiny-manifest/run-manifest.json
```

The last command verifies both directories and checks that the recorded setups match. For your own replay, use the recorded resolved config and the same dataset inputs, runtime settings, and versions of any custom components and their dependencies. The comparison allows a different output root and excludes graph plots. Dataset hashes cover loaded tensors before per-example transforms. A complete comparison requires fingerprints for both datasets and the directly loaded components. Use a new output root each time you repeat the example.

More examples:

- [`examples/MNIST`](./examples/MNIST): minimal example of decentralized training on MNIST.
- [`examples/HelloWorld`](./examples/HelloWorld): implements a complex use case involving advanced customization.
- [`examples/Scaffold`](./examples/Scaffold): the SCAFFOLD federated optimization algorithm.
- [`examples/GAN`](./examples/GAN): decentralized training of a generative model.
- [`examples/events`](./examples/events/): advanced events API and distributed runtime

## Project structure

```
dEngine/
├── dengine/                  # The core library
│   ├── bin/                  # Simulation entry point & CLI argument parsing
│   ├── config/                # Experiment config loading, schemas, and BUILTINS
│   ├── graph/                 # Network topology abstractions (static & dynamic)
│   ├── dataset/                # Built-in datasets (MNIST, EMNIST, CIFAR-10/100)
│   ├── partitioning/           # IID / non-IID data partitioning strategies
│   ├── models/                 # Reference model architectures
│   ├── scenarios/               # Centralized, federated, decentralized & event-driven engines
│   ├── training_strategies/     # Local update strategies
│   ├── callbacks/                # Checkpointing, logging, and hooks
│   ├── analysis/                  # Post-hoc metrics, plots, and experiment reports
│   └── interfaces.py              # Core protocols/interfaces used across the framework
├── configs/                   # Built-in YAML configs (graphs, datasets, partitioning, scenarios)
├── examples/                   # Runnable example projects
├── tests/                       # Test suite
└── bin/                          # Auxiliary scripts (dataset normalization, schedule generation, etc.)
```

## Built-in configs

dEngine ships with ready-to-use configs under [`configs/core`](./configs/core), addressable through `dengine.BUILTINS`:

| Category | Options |
|---|---|
| **Graph** | Barabási–Albert (small/medium), Erdős–Rényi (small/medium), complete graph, star, centralized |
| **Dataset** | MNIST (full/reduced), CIFAR-10 (standard/fast), CIFAR-100 |
| **Partitioning** | IID, non-IID, IID balanced excluding hub, IID/non-IID hub mix |
| **Scenario** | Centralized, Federated, Decentralized (homogeneous), Decentralized (heterogeneous) |

## Related projects

- [`dengine-uciml`](https://github.com/DecentralizedLearning/dengine-uciml) — extends dEngine with support for the UCI Machine Learning datasets.
- [`notebooks`](https://github.com/DecentralizedLearning/notebooks) — notebooks for visualizing and analyzing dEngine experiment results.

## License

dEngine is released under the [MIT License](./LICENSE).

## Citing
As of today you can either cite one of these two papers:
* Sabella, Samuele, et al. "[Operating Regimes of Decentralized Learning Under Mobility and Bandwidth Constraints.](https://arxiv.org/abs/2606.28342)" arXiv preprint arXiv:2606.28342 (2026).
* Sabella, Samuele, et al. "[The Built-In Robustness of Decentralized Federated Averaging to Bad Data.](https://arxiv.org/pdf/2502.18097?)" 2025 International Joint Conference on Neural Networks (IJCNN). IEEE, 2025.

> **Note**:
> Full source code will be released soon as we finalize upcoming extensions

## Pull Request & Commit Guidelines
* **Linear History is mandatory**. Rebase!
* **One commit per file:** Keep commits atomic by limiting each commit to a single modified file.
* **Write clear commit titles:** Commit titles are important for tracking changes and generating release notes. Use a descriptive, imperative summary.

## AI-Assisted Contributions
Time is sacred and human learning/understanding is the final goal.

Given that nowadays LLMs have lowered the requirements of writing software, a few guidelines are needed to steer the project correctly. Specifically, while writing thousands of lines has become extremely cheap, reviewing them is extremely costly. Due to a lack of time/energy, I feel the need to filter contributions before starting a code review. Contributions should come from people who have real interest in the project.

Throughout these few lines, I will refer to **"blind contributions"** as PRs for the sake of contributing, which lack an underlying motivation or interest in the project.

* Blind contributions are not welcome.
* Low-hanging fruits, issues which are marked as entry level, are there to let newcomers understand the project and learn better how to interact with or extend the codebase.
* Before contributing to the project, ask yourself:
  - Do I have or am I building sufficient background to understand and appreciate this work?
  - Do I like working in decentralized learning?
  - Have I used, or do I plan to use, the repository for my own work?

  If the answers to these questions are positive, then please contact me personally via email (samuele.sabella@iit.cnr.it). The project is still basically a solo-developer project and sharing the experience in the field is extremely valuable.
* In case you do not agree with the policy, you are free to fork this repo and manage it as you like since it is distributed with an MIT license.


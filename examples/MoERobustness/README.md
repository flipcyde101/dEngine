# dEngine MNIST mixture-of-experts pilot

The pilot compares two classifiers under a frozen training-data quality change on a single simulated client. It uses the public MNIST train and test files listed in [protocol.json](protocol.json). The change is an illustrative pixel-blend proxy for the question studied by Sabella et al. The paper uses latent interpolation and a network of clients. [Paper](https://arxiv.org/abs/2502.18097).

- `data.py` selects the split and constructs the deterministic proxy. The training records keep their original labels, and validation and test files stay clean.
- `models.py` contains the dense and four-expert top-1 classifiers. Shared layers receive identical initializations for a given seed.
- `metrics.py` and `run.py` set up the existing centralized scenario and save the fixed-epoch evaluation, per-image logits and route assignments, split IDs, weights and hashes.
- `summarize.py` checks complete arms, validates artifacts, independently recalculates classification metrics and reports paired results.
- `lab/` contains CPU dependency pins, a synthetic qualification run and a bounded seed worker.

## Local use

Use Python 3.13 and the platform's hash-locked environment. The source checkout supplies dEngine imports. All output directories must be new. Use a directory outside the source checkout for pilot outputs.

```text
python -m pip install --require-hashes --only-binary=:all: -r examples/MoERobustness/lab/linux.lock
python -m pytest -q -p no:cacheprovider tests/test_moe_robustness.py
python -m examples.MoERobustness.run --mode smoke --arch moe --data proxy --seed 11 --dataset-dir ./unused --output-dir ./logs/dengine-smoke
```

Use `windows.lock` for Windows. Smoke mode uses a generated fixture. It qualifies code behavior and never contributes research results. Pilot mode requires the four verified IDX files and a clean checkout. Run each of the four dense/MoE × clean/proxy arms in a separate process for each of seeds 11, 29 and 47. Use a new output directory per cell. Before running the example below, set `DENGINE_INPUT_ROOTS` to the absolute path of the existing MNIST directory and `DENGINE_OUTPUT_ROOTS` to the absolute path of the existing results directory.

```text
python -m examples.MoERobustness.run --mode pilot --arch moe --data proxy --seed 11 --dataset-dir ../mnist --output-dir ../results/seed-11-moe-proxy
```

For summarization, set `DENGINE_INPUT_ROOTS` to the results directory before running the command below. Keep `DENGINE_OUTPUT_ROOTS` set to the results directory.

```text
python -m examples.MoERobustness.summarize ../results ../results/paired-results.json
```

The summary reports missing or invalid cells. The primary effect is the dense model's class-4 F1 loss minus the MoE's class-4 F1 loss, both measured against each architecture's own clean-trained control. Display each paired effect. Interpret the median and range as exploratory observations conditional on this dataset and these implementations. Report feasibility failures before interpreting the comparison.

For a new isolated worker with its checkout at `/study`, a read-only MNIST mount at `/mnist`, and generated output under `/tmp/experiment`, set `DENGINE_INPUT_ROOTS=/mnist` and `DENGINE_OUTPUT_ROOTS=/tmp` in the operator's container configuration. Retain the original worker configuration and source revision when verifying archived study results.

Read the raw predictions and weights using NumPy with `allow_pickle=False`. The final epoch, five fixed epochs for the pilot, is the only test checkpoint. The same minimum validation and route-load rules apply to all three seeds. Record code revisions, dataset hashes, the exact environment and every attempt when preparing a research report. [Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

# Matched MNIST routing follow-up

This study compares the original dense classifier, the original expert classifier and a wider dense control. Each start produces six fits from scratch. Workers use an authenticated subset of MNIST training examples and development validation examples.

- `protocol.json` fixes the contrast, cohort sizes and interpretation rules.
- `cohort.py` selects the frozen development cohorts and the separately retained evaluation set.
- `materialize.py` prepares only the reviewed development subset from pinned public source files.
- `fit.py` runs a single fit through dEngine and records the actual sampler order and weights.
- `worker.py` runs a bounded group of fit processes and returns their evidence.
- `verify.py` checks the matched design and artifacts. `score.py` first verifies the entire training matrix and then scores the local evaluation set.

Use the Python 3.13 runtime and the original study's platform-specific hash lock. The local study folder, cloud receipts and new evaluation output belong outside the source checkout. Preserve the committed source, exact cohort manifests, provider and native receipts, saved predictions and any failed attempts.

The primary inference is conditional on this MNIST corpus and pixel-blend setting. Secondary controls and sensitivity checks support interpretation of the primary outcome.

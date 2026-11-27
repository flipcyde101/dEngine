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

## Paired class comparisons

After a complete verified scoring run, `study-results.json` includes `secondary_class_comparisons` and the output directory contains `class-comparisons.md`. This secondary exploratory view reports the dense-minus-expert difference in clean-to-proxy F1 loss for every digit and the corresponding wide-minus-expert capacity control. Positive values favor the expert model. The four matched start effects are averaged within each of the twenty cohorts. Each descriptive 95% Student t interval uses those twenty cohort means and is unadjusted for the twenty reported class contrasts. Digit 4 is the prespecified primary outcome.

For an already completed study, create the same report from its saved scores in a fresh directory. Run from the source checkout with the existing Python 3.13 study environment.

```powershell
python -m examples.MoERobustness.followup.class_report D:/research/evaluation/study-results.json D:/research/class-report
```

Use your own paths for the input and the fresh output directory. The command reads the saved scores, checks the complete matrix of 20 cohorts with 4 starts each and all six arms, and records the input JSON SHA-256 in `class-comparisons.json`. It also writes `class-comparisons.md` and preserves the input file. Existing output directories are rejected. The report uses the supplied scores as a secondary analysis. The original scoring command supplies the training receipt and sealed evaluation checks.

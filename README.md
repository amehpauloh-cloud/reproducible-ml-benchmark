# Reproducible ML Benchmark

A small binary-classification experiment demonstrating leakage-aware preprocessing, baseline comparison, reproducibility metadata and a fixed noise stress test.

**Scope:** an educational portfolio project on generated data. It is not a production fraud, credit or clinical model, and its scores are not evidence of performance on real users.

## Quick start

Use Python 3.11+ in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmark.py
```

On Windows, activate with `.venv\Scripts\activate`. The experiment creates `reports/benchmark.json`. Reproduce the committed run with seed 42 and 2,400 samples.

## Method

1. Generate 16 features with informative and redundant variables, class imbalance and label noise.
2. Create a stratified 75/25 train/test split with a fixed seed.
3. Fit scaling and logistic regression in a single pipeline using **training data only**.
4. Compare with a prior-probability baseline using accuracy, balanced accuracy, ROC AUC and Brier score.
5. Evaluate a predeclared noise stress scenario using feature scales estimated on training data.
6. Record the dataset hash, dependency versions, seed and split overlap for reproducibility.

Accuracy can hide majority-class guessing, so it is reported alongside class-balanced and probability-quality metrics. Lower Brier score is better. No hyperparameter search or test-set tuning is performed.

## Limitations

- The holdout and noise stress results use one split. They are exploratory, with no confidence interval or significance claim.
- Feature noise is a deliberately simple stress test, not a measured deployment shift.
- Synthetic relationships do not capture domain-specific data quality, fairness or operational issues.
- Calibration is measured with Brier score, not improved with a separate calibration set.
- Bitwise reproducibility across different numerical libraries/hardware is not guaranteed; versions and the data hash are recorded.

## Tests

Tests check split disjointness, training-only scaler fitting, repeated-run equality, metric bounds and rejection of undersized input. CI runs them on Python 3.11 and 3.12.

MIT licensed. See `LICENSE`.

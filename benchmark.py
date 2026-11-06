"""Seeded binary classification benchmark on synthetic data."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def metrics(y, probability):
    return {'accuracy': float(accuracy_score(y, probability >= .5)),
            'balanced_accuracy': float(balanced_accuracy_score(y, probability >= .5)),
            'roc_auc': float(roc_auc_score(y, probability)),
            'brier_score': float(brier_score_loss(y, probability))}


def run(seed=42, samples=2400):
    if samples < 200:
        raise ValueError('Use at least 200 samples for this demo')
    X, y = make_classification(n_samples=samples, n_features=16, n_informative=8,
                               n_redundant=4, weights=[.8, .2], flip_y=.03,
                               class_sep=1.0, random_state=seed)
    indices = np.arange(samples)
    train_ids, test_ids = train_test_split(indices, test_size=.25, stratify=y, random_state=seed)
    X_train, X_test = X[train_ids], X[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(X_train, y_train)  # scaler is fitted on training data only
    baseline = DummyClassifier(strategy='prior').fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    rng = np.random.default_rng(seed + 1)
    # Fixed, untuned stress scenario, not a real-world distribution-shift estimate.
    noise = rng.normal(0, 1.0, X_test.shape) * X_train.std(axis=0)
    perturbed = model.predict_proba(X_test + noise)[:, 1]
    report = {
        'dataset': 'sklearn.make_classification (synthetic, no personal data)',
        'seed': seed, 'samples': samples,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'data_sha256': hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        'split': {'train': len(train_ids), 'test': len(test_ids),
                  'overlap': int(len(set(train_ids) & set(test_ids))),
                  'test_positive_rate': float(y_test.mean())},
        'prior_baseline': metrics(y_test, baseline.predict_proba(X_test)[:, 1]),
        'logistic_regression': metrics(y_test, probability),
        'noise_stress': metrics(y_test, perturbed),
    }
    return report, model, X_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=2400)
    parser.add_argument('--output', type=Path, default=Path('reports/benchmark.json'))
    args = parser.parse_args()
    report, _, _ = run(args.seed, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

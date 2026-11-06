import unittest
import numpy as np
from benchmark import run


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.model, cls.train = run(samples=400)

    def test_split_disjoint(self):
        self.assertEqual(self.report['split']['overlap'], 0)
        self.assertEqual(self.report['split']['train'], 300)
        self.assertEqual(self.report['split']['test'], 100)

    def test_scaler_only_sees_training_data(self):
        scaler = self.model.named_steps['standardscaler']
        np.testing.assert_allclose(scaler.mean_, self.train.mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_, 300)

    def test_reproducible(self):
        again, _, _ = run(samples=400)
        self.assertEqual(self.report, again)

    def test_metrics_bounded(self):
        for scenario in ('prior_baseline', 'logistic_regression', 'noise_stress'):
            for value in self.report[scenario].values():
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)

    def test_reject_tiny_dataset(self):
        with self.assertRaises(ValueError):
            run(samples=10)


if __name__ == '__main__':
    unittest.main()

"""
Unit tests for classification and regression evaluation metrics.
Verifies Phase 5 metrics and statistical primitives in experiments/phase5/code/metrics.py.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add experiments/phase5/code to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE5_CODE_DIR = os.path.join(ROOT_DIR, "experiments", "phase5", "code")
if PHASE5_CODE_DIR not in sys.path:
    sys.path.insert(0, PHASE5_CODE_DIR)

from metrics import evaluate_classification, evaluate_regression, compute_ci


class TestMetrics(unittest.TestCase):
    def test_classification_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_prob = np.array([0.05, 0.1, 0.9, 0.95])
        res = evaluate_classification(y_true, y_pred_prob)

        self.assertAlmostEqual(res["roc_auc"], 1.0)
        self.assertAlmostEqual(res["bacc"], 1.0)
        self.assertAlmostEqual(res["acc"], 1.0)
        self.assertAlmostEqual(res["sens"], 1.0)
        self.assertAlmostEqual(res["spec"], 1.0)
        self.assertAlmostEqual(res["mcc"], 1.0)
        self.assertAlmostEqual(res["f1_macro"], 1.0)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["tn"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)

    def test_classification_inverted_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_prob = np.array([0.9, 0.8, 0.1, 0.2])
        res = evaluate_classification(y_true, y_pred_prob)

        self.assertAlmostEqual(res["roc_auc"], 0.0)
        self.assertAlmostEqual(res["bacc"], 0.0)
        self.assertAlmostEqual(res["acc"], 0.0)
        self.assertAlmostEqual(res["mcc"], -1.0)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["tn"], 0)
        self.assertEqual(res["fp"], 2)
        self.assertEqual(res["fn"], 2)

    def test_classification_mono_class(self):
        # When only one class is present, ROC-AUC should be NaN without crashing
        y_true = np.array([1, 1, 1, 1])
        y_pred_prob = np.array([0.6, 0.7, 0.8, 0.9])
        res = evaluate_classification(y_true, y_pred_prob)

        self.assertTrue(np.isnan(res["roc_auc"]))
        self.assertTrue(np.isnan(res["prauc"]))
        self.assertIn("brier", res)

    def test_regression_perfect_predictions(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        res = evaluate_regression(y_true, y_pred)

        self.assertAlmostEqual(res["mae"], 0.0)
        self.assertAlmostEqual(res["rmse"], 0.0)
        self.assertAlmostEqual(res["r2"], 1.0)
        self.assertAlmostEqual(res["pearson_r"], 1.0)
        self.assertAlmostEqual(res["spearman_rho"], 1.0)

    def test_regression_constant_prediction(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([25.0, 25.0, 25.0, 25.0])  # zero variance
        res = evaluate_regression(y_true, y_pred)

        # (|10-25| + |20-25| + |30-25| + |40-25|) / 4 = 40 / 4 = 10.0
        self.assertAlmostEqual(res["mae"], 10.0)
        # Pearson and Spearman should gracefully return 0.0 when std is 0
        self.assertAlmostEqual(res["pearson_r"], 0.0)
        self.assertAlmostEqual(res["spearman_rho"], 0.0)

    def test_regression_subject_records(self):
        df_records = pd.DataFrame({
            "subject#": [1, 1, 2, 2]
        })
        y_true = np.array([10.0, 12.0, 20.0, 24.0])
        y_pred = np.array([11.0, 11.0, 22.0, 22.0])  # subject 1 MAE = 1.0, subject 2 MAE = 2.0
        res = evaluate_regression(y_true, y_pred, df_records=df_records)

        self.assertIn("mean_subj_mae", res)
        self.assertIn("std_subj_mae", res)
        self.assertAlmostEqual(res["mean_subj_mae"], 1.5)
        self.assertAlmostEqual(res["std_subj_mae"], 0.5)

    def test_compute_ci(self):
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        mean, std, (ci_low, ci_high) = compute_ci(values, confidence=0.95)

        self.assertAlmostEqual(mean, 14.0)
        self.assertAlmostEqual(std, np.std(values, ddof=1))
        self.assertTrue(ci_low < mean < ci_high)

    def test_compute_ci_single_value(self):
        values = [15.0]
        mean, std, (ci_low, ci_high) = compute_ci(values)
        self.assertEqual(mean, 15.0)
        self.assertEqual(std, 0.0)
        self.assertEqual(ci_low, 15.0)
        self.assertEqual(ci_high, 15.0)


if __name__ == "__main__":
    unittest.main()

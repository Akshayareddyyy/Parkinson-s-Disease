"""
Unit tests for personalized patient calibration, normalization schemes, and baseline models
in experiments/phase4b/code/run_exp5_personalized_prediction.py.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4B_CODE_DIR = os.path.join(ROOT_DIR, "experiments", "phase4b", "code")
if PHASE4B_CODE_DIR not in sys.path:
    sys.path.insert(0, PHASE4B_CODE_DIR)

from phase4b_common import PREDICTORS
from run_exp5_personalized_prediction import extract_patient_calibration


class TestPersonalizedRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_records = 30
        self.k = 10

        # Construct single patient synthetic records
        data = {
            "subject#": np.repeat(1, self.n_records),
            "test_time": np.sort(np.random.uniform(0.0, 180.0, self.n_records)),
            "motor_UPDRS": 20.0 + 0.05 * np.arange(self.n_records) + np.random.normal(0, 0.5, self.n_records)
        }
        for feat in PREDICTORS:
            data[feat] = np.random.normal(10.0, 2.0, self.n_records)

        self.df_patient = pd.DataFrame(data)

    def test_calibration_split_exactness(self):
        """Verifies exactly k calibration records and N - k future records."""
        res = extract_patient_calibration(self.df_patient, k=self.k, norm_type="z_score")

        self.assertEqual(res["k"], self.k)
        self.assertEqual(res["N_total"], self.n_records)
        self.assertEqual(res["N_calib"], self.k)
        self.assertEqual(res["N_future"], self.n_records - self.k)
        self.assertEqual(len(res["delta_future"]), self.n_records - self.k)
        self.assertEqual(res["X_future"].shape[0], self.n_records - self.k)
        self.assertEqual(res["X_future"].shape[1], len(PREDICTORS))

    def test_baseline_and_delta_calculation(self):
        """Verifies y_base matches calibration mean and delta_future = y_future - y_base."""
        res = extract_patient_calibration(self.df_patient, k=self.k, norm_type="z_score")

        sorted_df = self.df_patient.sort_values("test_time").reset_index(drop=True)
        expected_y_base = float(sorted_df.iloc[:self.k]["motor_UPDRS"].mean())
        self.assertAlmostEqual(res["y_base"], expected_y_base, places=6)

        expected_delta = res["y_future_abs"] - res["y_base"]
        np.testing.assert_allclose(res["delta_future"], expected_delta, atol=1e-12)

    def test_normalization_types(self):
        """Verifies z_score, robust_iqr, and min_max execute cleanly without NaNs."""
        for norm in ["z_score", "robust_iqr", "min_max"]:
            res = extract_patient_calibration(self.df_patient, k=self.k, norm_type=norm)
            self.assertFalse(np.any(np.isnan(res["X_future"])), f"{norm} should not contain NaNs")
            self.assertFalse(np.any(np.isinf(res["X_future"])), f"{norm} should not contain Infs")

    def test_invalid_norm_type_raises_error(self):
        """Verifies ValueError is raised for unrecognized normalization type."""
        with self.assertRaises(ValueError):
            extract_patient_calibration(self.df_patient, k=self.k, norm_type="invalid_norm")

    def test_records_less_than_or_equal_to_k_raises_error(self):
        """Verifies ValueError is raised when patient has <= k records."""
        df_short = self.df_patient.iloc[:self.k]
        with self.assertRaises(ValueError):
            extract_patient_calibration(df_short, k=self.k)


if __name__ == "__main__":
    unittest.main()

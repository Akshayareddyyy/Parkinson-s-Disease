"""
Unit tests for the reusable dataset schema validation tool in code/validate_dataset_schema.py.
Uses small synthetic fixtures to test valid data, missing columns, invalid types,
missing values, and invalid numeric values.
"""

import sys
import os
import unittest
import tempfile
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from validate_dataset_schema import validate_dataset, ValidationError, EXPECTED_COLUMNS


class TestValidateDatasetSchema(unittest.TestCase):

    def setUp(self):
        # Create a minimal valid synthetic dataset fixture
        n = 10
        data = {
            "subject#": np.repeat([1, 2], n // 2),
            "age": np.repeat([60, 65], n // 2),
            "sex": np.repeat([0, 1], n // 2),
            "test_time": np.linspace(0.0, 50.0, n),
            "motor_UPDRS": np.linspace(15.0, 25.0, n),
            "total_UPDRS": np.linspace(20.0, 35.0, n)  # total >= motor
        }
        for col in EXPECTED_COLUMNS:
            if col not in data:
                data[col] = np.random.uniform(0.001, 1.0, n)

        self.valid_df = pd.DataFrame(data)

    def _write_temp_csv(self, df):
        tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.to_csv(tf.name, index=False)
        tf.close()
        return tf.name

    def test_valid_synthetic_dataset_passes(self):
        """Verifies clean synthetic dataset passes all validation checks."""
        csv_path = self._write_temp_csv(self.valid_df)
        try:
            report = validate_dataset(csv_path, verbose=False)
            self.assertTrue(report["all_checks_passed"])
            self.assertEqual(report["n_rows"], len(self.valid_df))
            self.assertEqual(report["n_cols"], 22)
            self.assertEqual(report["unique_subjects"], 2)
        finally:
            os.remove(csv_path)

    def test_missing_file_raises_error(self):
        """Verifies FileNotFoundError is raised when file does not exist."""
        with self.assertRaises(FileNotFoundError):
            validate_dataset("non_existent_file_xyz.csv", verbose=False)

    def test_missing_columns_raises_validation_error(self):
        """Verifies ValidationError is raised when required columns are missing."""
        df_missing = self.valid_df.drop(columns=["motor_UPDRS", "Jitter(%)"])
        csv_path = self._write_temp_csv(df_missing)
        try:
            with self.assertRaises(ValidationError) as ctx:
                validate_dataset(csv_path, verbose=False)
            self.assertIn("Missing required columns", str(ctx.exception))
        finally:
            os.remove(csv_path)

    def test_missing_values_raises_validation_error(self):
        """Verifies ValidationError is raised when NaN values are present."""
        df_nulls = self.valid_df.copy()
        df_nulls.loc[0, "age"] = np.nan
        csv_path = self._write_temp_csv(df_nulls)
        try:
            with self.assertRaises(ValidationError) as ctx:
                validate_dataset(csv_path, verbose=False)
            self.assertIn("null values", str(ctx.exception))
        finally:
            os.remove(csv_path)

    def test_infinite_values_raises_validation_error(self):
        """Verifies ValidationError is raised when infinite values are present."""
        df_inf = self.valid_df.copy()
        df_inf.loc[0, "RPDE"] = np.inf
        csv_path = self._write_temp_csv(df_inf)
        try:
            with self.assertRaises(ValidationError) as ctx:
                validate_dataset(csv_path, verbose=False)
            self.assertIn("infinite values", str(ctx.exception))
        finally:
            os.remove(csv_path)

    def test_invalid_subject_range_raises_validation_error(self):
        """Verifies ValidationError is raised when subject# is out of [1, 42] range."""
        df_sub = self.valid_df.copy()
        df_sub.loc[0, "subject#"] = 999
        csv_path = self._write_temp_csv(df_sub)
        try:
            with self.assertRaises(ValidationError) as ctx:
                validate_dataset(csv_path, verbose=False)
            self.assertIn("subject# values must be integers between 1 and 42", str(ctx.exception))
        finally:
            os.remove(csv_path)

    def test_invalid_sex_values_raises_validation_error(self):
        """Verifies ValidationError is raised when sex is not in {0, 1}."""
        df_sex = self.valid_df.copy()
        df_sex.loc[0, "sex"] = 2
        csv_path = self._write_temp_csv(df_sex)
        try:
            with self.assertRaises(ValidationError) as ctx:
                validate_dataset(csv_path, verbose=False)
            self.assertIn("sex must contain only binary values", str(ctx.exception))
        finally:
            os.remove(csv_path)

    def test_inverted_targets_raises_validation_error(self):
        """Verifies ValidationError is raised when motor_UPDRS > total_UPDRS."""
        df_inv = self.valid_df.copy()
        df_inv.loc[0, "motor_UPDRS"] = 40.0
        df_inv.loc[0, "total_UPDRS"] = 20.0
        csv_path = self._write_temp_csv(df_inv)
        try:
            with self.assertRaises(ValidationError) as ctx:
                validate_dataset(csv_path, verbose=False)
            self.assertIn("motor_UPDRS > total_UPDRS", str(ctx.exception))
        finally:
            os.remove(csv_path)

    def test_negative_acoustic_values_raises_validation_error(self):
        """Verifies ValidationError is raised when perturbation features are negative."""
        df_neg = self.valid_df.copy()
        df_neg.loc[0, "Jitter(%)"] = -0.05
        csv_path = self._write_temp_csv(df_neg)
        try:
            with self.assertRaises(ValidationError) as ctx:
                validate_dataset(csv_path, verbose=False)
            self.assertIn("negative values", str(ctx.exception))
        finally:
            os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()

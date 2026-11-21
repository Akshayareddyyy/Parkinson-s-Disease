"""
Unit tests for biomedical dataset schema validation, column specifications,
data type integrity, physiological bounds, and invalid input rejection.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT_DIR, "data", "parkinsons_updrs.csv")

EXPECTED_COLUMNS = [
    "subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS",
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"
]


def validate_schema(df):
    """Schema validation logic verifying columns, nulls, types, and physiological ranges."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        raise ValueError(f"Dataset contains null values: {null_counts[null_counts > 0].to_dict()}")

    # Physiological range validations
    if not np.all(df["sex"].isin([0, 1])):
        raise ValueError("Column 'sex' must contain only 0 and 1.")

    if np.any(df["motor_UPDRS"] <= 0):
        raise ValueError("motor_UPDRS must be strictly positive.")

    if np.any(df["motor_UPDRS"] > df["total_UPDRS"]):
        raise ValueError("motor_UPDRS cannot exceed total_UPDRS.")

    if np.any(df["Jitter(%)"] < 0) or np.any(df["Shimmer"] < 0):
        raise ValueError("Acoustic perturbation features must be non-negative.")

    return True


class TestDatasetSchema(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_FILE)

    def test_complete_column_specification(self):
        """Verifies dataset has all 22 required columns."""
        self.assertEqual(len(self.df.columns), 22)
        for col in EXPECTED_COLUMNS:
            self.assertIn(col, self.df.columns)

    def test_zero_null_or_missing_values(self):
        """Verifies dataset has zero null, NaN, or infinite values."""
        self.assertEqual(self.df.isnull().sum().sum(), 0, "No null values allowed.")
        self.assertEqual(np.isinf(self.df.values).sum(), 0, "No infinite values allowed.")

    def test_clinical_bounds_and_relationships(self):
        """Verifies clinical relationships: motor_UPDRS <= total_UPDRS and valid bounds."""
        # Clinical constraint: Total UPDRS includes Motor UPDRS
        invalid_updrs = self.df[self.df["motor_UPDRS"] > self.df["total_UPDRS"]]
        self.assertEqual(len(invalid_updrs), 0, "motor_UPDRS must always be <= total_UPDRS.")

        # Range bounds
        self.assertTrue((self.df["motor_UPDRS"] >= 5.0).all() and (self.df["motor_UPDRS"] <= 40.0).all())
        self.assertTrue((self.df["total_UPDRS"] >= 7.0).all() and (self.df["total_UPDRS"] <= 55.0).all())
        self.assertTrue((self.df["age"] >= 35).all() and (self.df["age"] <= 85).all())

    def test_schema_validator_catches_missing_columns(self):
        """Verifies validator raises ValueError when required columns are absent."""
        df_incomplete = self.df.drop(columns=["motor_UPDRS"])
        with self.assertRaises(ValueError):
            validate_schema(df_incomplete)

    def test_schema_validator_catches_nulls(self):
        """Verifies validator raises ValueError when NaN values are present."""
        df_corrupt = self.df.copy()
        df_corrupt.iloc[0, 1] = np.nan
        with self.assertRaises(ValueError):
            validate_schema(df_corrupt)

    def test_schema_validator_catches_inverted_targets(self):
        """Verifies validator raises ValueError if motor_UPDRS > total_UPDRS."""
        df_invalid = self.df.copy()
        df_invalid.loc[0, "motor_UPDRS"] = 50.0
        df_invalid.loc[0, "total_UPDRS"] = 20.0
        with self.assertRaises(ValueError):
            validate_schema(df_invalid)


if __name__ == "__main__":
    unittest.main()

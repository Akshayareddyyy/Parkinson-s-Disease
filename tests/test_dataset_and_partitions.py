"""
Unit tests for UCI Parkinson's Telemonitoring Dataset loading and subject-wise partitioning.
Verifies:
  - Dataset integrity: 5,875 records, 42 unique subjects.
  - Zero subject overlap across dev (26), val (7), and test (9) partitions.
  - Development-only median thresholding to prevent target leakage.
"""

import os
import sys
import unittest
import numpy as np

# Add project root and code directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))

from common import load, partition, make_sets, FEATS, DATA


class TestDatasetAndPartitions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load()

    def test_dataset_record_and_subject_counts(self):
        """Verifies exact record count and independent subject count."""
        self.assertEqual(len(self.df), 5875, "Dataset must contain exactly 5,875 recordings.")
        self.assertEqual(self.df['subject#'].nunique(), 42, "Dataset must contain exactly 42 unique subjects.")

    def test_required_columns_exist(self):
        """Verifies demographic, target, and acoustic features are present."""
        expected_cols = [
            'subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
            'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
            'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")

    def test_feature_list_excludes_leakage(self):
        """Verifies FEATS excludes subject#, motor_UPDRS, and total_UPDRS."""
        self.assertNotIn('subject#', FEATS, "subject# must be excluded from features.")
        self.assertNotIn('motor_UPDRS', FEATS, "motor_UPDRS must be excluded from features.")
        self.assertNotIn('total_UPDRS', FEATS, "total_UPDRS must be excluded from features to prevent proxy leakage.")
        self.assertEqual(len(FEATS), 19, "Baseline acoustic feature set must contain exactly 19 features.")

    def test_subject_wise_partitions_disjointness_across_10_seeds(self):
        """Verifies strict subject disjointness for all 10 audited seeds."""
        for seed in range(10):
            dev_subs, val_subs, test_subs = partition(self.df, seed, n_dev=26, n_val=7, n_test=9)
            
            # Check subject counts
            self.assertEqual(len(dev_subs), 26, f"Seed {seed}: Dev set must contain 26 subjects.")
            self.assertEqual(len(val_subs), 7, f"Seed {seed}: Val set must contain 7 subjects.")
            self.assertEqual(len(test_subs), 9, f"Seed {seed}: Test set must contain 9 subjects.")
            
            # Check disjointness
            set_dev = set(dev_subs)
            set_val = set(val_subs)
            set_test = set(test_subs)
            
            self.assertTrue(set_dev.isdisjoint(set_val), f"Seed {seed}: Dev and Val subjects overlap!")
            self.assertTrue(set_dev.isdisjoint(set_test), f"Seed {seed}: Dev and Test subjects overlap!")
            self.assertTrue(set_val.isdisjoint(set_test), f"Seed {seed}: Val and Test subjects overlap!")
            self.assertEqual(len(set_dev | set_val | set_test), 42, f"Seed {seed}: Total partitioned subjects must equal 42.")

    def test_make_sets_record_accounting_and_thresholding(self):
        """Verifies make_sets record sums, threshold calculation, and binary labels."""
        for seed in range(3):  # Check first 3 seeds
            D, V, T, thr = make_sets(self.df, seed)
            
            # Check record accounting
            total_records = len(D) + len(V) + len(T)
            self.assertEqual(total_records, 5875, f"Seed {seed}: Sum of records must equal 5875.")
            
            # Check threshold: dev median
            expected_thr = float(D['motor_UPDRS'].median())
            self.assertAlmostEqual(thr, expected_thr, places=6, msg="Threshold must be dev-set median motor_UPDRS.")
            
            # Check binary label construction
            for partition_name, p_df in [('Dev', D), ('Val', V), ('Test', T)]:
                self.assertIn('y', p_df.columns, f"{partition_name} must have binary target column 'y'.")
                expected_y = (p_df['motor_UPDRS'] >= thr).astype(int)
                np.testing.assert_array_equal(p_df['y'].values, expected_y.values)


if __name__ == '__main__':
    unittest.main()

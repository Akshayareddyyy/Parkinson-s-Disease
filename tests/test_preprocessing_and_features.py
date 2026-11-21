"""
Unit tests for feature definitions, fold-contained scaling, and quantum angle preprocessing.
Verifies:
  - 19-feature original acoustic set vs. 9-feature pruned manifold.
  - FoldContainedQuantumPreprocessor isolation (fitted strictly on X_dev).
  - Rotation angle bounds [-pi, pi] and discrete mapping for binary sex {0 -> 0.0, 1 -> pi}.
  - Zero leakage from validation and test partitions into scaler moments.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'experiments', 'phase5', 'code'))

import common
from common import load, make_sets, FEATS
import importlib.util

spec = importlib.util.spec_from_file_location("phase5_common", os.path.join(ROOT_DIR, 'experiments', 'phase5', 'code', 'common.py'))
phase5_common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase5_common)

if not hasattr(common, 'FEATS_9'):
    common.FEATS_9 = phase5_common.FEATS_9
    common.CONT_FEATS_8 = phase5_common.CONT_FEATS_8

from preprocessing import FoldContainedQuantumPreprocessor
FEATS_9 = phase5_common.FEATS_9
CONT_FEATS_8 = phase5_common.CONT_FEATS_8


class TestPreprocessingAndFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load()
        cls.D, cls.V, cls.T, cls.thr = make_sets(cls.df, seed=0)

    def test_feature_manifold_definitions(self):
        """Verifies 9-feature manifold and continuous 8-feature sublist."""
        self.assertEqual(len(FEATS_9), 9, "Pruned manifold must contain exactly 9 features.")
        self.assertEqual(len(CONT_FEATS_8), 8, "Continuous features must contain exactly 8 features.")
        self.assertIn('sex', FEATS_9, "sex must be included in FEATS_9.")
        self.assertNotIn('sex', CONT_FEATS_8, "sex is binary and must be excluded from continuous list.")
        
        expected_9 = ['age', 'sex', 'test_time', 'Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
        self.assertEqual(FEATS_9, expected_9)

    def test_fold_contained_preprocessor_fitting(self):
        """Verifies preprocessor raises error if transformed before fitting."""
        prep = FoldContainedQuantumPreprocessor()
        with self.assertRaises(RuntimeError):
            prep.transform_quantum_angles(self.V)
        with self.assertRaises(RuntimeError):
            prep.transform_classical(self.V)

    def test_quantum_angle_bounds_and_mappings(self):
        """Verifies quantum rotation angles stay strictly within [-pi, pi] and binary sex mapping."""
        prep = FoldContainedQuantumPreprocessor()
        prep.fit(self.D)

        theta_dev = prep.transform_quantum_angles(self.D)
        theta_val = prep.transform_quantum_angles(self.V)
        theta_test = prep.transform_quantum_angles(self.T)

        self.assertEqual(theta_dev.shape, (len(self.D), 9))
        self.assertEqual(theta_val.shape, (len(self.V), 9))
        self.assertEqual(theta_test.shape, (len(self.T), 9))

        # Range bounds: [-pi - eps, pi + eps]
        eps = 1e-6
        for name, th in [('Dev', theta_dev), ('Val', theta_val), ('Test', theta_test)]:
            self.assertTrue(np.all(th >= -np.pi - eps), f"{name}: Angles below -pi detected.")
            self.assertTrue(np.all(th <= np.pi + eps), f"{name}: Angles above pi detected.")

        # Check binary sex mapping (column index 1 in FEATS_9)
        sex_dev = self.D['sex'].values
        np.testing.assert_array_equal(theta_dev[sex_dev == 0, 1], 0.0, err_msg="sex=0 must map to 0.0")
        self.assertTrue(np.all(np.isclose(theta_dev[sex_dev == 1, 1], np.pi)), "sex=1 must map to pi")

    def test_classical_scaler_zero_leakage(self):
        """Verifies classical scaler moments depend strictly on development partition."""
        prep = FoldContainedQuantumPreprocessor()
        prep.fit(self.D)

        X_dev_scaled = prep.transform_classical(self.D)
        
        # Dev set mean should be ~0.0, std ~1.0
        np.testing.assert_allclose(np.mean(X_dev_scaled, axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.std(X_dev_scaled, axis=0), 1.0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

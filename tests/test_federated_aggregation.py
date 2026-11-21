"""
Unit tests for decentralized federated learning primitives.
Verifies:
  - Exact mathematical parity between decentralized sufficient statistics aggregation and StandardScaler (< 1e-12).
  - 26-client autonomous edge partitioning (one client per patient, 100% development record allocation).
  - Federated server aggregation weights normalization.
"""

import os
import sys
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'experiments', 'phase4', 'code'))

from common import load, make_sets, FEATS
from phase4_common import decentralized_standardization, get_natural_26_clients


class TestFederatedAggregation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load()
        cls.D, cls.V, cls.T, cls.thr = make_sets(cls.df, seed=0)
        cls.clients = get_natural_26_clients(cls.D)

    def test_client_allocation_exactness(self):
        """Verifies exactly 26 clients created, each corresponding to 1 unique subject."""
        self.assertEqual(len(self.clients), 26, "Must create exactly 26 edge clients.")
        
        assigned_subjects = set()
        total_client_records = 0
        for c in self.clients:
            self.assertIn('subject_id', c, "Each edge client must contain 'subject_id'.")
            assigned_subjects.add(c['subject_id'])
            total_client_records += c['n_records']

        self.assertEqual(len(assigned_subjects), 26, "Must cover all 26 development subjects.")
        self.assertEqual(total_client_records, len(self.D), "Total client records must equal development record count.")

    def test_decentralized_standardization_parity_with_standard_scaler(self):
        """Verifies sufficient statistics aggregation matches sklearn StandardScaler to < 1e-12."""
        client_X_list = [self.D.iloc[c['indices']][FEATS].values for c in self.clients]
        XV = self.V[FEATS].values
        XT = self.T[FEATS].values

        # 1. Decentralized sufficient statistics aggregation
        ZD_clients, ZV_dec, ZT_dec, mean_dec, std_dec = decentralized_standardization(client_X_list, XV, XT)

        # 2. Centralized StandardScaler
        scaler = StandardScaler()
        XD_all = self.D[FEATS].values
        scaler.fit(XD_all)
        ZV_cent = scaler.transform(XV)
        ZT_cent = scaler.transform(XT)

        # Check mean and scale parity
        np.testing.assert_allclose(mean_dec, scaler.mean_, atol=1e-12, err_msg="Mean mismatch between decentralized and centralized scaler.")
        np.testing.assert_allclose(std_dec, scaler.scale_, atol=1e-12, err_msg="Std mismatch between decentralized and centralized scaler.")

        # Check transformed validation and test data parity
        np.testing.assert_allclose(ZV_dec, ZV_cent, atol=1e-12, err_msg="Validation data transform mismatch.")
        np.testing.assert_allclose(ZT_dec, ZT_cent, atol=1e-12, err_msg="Test data transform mismatch.")

    def test_server_aggregation_weights(self):
        """Verifies FedAvg sample-weighted aggregation weights sum to 1.0."""
        total_n = len(self.D)
        weights = [c['n_records'] / total_n for c in self.clients]
        self.assertAlmostEqual(sum(weights), 1.0, places=9, msg="Client aggregation weights must sum to 1.0.")
        self.assertTrue(all(w > 0 for w in weights), "All client weights must be strictly positive.")


if __name__ == '__main__':
    unittest.main()

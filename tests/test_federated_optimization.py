"""
Unit tests for federated optimization, FedAvg, FedProx, and multicenter aggregation
in experiments/phase4/code/run_phase4_federated.py and phase4_common.py.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4_CODE_DIR = os.path.join(ROOT_DIR, "experiments", "phase4", "code")
if PHASE4_CODE_DIR not in sys.path:
    sys.path.insert(0, PHASE4_CODE_DIR)

from run_phase4_federated import run_federated_logistic, sigmoid
from phase4_common import get_multicenter_clients


class TestFederatedOptimization(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.d = 5
        self.n_clients = 4
        self.samples_per_client = 20

        # Construct synthetic client data
        self.client_data = []
        all_X = []
        all_y = []

        for cid in range(self.n_clients):
            Xc = np.random.randn(self.samples_per_client, self.d)
            # Make client 0 mono-class (y=1) to test extreme label skew
            if cid == 0:
                yc = np.ones(self.samples_per_client)
            elif cid == 1:
                yc = np.zeros(self.samples_per_client)
            else:
                yc = (np.dot(Xc, np.ones(self.d)) > 0).astype(float)

            self.client_data.append({
                "client_id": cid,
                "X": Xc,
                "y": yc,
                "n_samples": len(yc)
            })
            all_X.append(Xc)
            all_y.append(yc)

        self.ZD = np.vstack(all_X)
        self.yD = np.concatenate(all_y)
        self.ZV = np.random.randn(15, self.d)
        self.yV = np.random.randint(0, 2, size=15).astype(float)
        self.ZT = np.random.randn(15, self.d)
        self.yT = np.random.randint(0, 2, size=15).astype(float)

    def test_sigmoid_numerical_stability(self):
        """Verifies sigmoid function handles extreme values without overflow."""
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertAlmostEqual(sigmoid(100.0), 1.0)
        self.assertAlmostEqual(sigmoid(-100.0), 0.0)

    def test_fedavg_execution_and_weight_dimensions(self):
        """Verifies FedAvg runs across multiple rounds and produces valid parameter vectors."""
        w, b, history = run_federated_logistic(
            self.client_data, self.ZV, self.yV, self.ZT, self.yT, self.ZD, self.yD,
            frac=1.0, mu=0.0, rounds=3, E=2, lr=0.05, seed=42
        )

        self.assertEqual(len(w), self.d, "Weight vector must match feature dimension.")
        self.assertFalse(np.any(np.isnan(w)), "Weights must not contain NaN.")
        self.assertFalse(np.isnan(b), "Bias must not be NaN.")
        self.assertEqual(len(history), 3, "History must record exactly 3 rounds.")

        first_round = history[0]
        self.assertIn("round", first_round)
        self.assertIn("val_auc", first_round)
        self.assertIn("test_auc", first_round)

    def test_fedprox_proximal_regularization_effect(self):
        """Verifies proximal term mu > 0 constrains local model divergence."""
        # Run FedAvg (mu = 0.0)
        w_fedavg, b_fedavg, _ = run_federated_logistic(
            self.client_data, self.ZV, self.yV, self.ZT, self.yT, self.ZD, self.yD,
            frac=1.0, mu=0.0, rounds=2, E=5, lr=0.1, seed=42
        )

        # Run FedProx with strong penalty (mu = 5.0)
        w_fedprox, b_fedprox, _ = run_federated_logistic(
            self.client_data, self.ZV, self.yV, self.ZT, self.yT, self.ZD, self.yD,
            frac=1.0, mu=5.0, rounds=2, E=5, lr=0.1, seed=42
        )

        # Proximal penalty restricts update norm relative to unconstrained gradient descent
        norm_fedavg = np.linalg.norm(w_fedavg)
        norm_fedprox = np.linalg.norm(w_fedprox)
        self.assertLess(norm_fedprox, norm_fedavg, "FedProx with strong mu must produce smaller parameter norm.")

    def test_partial_client_participation(self):
        """Verifies 50% client participation (frac=0.5) successfully executes and updates weights."""
        w, b, history = run_federated_logistic(
            self.client_data, self.ZV, self.yV, self.ZT, self.yT, self.ZD, self.yD,
            frac=0.5, mu=0.0, rounds=4, E=2, lr=0.05, seed=42
        )
        self.assertEqual(len(w), self.d)
        self.assertEqual(len(history), 4)

    def test_multicenter_partitioning(self):
        """Verifies get_multicenter_clients partitions development dataframe into K centers."""
        df_dev = pd.DataFrame({
            "subject#": np.repeat(range(1, 11), 10),
            "age": np.random.randint(50, 75, 100),
            "motor_UPDRS": np.random.uniform(10.0, 35.0, 100),
            "y": np.random.randint(0, 2, 100)
        })

        centers = get_multicenter_clients(df_dev, K=3, mode="natural", seed=0)
        self.assertEqual(len(centers), 3, "Must create exactly 3 centers.")
        total_records = sum(c["n_records"] for c in centers)
        self.assertEqual(total_records, len(df_dev), "Center records must partition all development records.")


if __name__ == "__main__":
    unittest.main()

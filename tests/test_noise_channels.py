"""
Unit tests for quantum noise channels, Kraus operator trace preservation,
and noisy circuit evaluators in experiments/phase5/code/noise_simulation.py.
"""

import sys
import os
import unittest
import numpy as np
import torch
import pennylane as qml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE5_CODE_DIR = os.path.join(ROOT_DIR, "experiments", "phase5", "code")
if PHASE5_CODE_DIR not in sys.path:
    sys.path.insert(0, PHASE5_CODE_DIR)

from noise_simulation import create_noisy_evaluator, evaluate_vqc_under_noise, evaluate_pqcr_under_noise
from quantum_classifier import VQCModel
from quantum_regressor import PQCRModel


class TestQuantumNoiseChannels(unittest.TestCase):

    def test_depolarizing_kraus_trace_preservation(self):
        """Verifies single-qubit depolarizing channel Kraus operators sum to Identity."""
        p = 0.05
        # Standard PennyLane/Nielsen-Chuang depolarizing channel Kraus representation:
        # E(rho) = (1 - p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
        K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p / 3.0) * np.array([[0, 1], [1, 0]], dtype=complex)
        K2 = np.sqrt(p / 3.0) * np.array([[0, -1j], [1j, 0]], dtype=complex)
        K3 = np.sqrt(p / 3.0) * np.array([[1, 0], [0, -1]], dtype=complex)

        kraus_ops = [K0, K1, K2, K3]
        completeness = sum(K.conj().T @ K for K in kraus_ops)
        np.testing.assert_allclose(completeness, np.eye(2, dtype=complex), atol=1e-12)

    def test_phase_damping_kraus_and_population_preservation(self):
        """
        Verifies phase damping channel satisfies completeness and strictly preserves
        computational basis populations (Pauli-Z expectations).
        """
        gamma = 0.1
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(gamma)]], dtype=complex)

        # 1. Trace preservation: K0^dag K0 + K1^dag K1 = I
        completeness = K0.conj().T @ K0 + K1.conj().T @ K1
        np.testing.assert_allclose(completeness, np.eye(2, dtype=complex), atol=1e-12)

        # 2. Population preservation on an arbitrary mixed single-qubit density matrix
        # rho = [[a, b], [b*, c]] with a + c = 1
        rho = np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=complex)
        rho_damped = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

        # Diagonal elements (populations) must remain identical
        self.assertAlmostEqual(rho[0, 0].real, rho_damped[0, 0].real, places=12)
        self.assertAlmostEqual(rho[1, 1].real, rho_damped[1, 1].real, places=12)

        # Off-diagonal coherences must be damped by sqrt(1 - gamma)
        expected_off_diag = rho[0, 1] * np.sqrt(1.0 - gamma)
        self.assertAlmostEqual(rho_damped[0, 1], expected_off_diag, places=12)

    def test_create_noisy_evaluator_circuit_output_bounds(self):
        """Verifies noisy evaluator generates expectation values within [-1, 1]."""
        # Test with 3 qubits for speed
        n_qubits = 3
        depth_L = 1
        evaluator_depol = create_noisy_evaluator(
            n_qubits=n_qubits, depth_L=depth_L, noise_type="depol", noise_param=0.01
        )
        weights = torch.zeros((depth_L, n_qubits, 2), dtype=torch.float64)
        x = torch.zeros(n_qubits, dtype=torch.float64)

        expvals = evaluator_depol(weights, x)
        self.assertEqual(len(expvals), n_qubits)
        for val in expvals:
            self.assertTrue(-1.0 <= float(val) <= 1.0)

    def test_evaluate_vqc_under_noise_synthetic_batch(self):
        """Verifies evaluate_vqc_under_noise with small synthetic batch."""
        model = VQCModel(n_qubits=9, depth_L=2, seed=42)
        X_syn = np.zeros((2, 9), dtype=np.float64)
        y_syn = np.array([0, 1], dtype=np.int32)

        metrics, probs = evaluate_vqc_under_noise(
            model, X_syn, y_syn, noise_type="depol", noise_param=0.001, batch_size=2
        )
        self.assertEqual(len(probs), 2)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertIn("roc_auc", metrics)
        self.assertIn("bacc", metrics)
        self.assertIn("eval_time_s", metrics)
        self.assertGreater(metrics["eval_time_s"], 0.0)

    def test_evaluate_pqcr_under_noise_synthetic_batch(self):
        """Verifies evaluate_pqcr_under_noise with small synthetic batch."""
        model = PQCRModel(n_qubits=9, depth_L=2, seed=42)
        X_syn = np.zeros((2, 9), dtype=np.float64)
        y_syn = np.array([15.0, 25.0], dtype=np.float64)

        metrics, preds = evaluate_pqcr_under_noise(
            model, X_syn, y_syn, noise_type="dephase", noise_param=0.01, batch_size=2
        )
        self.assertEqual(len(preds), 2)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("eval_time_s", metrics)
        self.assertFalse(np.isnan(metrics["mae"]))


if __name__ == "__main__":
    unittest.main()

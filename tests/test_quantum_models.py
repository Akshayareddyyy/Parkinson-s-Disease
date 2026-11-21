"""
Unit tests for 9-Qubit Quantum Machine Learning architectures (VQC and PQC-R).
Verifies:
  - Circuit and total trainable parameter reconciliation (L=2 -> 46, L=3 -> 64).
  - Forward pass output dimensions and probability ranges.
  - Autograd backpropagation and non-vanishing gradient flow.
"""

import os
import sys
import unittest
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'experiments', 'phase5', 'code'))

from quantum_classifier import VQCModel
from quantum_regressor import PQCRModel


class TestQuantumModels(unittest.TestCase):

    def test_vqc_parameter_counts(self):
        """Verifies 9-qubit VQC has 46 parameters for L=2 and 64 for L=3."""
        # L=2: 2 layers * 9 qubits * 2 angles = 36 circuit params + 10 readout = 46
        vqc_l2 = VQCModel(n_qubits=9, depth_L=2, seed=42)
        total_l2 = sum(p.numel() for p in vqc_l2.parameters() if p.requires_grad)
        circuit_l2 = vqc_l2.variational_weights.numel()
        readout_l2 = vqc_l2.readout.weight.numel() + vqc_l2.readout.bias.numel()

        self.assertEqual(circuit_l2, 36, "VQC L=2 must have 36 circuit rotation angles.")
        self.assertEqual(readout_l2, 10, "VQC readout must have 10 parameters (9 weights + 1 bias).")
        self.assertEqual(total_l2, 46, "VQC L=2 must have exactly 46 total trainable parameters.")

        # L=3: 3 layers * 9 qubits * 2 angles = 54 circuit params + 10 readout = 64
        vqc_l3 = VQCModel(n_qubits=9, depth_L=3, seed=42)
        total_l3 = sum(p.numel() for p in vqc_l3.parameters() if p.requires_grad)
        circuit_l3 = vqc_l3.variational_weights.numel()

        self.assertEqual(circuit_l3, 54, "VQC L=3 must have 54 circuit rotation angles.")
        self.assertEqual(total_l3, 64, "VQC L=3 must have exactly 64 total trainable parameters.")

    def test_pqcr_parameter_counts(self):
        """Verifies 9-qubit PQC-R has 46 parameters for L=2 and 64 for L=3."""
        pqcr_l2 = PQCRModel(n_qubits=9, depth_L=2, seed=42)
        total_l2 = sum(p.numel() for p in pqcr_l2.parameters() if p.requires_grad)
        self.assertEqual(total_l2, 46, "PQC-R L=2 must have exactly 46 total trainable parameters.")

        pqcr_l3 = PQCRModel(n_qubits=9, depth_L=3, seed=42)
        total_l3 = sum(p.numel() for p in pqcr_l3.parameters() if p.requires_grad)
        self.assertEqual(total_l3, 64, "PQC-R L=3 must have exactly 64 total trainable parameters.")

    def test_vqc_forward_and_backward_pass(self):
        """Verifies batch forward pass output shapes, probability bounds, and autograd gradient flow."""
        vqc = VQCModel(n_qubits=9, depth_L=2, seed=0)
        batch_size = 4
        x = torch.zeros((batch_size, 9), dtype=torch.float64)

        logits, probs, qfeat = vqc(x)

        self.assertEqual(logits.shape, (batch_size, 1), "Logits shape must be (batch_size, 1).")
        self.assertEqual(probs.shape, (batch_size, 1), "Probs shape must be (batch_size, 1).")
        self.assertEqual(qfeat.shape, (batch_size, 9), "Quantum features shape must be (batch_size, 9).")

        # Probs must be in [0, 1]
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0), "Probabilities must be in [0, 1].")

        # Expectation values must be in [-1, 1]
        self.assertTrue(torch.all(qfeat >= -1.0) and torch.all(qfeat <= 1.0), "Pauli-Z expectations must be in [-1, 1].")

        # Backward pass
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(vqc.variational_weights.grad, "Gradients must exist after backward pass.")
        grad_norm = vqc.variational_weights.grad.norm().item()
        self.assertGreater(grad_norm, 0.0, "Gradient norm must be strictly positive.")

    def test_pqcr_forward_and_backward_pass(self):
        """Verifies PQC-R regression forward and backward pass."""
        pqcr = PQCRModel(n_qubits=9, depth_L=2, seed=0)
        batch_size = 4
        x = torch.zeros((batch_size, 9), dtype=torch.float64)

        preds, qfeat = pqcr(x)

        self.assertEqual(preds.shape, (batch_size, 1), "Predictions shape must be (batch_size, 1).")
        self.assertEqual(qfeat.shape, (batch_size, 9), "Quantum features shape must be (batch_size, 9).")

        loss = preds.sum()
        loss.backward()
        self.assertIsNotNone(pqcr.variational_weights.grad, "Gradients must exist after backward pass.")
        grad_norm = pqcr.variational_weights.grad.norm().item()
        self.assertGreater(grad_norm, 0.0, "Gradient norm must be strictly positive.")


if __name__ == '__main__':
    unittest.main()

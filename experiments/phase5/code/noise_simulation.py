"""
Phase 5: Simulated Quantum Noise Resilience Evaluation
Evaluates frozen VQC and PQC-R models under:
  1. Depolarizing Noise Channel (error rates p in [1e-4, 1e-3, 5e-3, 1e-2])
  2. Phase Damping / Dephasing Channel (gamma in [1e-3, 1e-2, 5e-2])
  3. Finite Measurement Shots (S in [1000, 4000, 10000])
All results are explicitly labeled as classical numerical simulations (no hardware claims).
"""

import time
import numpy as np
import torch
import pennylane as qml

from metrics import evaluate_classification, evaluate_regression

def create_noisy_evaluator(n_qubits=9, depth_L=2, noise_type='depol', noise_param=0.001, shots=None):
    """
    Creates a noisy PennyLane QNode matching the HEA ansatz with frozen weights.
    """
    if noise_type in ['depol', 'dephase']:
        dev = qml.device('default.mixed', wires=n_qubits)
    else:
        # Shots evaluation
        dev = qml.device('default.qubit', wires=n_qubits, shots=shots)
        
    @qml.qnode(dev, interface='torch')
    def circuit(weights, x):
        # Angle encoding
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
            if noise_type == 'depol':
                qml.DepolarizingChannel(noise_param, wires=i)
            elif noise_type == 'dephase':
                qml.PhaseDamping(noise_param, wires=i)
                
        # HEA layers
        for l in range(depth_L):
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
                if noise_type == 'depol':
                    qml.DepolarizingChannel(noise_param, wires=i)
                elif noise_type == 'dephase':
                    qml.PhaseDamping(noise_param, wires=i)
                    
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if noise_type == 'depol':
                    qml.DepolarizingChannel(noise_param, wires=i)
                    qml.DepolarizingChannel(noise_param, wires=i + 1)
            qml.CNOT(wires=[n_qubits - 1, 0])
            if noise_type == 'depol':
                qml.DepolarizingChannel(noise_param, wires=n_qubits - 1)
                qml.DepolarizingChannel(noise_param, wires=0)
                
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
    return circuit

def evaluate_vqc_under_noise(vqc_model, X_te, y_te, noise_type, noise_param=None, shots=None, batch_size=32):
    """
    Evaluates a frozen VQC model on test data under a specific noise configuration.
    """
    weights = vqc_model.variational_weights.detach()
    readout_w = vqc_model.readout.weight.detach()
    readout_b = vqc_model.readout.bias.detach()
    
    noisy_circuit = create_noisy_evaluator(
        n_qubits=vqc_model.n_qubits, depth_L=vqc_model.depth_L,
        noise_type=noise_type, noise_param=noise_param, shots=shots
    )
    
    X_te_t = torch.tensor(X_te, dtype=torch.float64)
    n_samples = X_te_t.shape[0]
    
    q_features_list = []
    t0 = time.perf_counter()
    
    # Process in mini-batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        xb = X_te_t[start_idx:end_idx]
        xb_T = xb.T
        
        expvals = noisy_circuit(weights, xb_T)
        if xb.shape[0] == 1:
            q_feat = torch.stack(expvals).unsqueeze(0)
        else:
            q_feat = torch.stack(expvals, dim=1)
        q_features_list.append(q_feat)
        
    all_q_feat = torch.cat(q_features_list, dim=0)
    logits = all_q_feat @ readout_w.T + readout_b
    probs = torch.sigmoid(logits).numpy().flatten()
    eval_time = time.perf_counter() - t0
    
    metrics = evaluate_classification(y_te, probs)
    metrics['eval_time_s'] = eval_time
    return metrics, probs

def evaluate_pqcr_under_noise(pqcr_model, X_te, y_te, noise_type, noise_param=None, shots=None, batch_size=32):
    """
    Evaluates a frozen PQC-R model on test data under a specific noise configuration.
    """
    weights = pqcr_model.variational_weights.detach()
    readout_w = pqcr_model.readout.weight.detach()
    readout_b = pqcr_model.readout.bias.detach()
    
    noisy_circuit = create_noisy_evaluator(
        n_qubits=pqcr_model.n_qubits, depth_L=pqcr_model.depth_L,
        noise_type=noise_type, noise_param=noise_param, shots=shots
    )
    
    X_te_t = torch.tensor(X_te, dtype=torch.float64)
    n_samples = X_te_t.shape[0]
    
    q_features_list = []
    t0 = time.perf_counter()
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        xb = X_te_t[start_idx:end_idx]
        xb_T = xb.T
        
        expvals = noisy_circuit(weights, xb_T)
        if xb.shape[0] == 1:
            q_feat = torch.stack(expvals).unsqueeze(0)
        else:
            q_feat = torch.stack(expvals, dim=1)
        q_features_list.append(q_feat)
        
    all_q_feat = torch.cat(q_features_list, dim=0)
    preds = (all_q_feat @ readout_w.T + readout_b).numpy().flatten()
    eval_time = time.perf_counter() - t0
    
    metrics = evaluate_regression(y_te, preds)
    metrics['eval_time_s'] = eval_time
    return metrics, preds


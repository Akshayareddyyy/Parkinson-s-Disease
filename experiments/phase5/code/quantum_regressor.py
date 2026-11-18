"""
Phase 5: 9-Qubit Parameterized Quantum Regressor (PQC-R)
Architecture:
  - 9 Qubits
  - State Preparation: Angle Encoding Rx(theta_j)
  - Hardware-Efficient Ansatz (HEA):
      L layers of Ry(theta_1) Rz(theta_2) + Circular CNOT Ladder
  - Observable: PauliZ expectation values on all 9 qubits
  - Readout: Linear layer w^T <Z> + b -> Continuous prediction y_hat
  - Vectorized mini-batch parameter broadcasting via PyTorch autograd
"""

import time, math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

class PQCRModel(nn.Module):
    def __init__(self, n_qubits=9, depth_L=2, seed=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth_L = depth_L
        
        # Initialize PennyLane device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Build vectorized QNode
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(weights, x):
            # x shape: (n_qubits, batch_size)
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
            for l in range(depth_L):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.circuit = circuit
        
        # Variational angles initialized uniformly near zero
        torch.manual_seed(seed)
        init_angles = (torch.rand(depth_L, n_qubits, 2, dtype=torch.float64) - 0.5) * 0.1 * math.pi
        self.variational_weights = nn.Parameter(init_angles)
        
        # Linear readout layer (9 expectation values -> 1 continuous prediction)
        self.readout = nn.Linear(n_qubits, 1, dtype=torch.float64)
        nn.init.xavier_uniform_(self.readout.weight)
        # Initialize bias to population mean scale (~20.0)
        nn.init.constant_(self.readout.bias, 20.0)
        
    def forward(self, x):
        """
        x: tensor of shape (batch_size, n_qubits)
        Returns: continuous predictions y_hat of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        x_T = x.T
        expvals = self.circuit(self.variational_weights, x_T)
        
        if batch_size == 1:
            q_features = torch.stack(expvals).unsqueeze(0)
        else:
            q_features = torch.stack(expvals, dim=1)
            
        preds = self.readout(q_features)
        return preds, q_features

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train_pqcr(
    model, X_tr, y_tr, X_val, y_val,
    epochs=30, batch_size=64, lr=0.01, weight_decay=1e-4, patience=5
):
    """
    Trains PQC-R using Adam optimizer, Cosine Annealing, and Validation Early Stopping on MSE loss.
    """
    device = torch.device('cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float64, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float64, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float64, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float64, device=device).unsqueeze(1)
    
    n_samples = X_tr_t.shape[0]
    best_val_loss = float('inf')
    best_model_weights = None
    best_epoch = 0
    patience_counter = 0
    
    history = []
    t_start_total = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        
        grad_norms = []
        grad_variances = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]
            
            xb = X_tr_t[batch_idx]
            yb = y_tr_t[batch_idx]
            
            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"NaN/Inf loss detected at epoch {epoch}, batch {start_idx}")
                
            loss.backward()
            
            # Gradient diagnostics
            total_norm = 0.0
            grad_list = []
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    grad_list.append(p.grad.data.cpu().numpy().flatten())
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            all_grads = np.concatenate(grad_list)
            grad_variances.append(float(np.var(all_grads)))
            
            optimizer.step()
            epoch_loss += loss.item() * (end_idx - start_idx)
            
        scheduler.step()
        train_loss = epoch_loss / n_samples
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_preds, _ = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            
        t_epoch = time.perf_counter() - t_epoch_start
        mean_grad_norm = float(np.mean(grad_norms))
        mean_grad_var = float(np.mean(grad_variances))
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mean_grad_norm': mean_grad_norm,
            'mean_grad_var': mean_grad_var,
            'epoch_time_s': t_epoch
        })
        
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    total_train_time = time.perf_counter() - t_start_total
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return {
        'model': model,
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_train_time': total_train_time
    }

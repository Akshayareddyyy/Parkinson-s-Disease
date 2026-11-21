# Experimental Progression & Reproduction Guide

This directory houses the source code, raw result logs, model checkpoints, audit reports, and analytical figures for all formal phases of the Parkinson's motor severity study:

- `phase3b/`: Advanced classical ML algorithms, loss functions, and evaluation metrics.
- `phase4/`: Federated learning under simulated hospital edge topology and non-IID client heterogeneity.
- `phase4b/`: Feature optimization, personalized regression calibration windows, and federated improvements.
- `phase5/`: 9-qubit Variational Quantum Classifiers (VQC) and Parameterized Quantum Circuit Regressors (PQC-R) under noiseless and simulated noisy channels.

---

## 1. Directory Structure

```
experiments/
├── phase3b/
│   ├── code/                   # Training and evaluation scripts (Exp 1 - Exp 5)
│   ├── results/                # Raw per-seed runs and aggregated CSV summaries
│   ├── figures/                # Publication figures (loss, feature selection, models)
│   └── audit/                  # Methodology audits and reproducibility verification
├── phase4/
│   ├── code/                   # Federated learning algorithms (FedAvg, FedProx, FedBN)
│   ├── results/                # Per-seed FL convergence histories and client summaries
│   ├── figures/                # Heterogeneity, communication tradeoffs, convergence plots
│   ├── audit/                  # Fairness protocols and audits
│   └── independent_verification/# Verification scripts, matched comparisons, participation audit
├── phase4b/
│   ├── code/                   # Advanced ML, personalized calibration, federated tuning
│   ├── results/                # Complete CSV logs for Exp 1 - Exp 5
│   ├── figures/                # Visualizations (personalized windows, feature engineering)
│   ├── audit/                  # Pre- and post-execution fairness audits
│   └── final_verification/     # Verification reports, paired-seed comparisons, control matrices
└── phase5/
    ├── code/                   # VQC, PQC-R, PennyLane noise simulations, preprocessors
    ├── checkpoints/            # 20 frozen PyTorch checkpoints (.pt) for 10 seeds (L=2, L=3)
    ├── results/                # Per-seed CSV logs, summaries, stratified noise outputs
    ├── figures/                # Gradient flow, classical vs quantum comparisons, noise resilience
    ├── logs/                   # Execution stdout logs and memory benchmarks
    └── audit/                  # Final consistency audit, manuscript-readiness report
```

---

## 2. Experimental Phases & Reproduction Commands

### Phase 3B: Advanced Classical Machine Learning
Evaluated whether balanced loss weighting, non-linear tree ensembles, and feature pruning improve motor UPDRS prediction.
- **Entry Points:**
  ```powershell
  python experiments/phase3b/code/run_exp1_subject_weighting.py
  python experiments/phase3b/code/run_exp2_improved_algorithms.py
  python experiments/phase3b/code/run_exp3_feature_engineering.py
  python experiments/phase3b/code/run_exp4_nested_cv.py
  python experiments/phase3b/code/run_exp5_personalized_calibration.py
  python experiments/phase3b/code/run_phase3b_master.py
  ```
- **Key Findings:** Extra Trees achieved the highest classification ROC-AUC (0.6685 ± 0.0827). Linear regression models suffered from high variance across cross-subject test folds.

### Phase 4: Federated Learning Topology
Simulated 26 hospital edge clients (one client per development patient) with non-IID single-class data distributions ($y_i \in \{0\}$ or $y_i \in \{1\}$ for several clients).
- **Entry Points:**
  ```powershell
  python experiments/phase4/code/run_phase4_federated.py
  python experiments/phase4/code/plot_phase4_figures.py
  python experiments/phase4/independent_verification/verify_phase4.py
  ```
- **Key Findings:** Standard FedAvg experienced client-level gradient drift due to mono-class clients. Proposed *Safe Local Class-Weighted FedAvg* stabilized training, achieving ROC-AUC 0.6414 ± 0.1118 across 26 clients.

### Phase 4B: Personalized Calibration & Feature Optimization
Evaluated patient-specific calibration windows ($k \in [3, 15]$ recordings per subject) to overcome cross-patient acoustic baseline variance.
- **Entry Points:**
  ```powershell
  python experiments/phase4b/code/run_exp1_advanced_classification.py
  python experiments/phase4b/code/run_exp2_advanced_regression.py
  python experiments/phase4b/code/run_exp3_feature_engineering.py
  python experiments/phase4b/code/run_exp4_federated_improvement.py
  python experiments/phase4b/code/run_exp5_personalized_prediction.py
  python experiments/phase4b/code/run_phase4b_master.py
  ```
- **Key Findings:** Patient-specific offset calibration dramatically dropped regression error from MAE ~6.9 to MAE ~2.7. However, rigorous audit against a non-parametric *Mean Drift Dummy* revealed that acoustic regression models captured minimal true functional progression beyond historical subject mean tracking.

### Phase 5: Quantum Machine Learning (VQC & PQC-R)
Evaluated 9-qubit hardware-matched quantum variational circuits for classification (VQC) and regression (PQC-R) using PennyLane and PyTorch.
- **Circuit Architecture:**
  - 9 qubits matching the 9 decorrelated clinical features (`FEATS_9`).
  - Angle embedding: $R_y(\phi)$ rotations where continuous features are normalized to $[-\pi, \pi]$ and biological sex is encoded as $\{0.0, \pi\}$.
  - Variational ansatz: Strongly entangling layers with $R_x(\theta_1)$, $R_z(\theta_2)$, and circular CNOT entanglement gates.
  - Parameter counts: Depth $L=2 \implies 46$ trainable parameters (36 circuit + 10 readout); Depth $L=3 \implies 64$ trainable parameters (54 circuit + 10 readout).
- **Noise Simulation:**
  - Depolarizing noise channel ($\lambda \in [0.001, 0.05]$): contracts Bloch vector toward maximally mixed state $I/2$.
  - Phase damping noise channel ($\gamma \in [0.01, 0.2]$): suppresses off-diagonal coherences while strictly preserving diagonal populations ($\langle Z \rangle$ expectations).
  - Finite-shot measurement sampling ($S \in [100, 10000]$): introduces statistical variance scaling as $\mathcal{O}(1/\sqrt{S})$.
- **Entry Points:**
  ```powershell
  # Pre-execution environment and hardware checks
  python experiments/phase5/code/pre_execution_audit.py
  python experiments/phase5/code/smoke_test.py

  # Primary quantum experiments
  python experiments/phase5/code/run_classification.py
  python experiments/phase5/code/run_regression.py
  python experiments/phase5/code/run_stratified_noise.py
  python experiments/phase5/code/generate_figures.py
  ```

---

## 3. Pre-Trained Model Checkpoints

Phase 5 includes 20 fully trained PyTorch model checkpoints stored in `experiments/phase5/checkpoints/`:

- **VQC Classification Checkpoints (10 seeds):**
  - `vqc_seed_0.pt` through `vqc_seed_9.pt`
  - Validation-selected depth configuration: Seeds 0, 1, 2, 4, 7, 9 selected Depth $L=2$; Seeds 3, 5, 6, 8 selected Depth $L=3$.
- **PQC-R Regression Checkpoints (10 seeds):**
  - `pqcr_seed_0.pt` through `pqcr_seed_9.pt`
  - Validation-selected depth configuration: Seeds 1, 3, 5, 8 selected Depth $L=2$; Seeds 0, 2, 4, 6, 7, 9 selected Depth $L=3$.

Each checkpoint contains:
```python
checkpoint = {
    'epoch': int,
    'model_state_dict': dict,
    'val_loss': float,
    'val_metric': float,
    'depth_L': int,
    'n_qubits': 9,
    'seed': int
}
```

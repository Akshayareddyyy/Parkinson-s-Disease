# Phase 5: Pre-Execution Verification Checklist & Final Parity Matrix
## Methodological Audit for Exactly Two Quantum Models (VQC & PQC-R)

**Date:** 2026-11-14  
**Author:** Antigravity Research Team  
**Governing Document:** [`experiment_plan.md`](file:///c:/Users/akank/Downloads/parkinsons_severity/experiments/phase5/experiment_plan.md)  
**Related Artifacts:** [`final_model_feature_parity_matrix.csv`](file:///c:/Users/akank/Downloads/parkinsons_severity/experiments/phase5/final_model_feature_parity_matrix.csv)  
**Status:** DRAFT CHECKLIST — AWAITING FORMAL USER APPROVAL (ZERO CODE WRITTEN, ZERO TRAINING EXECUTED)  

---

## 1. Executive Summary & Revised Scope

This document certifies that Phase 5 has been restructured to evaluate **exactly TWO quantum machine-learning models**:

1. **Model 1: 9-Qubit Variational Quantum Classifier (VQC)** for binary motor severity classification.
   - Matched classical controls: `ExtraTreesClassifier` (9 features) and `Classical RBF-SVM` (9 features).
2. **Model 2: 9-Qubit Parameterized Quantum Regressor (PQC-R)** for continuous motor-UPDRS prediction.
   - Matched classical controls: `ElasticNet` (9 features) and `Classical RBF-SVR` (9 features).

### Permanently Excluded from Implementation:
- [x] QSVC (removed).
- [x] 4-qubit VQC and PQC-R variants (removed).
- [x] Personalized PQC-Delta (removed).
- [x] Standalone quantum kernel experiments (removed).
- [x] Any other quantum model architectures (strictly prohibited).

---

## 2. Audited Partition Distributions & Record Counts (Seeds 0–9)

The plan explicitly accounts for variable recording densities across the 10 audited seeds (ranging from 101 to 168 records per subject). Sample counts are verified directly from `parkinsons_updrs.csv` (5,875 total records, 42 unique subjects):

### Seed-by-Seed Record Distribution:
| Seed | Dev Subjects | Val Subjects | Test Subjects | Dev Records | Val Records | Locked-Test Records | Total Records |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 26 | 7 | 9 | **3,578** | 991 | 1,306 | 5,875 |
| **1** | 26 | 7 | 9 | **3,565** | 1,055 | 1,255 | 5,875 |
| **2** | 26 | 7 | 9 | **3,652** | 978 | 1,245 | 5,875 |
| **3** | 26 | 7 | 9 | **3,692** | 939 | 1,244 | 5,875 |
| **4** | 26 | 7 | 9 | **3,565** | 1,025 | 1,285 | 5,875 |
| **5** | 26 | 7 | 9 | **3,676** | 934 | 1,265 | 5,875 |
| **6** | 26 | 7 | 9 | **3,697** | 976 | 1,202 | 5,875 |
| **7** | 26 | 7 | 9 | **3,517** | 1,033 | 1,325 | 5,875 |
| **8** | 26 | 7 | 9 | **3,678** | 982 | 1,215 | 5,875 |
| **9** | 26 | 7 | 9 | **3,650** | 889 | 1,336 | 5,875 |
| **Mean $\pm$ SD** | **26.0 $\pm$ 0.0** | **7.0 $\pm$ 0.0** | **9.0 $\pm$ 0.0** | **3,627.0 $\pm$ 61.9** | **980.2 $\pm$ 49.3** | **1,267.8 $\pm$ 44.4** | **5,875** |

### Verified Partition Properties:
- [x] **Zero Subject Overlap:** $\mathcal{S}_{\text{dev}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{dev}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$ in all 10 seeds.
- [x] **No Arbitrary Downsampling:** Every development record allocated to a seed is passed to the optimizer.
- [x] **Preserved Subject Alignment:** The identical subject groupings from Phase 3, 3B, 4, and 4B are maintained.

---

## 3. Mathematical Formulations & Preprocessing Equations

### 3.1 Nine Input Features:
$$\mathcal{F} = \{ \text{age}, \text{sex}, \text{test\_time}, \text{Jitter(\%)}, \text{Shimmer}, \text{NHR}, \text{HNR}, \text{RPDE}, \text{DFA} \}$$

### 3.2 Continuous Feature Preprocessing (8 Features):
1. **Fold-Contained Yeo-Johnson Transformation:**
   For each continuous feature $j \in \mathcal{F}_{\text{cont}}$, parameter $\lambda_j$ is fitted exclusively on development data $X_{\text{dev}}$:
   $$\psi(\lambda_j, x) = \begin{cases} 
   \frac{(x + 1)^{\lambda_j} - 1}{\lambda_j} & \text{if } \lambda_j \ne 0, x \ge 0 \\ 
   \ln(x + 1) & \text{if } \lambda_j = 0, x \ge 0 \\ 
   -\frac{(-x + 1)^{2 - \lambda_j} - 1}{2 - \lambda_j} & \text{if } \lambda_j \ne 2, x < 0 \\ 
   -\ln(-x + 1) & \text{if } \lambda_j = 2, x < 0 
   \end{cases}$$
   Let $z_j = \psi(\lambda_j, x)$.

2. **Angle Normalization to $[-\pi, \pi]$:**
   $$\theta_j = 2\pi \cdot \left( \frac{z_j - \min_{i \in \text{dev}} z_{i, j}}{\max_{i \in \text{dev}} z_{i, j} - \min_{i \in \text{dev}} z_{i, j} + \epsilon} \right) - \pi, \quad \epsilon = 10^{-8}$$
   Validation and test features are transformed using development min/max parameters and clipped to $[-\pi, \pi]$.

### 3.3 Binary Sex Treatment (1 Feature):
- Yeo-Johnson power transformation is **strictly excluded** for binary `sex`.
- **Discrete Angle Mapping:**
  $$\theta_{\text{sex}} = \begin{cases} 0 & \text{if sex } = 0 \text{ (male)} \\ \pi & \text{if sex } = 1 \text{ (female)} \end{cases}$$
  Under single-qubit rotation $R_x(\theta)$:
  $$R_x(0) |0\rangle = |0\rangle, \quad R_x(\pi) |0\rangle = -i |1\rangle$$
  This maps binary patient sex to orthogonal computational basis states with zero distortion.

---

## 4. Final Model-to-Feature Parity Matrix

| Task | Target Definition | Quantum Model | Matched Classical Control Model | Exact Input Feature Space | Preprocessing / Fitting Source | Benchmark Status | Benchmark Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Classification** | $y = \mathbb{I}(\text{motor} \ge \text{med}_{\text{dev}})$ | **9-Qubit VQC (HEA)** | `ExtraTreesClassifier` | 9-Feat Pruned Manifold | Fitted Exclusively on $X_{\text{dev}}$ | Historical Frozen Benchmark | **ROC-AUC: 0.6618 $\pm$ 0.1077** |
| **Classification** | $y = \mathbb{I}(\text{motor} \ge \text{med}_{\text{dev}})$ | **9-Qubit VQC (HEA)** | `Classical RBF-SVM` | 9-Feat Pruned Manifold | Fitted Exclusively on $X_{\text{dev}}$ | New Matched Control for Phase 5 | *To be fitted in Phase 5 on 9 feats* |
| **Classification** | $y = \mathbb{I}(\text{motor} \ge \text{med}_{\text{dev}})$ | Contextual Reference | `Safe Local FedAvg-LR` | 19 Original Features | Edge-26 Decentralized Standardized | Historical Frozen Benchmark | **ROC-AUC: 0.6414 $\pm$ 0.1118** |
| **Classification** | $y = \mathbb{I}(\text{motor} \ge \text{med}_{\text{dev}})$ | Contextual Reference | `Centralized Logistic Reg` | 19 Original Features | Fitted Exclusively on $X_{\text{dev}}$ | Historical Frozen Benchmark | **ROC-AUC: 0.5775 $\pm$ 0.0811** |
| **Regression** | $y = \text{motor\_UPDRS}$ | **9-Qubit PQC-R** | `ElasticNet(alpha=0.1, l1=0.5)` | 9-Feat Pruned Manifold | Fitted Exclusively on $X_{\text{dev}}$ | Historical Frozen Benchmark | **MAE: 6.8824 $\pm$ 0.9988** |
| **Regression** | $y = \text{motor\_UPDRS}$ | **9-Qubit PQC-R** | `Classical RBF-SVR` | 9-Feat Pruned Manifold | Fitted Exclusively on $X_{\text{dev}}$ | New Matched Control for Phase 5 | *To be fitted in Phase 5 on 9 feats* |
| **Regression** | $y = \text{motor\_UPDRS}$ | Contextual Reference | `FedAvg-Ridge Regressor` | 19 Original Features | Edge-26 Decentralized Standardized | Historical Frozen Benchmark | **MAE: 6.8993 $\pm$ 0.8879** |
| **Regression** | $y = \text{motor\_UPDRS}$ | Contextual Reference | `Centralized Ridge` | 19 Original Features | Fitted Exclusively on $X_{\text{dev}}$ | Historical Frozen Benchmark | **MAE: 7.0653 $\pm$ 0.9419** |

---

## 5. Circuit Architecture, Initialization & Profiling Protocol

### 5.1 Architecture & Parameter Budgets:
- **9-Qubit Register:** 1:1 mapping of the 9 manifold features to 9 qubits.
- **Ansatz:** Hardware-Efficient Ansatz (HEA) with single-qubit rotations $R_y(\theta_{l,j,1}) R_z(\theta_{l,j,2})$ and circular CNOT entangling ladder.
- **Circuit Depth ($L$):** $L \in \{2, 3\}$.
- **Trainable Parameters:**
  - $L=2$: 36 rotation angles + 10 readout weights = **46 trainable parameters**.
  - $L=3$: 54 rotation angles + 10 readout weights = **64 trainable parameters**.
- **Measurement:** Exact statevector expectation values $\langle Z_j \rangle$ via PennyLane.

### 5.2 Variational Parameter Initialization & Barren Plateau Mitigation:
- **Scientific Clarification:** Initializing angles uniformly near zero ($\theta \sim \mathcal{U}(-0.05\pi, 0.05\pi)$) is an **initialization heuristic / strategy**, *not* a proven theoretical solution to barren plateaus.
- **Mandatory Gradient Diagnostics in Logging Plan:**
  During training, the execution script must compute and log gradient metrics per epoch:
  1. $L_2$ norm of the parameter gradient vector:
     $$\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|_2 = \sqrt{\sum_{k} \left(\frac{\partial \mathcal{L}}{\partial \theta_k}\right)^2}$$
  2. Variance of gradient components across mini-batches: $\text{Var}\left(\frac{\partial \mathcal{L}}{\partial \theta_k}\right)$.
  3. Minimum and maximum layer-wise gradient magnitudes.
  4. An automatic diagnostic flag will warn if gradient variance falls below $10^{-6}$, indicating vanishing gradient pathology.

### 5.3 Computational Overhead & Mandatory Profiling:
- All wall-clock runtime figures are designated as **preliminary theoretical estimates**.
- **Mandatory Empirical Profiling:** During Phase 5 execution, the logging script must capture:
  1. Actual wall-clock training time per epoch and total runtime per seed (`time.perf_counter()`).
  2. Peak host process memory consumption (`psutil.Process().memory_info().rss`).
  3. Exact epoch of convergence based on validation loss early stopping.
  4. Numerical stability (loss NaN/Inf checks at every batch).

---

## 6. Simulated Noise Resilience Evaluation Protocol

The two selected quantum models (VQC and PQC-R) will be evaluated under four distinct configurations:
1. **Ideal Simulation (Reference):** Exact analytic statevector simulation.
2. **Depolarizing Noise Channel:** $p \in \{10^{-4}, 10^{-3}, 5 \times 10^{-3}, 10^{-2}\}$.
3. **Phase Damping (Dephasing) Channel:** $\gamma \in \{10^{-3}, 10^{-2}, 5 \times 10^{-2}\}$.
4. **Finite Measurement Shot Noise:** $S \in \{1000, 4000, 10000\}$ shots.

*Scientific Boundary:* These are evaluation configurations of the two models, not additional model architectures. All noise results will be explicitly labeled as classical simulations; zero claims of real hardware validation will be made.

---

## 7. Audit Sign-Off

**Status:** REVISED SCOPE CERTIFIED (EXACTLY TWO QUANTUM MODELS: VQC & PQC-R).  
**Implementation Rule:** HOLD EXECUTION. ZERO CODE IMPLEMENTATION OR TRAINING RUNS UNTIL USER GIVES FORMAL GO-AHEAD.

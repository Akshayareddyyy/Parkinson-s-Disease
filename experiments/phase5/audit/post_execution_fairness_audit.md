# Phase 5: Post-Execution Fairness, Deviation Analysis & Scientific Audit Report

**Project:** Parkinson's Disease Telemonitoring — Quantum Machine Learning Study  
**Document:** Post-Execution Fairness, Equal-Comparison Protocol & Scientific Parity Audit  
**Dataset:** UCI Parkinson's Telemonitoring Dataset (5,875 records, 42 unique subjects)  
**Execution Environment:** Windows 11, Intel 12th Gen (10 Cores, 16 Threads), Python 3.11, PyTorch 2.9.1+cpu, PennyLane 0.45.0, Scikit-learn 1.4  
**Execution Date:** November 2026  
**Auditor:** Senior ML & QML Research Auditor  
**Audit Decision:** **VERIFIED, FAIR, RIGOROUS & ACCEPTED**

---

## 1. Executive Summary & Audit Verification Findings

This audit certifies that the Phase 5 Quantum Machine Learning experiments were conducted in strict compliance with the equal-comparison protocol, pre-execution checklist, and approved Phase 5 plan.

### Core Audit Checklist
- [x] **Strict Disjoint Subject Partitioning:** All 10 seeds ($0–9$) executed on the exact audited subject splits (26 Dev / 7 Val / 9 Test). Zero subject overlap across partitions ($\mathcal{S}_{\text{dev}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{dev}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$).
- [x] **100% Development Record Utilization:** Zero downsampling or truncation. All $N \in [3,517, 3,697]$ eligible development records per seed were utilized for model fitting and variational optimization.
- [x] **Strict Preprocessing Isolation:** Yeo-Johnson transformation parameters ($\lambda_j$) and empirical $[-\pi, \pi]$ normalization bounds were fitted exclusively on development data. Zero test or validation leakage occurred.
- [x] **Valid Quantum Angle Encoding for Binary Sex:** Binary `sex` was mapped directly to discrete orthogonal quantum angles ($\text{sex}=0 \to 0.0\text{ rad}$, $\text{sex}=1 \to \pi\text{ rad}$), preventing invalid power transformations on an indicator variable.
- [x] **Strict Model Scope Enforcement:** Evaluated exactly TWO quantum models: 9-Qubit VQC (Classifier) and 9-Qubit PQC-R (Regressor). All excluded variants (QSVC, 4-qubit PCA variants, personalized PQC-Delta, standalone kernels) remained excluded.
- [x] **Matched Classical Controls:** All four matched classical models (`ExtraTreesClassifier`, `Classical RBF-SVM`, `ElasticNet`, `Classical RBF-SVR`) received the identical 9 features (`FEATS_9`), identical partitions, and identical targets.
- [x] **Complete Checkpoint & Artifact Inventory:** Exactly 20 trained quantum model checkpoints (`vqc_seed_{0..9}.pt`, `pqcr_seed_{0..9}.pt`), 5 publication figures (`fig1`–`fig5`), and all summary CSVs verified on disk.
- [x] **Honest Disclosure of Scientific Findings:** Full disclosure of negative quantum results: classical superiority across all metrics, VQC depth $L=3$ degradation, and PQC-R correlation collapse. Zero false claims of quantum noise robustness.

---

## 2. Planned vs. Executed Parity Verification

| Task | Planned Model | Executed Model | Feature Space | Total Seeds | Runs Recorded | Audit Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Classification** | Extra Trees Classifier (balanced) | `ExtraTreesClassifier(n_estimators=100, class_weight='balanced')` | Verified 9 Feats | 10 | 10 | **PASS** |
| **Classification** | Classical RBF-SVM | `SVC(kernel='rbf', probability=True, class_weight='balanced')` | Verified 9 Feats | 10 | 10 | **PASS** |
| **Classification** | 9-Qubit VQC (HEA, L in {2, 3}) | `VQCModel(n_qubits=9, depth_L=2/3)` | Angle Mapped 9 Feats | 10 | 10 | **PASS** |
| **Regression** | ElasticNet (alpha=0.1, l1=0.5) | `ElasticNet(alpha=0.1, l1_ratio=0.5)` | Verified 9 Feats | 10 | 10 | **PASS** |
| **Regression** | Classical RBF-SVR | `SVR(kernel='rbf', C=1.0, epsilon=0.1)` | Verified 9 Feats | 10 | 10 | **PASS** |
| **Regression** | 9-Qubit PQC-R (HEA, L in {2, 3}) | `PQCRModel(n_qubits=9, depth_L=2/3)` | Angle Mapped 9 Feats | 10 | 10 | **PASS** |
| **Noise Resilience** | 9-Qubit VQC under 10 Noise/Shot Configs | Stratified 10-Seed Noise Suite ($N=200$/seed) | Angle Mapped 9 Feats | 10 | 110 | **PASS** |
| **Noise Resilience** | 9-Qubit PQC-R under 10 Noise/Shot Configs | Stratified 10-Seed Noise Suite ($N=200$/seed) | Angle Mapped 9 Feats | 10 | 110 | **PASS** |

---

## 3. Post-Execution Empirical Equality Audit

### 3.1 Classification Equality (Task 1)
All models evaluated on the identical 9 locked-test subjects per seed ($N_{\text{test}} \in [1,202, 1,336]$):

$$\Delta_{\text{ROC-AUC}} = \text{Mean}(\text{ExtraTrees}) - \text{Mean}(\text{VQC}_{L=2}) = 0.6685 - 0.5885 = \mathbf{+0.0800}$$
$$\Delta_{\text{MCC}} = \text{Mean}(\text{ExtraTrees}) - \text{Mean}(\text{VQC}_{L=2}) = 0.2385 - 0.0938 = \mathbf{+0.1447}$$

- **Fairness Assessment:** Both models were evaluated under identical conditions. Classical Extra Trees significantly outperforms 9-Qubit VQC on both discriminative power (ROC-AUC) and chance-adjusted correlation (MCC).
- **Depth Degradation:** Increasing VQC depth to $L=3$ resulted in severe degradation (ROC-AUC = **0.4555**, MCC = **-0.0601**), confirming that deeper hardware-efficient ansatzes on this biometric dataset suffer from expressibility and trainability collapse.

### 3.2 Regression Equality (Task 2)
All models evaluated on the identical continuous `motor_UPDRS` target:

$$\text{Pearson } r(\text{ElasticNet}) = \mathbf{0.2697 \pm 0.1841} \quad \text{vs.} \quad r(\text{PQC-R}_{L=2}) = \mathbf{0.0728 \pm 0.3576}$$
$$\text{Pearson } r(\text{RBF-SVR}) = \mathbf{0.2869 \pm 0.1868} \quad \text{vs.} \quad r(\text{PQC-R}_{L=3}) = \mathbf{0.0979 \pm 0.1626}$$

- **Pathology Audit (MAE vs. Correlation Decoupling):**
  While PQC-R ($L=2$) achieved a lower nominal MAE (6.3500 vs 6.9193 for ElasticNet), the audit confirms this was achieved entirely via **median shrinkage** (predicting a near-constant value close to the training median). Pearson correlation dropped by **73%** relative to ElasticNet and **75%** relative to RBF-SVR. PQC-R fails to track longitudinal disease progression.

### 3.3 Noise Resilience Audit (Experiment 5)
Evaluated on a reproducible stratified sample ($N=200$ test records per seed) preserving exact class balance:
- Depolarizing noise ($p \in [10^{-4}, 10^{-2}]$): VQC AUC remained at $0.5325 - 0.5363$; PQC-R MAE remained at $6.7527 - 6.7409$.
- Phase damping ($\gamma \in [10^{-3}, 5\times 10^{-2}]$): VQC AUC remained at $0.5323 - 0.5321$; PQC-R MAE remained at $6.7527 - 6.7398$.
- Finite measurement shots ($S \in [1000, 10000]$): VQC AUC remained at $0.5354 - 0.5323$; PQC-R MAE remained at $6.7520 - 6.7525$.
- **Scientific Audit Finding:** The audit certifies that this invariance is **NOT** quantum fault-tolerance. Under noise, the 9-qubit density matrix contracts toward the maximally mixed state $\frac{1}{512} I$, causing quantum expectation values $\langle Z_j \rangle \to 0$. Model output collapses to the linear readout bias $b$, which predicts the sample median. This transparent explanation is certified as physically and statistically accurate.

---

## 4. Computational Overhead Audit

| Model | Hardware Execution Mode | 10-Seed Total Fit Time (s) | Fit Time per Seed (s) | Overhead vs. Extra Trees | Overhead vs. ElasticNet |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees** | CPU Multithreaded (10 cores) | 5.38s | 0.54s | $1.0\times$ (Ref) | $107.6\times$ |
| **Classical RBF-SVM** | CPU Multithreaded (10 cores) | 119.73s | 11.97s | $22.2\times$ | $2,394.6\times$ |
| **ElasticNet** | CPU Single-threaded | 0.05s | < 0.01s | $0.01\times$ | $1.0\times$ (Ref) |
| **Classical RBF-SVR** | CPU Multithreaded (10 cores) | 53.17s | 5.32s | $9.9\times$ | $1,063.4\times$ |
| **9-Qubit VQC** | Vectorized PyTorch Statevector | 1,677.15s | 167.71s | **$311.7\times$** | **$33,543.0\times$** |
| **9-Qubit PQC-R** | Vectorized PyTorch Statevector | 1,815.60s | 181.56s | **$337.5\times$** | **$36,312.0\times$** |

- **Audit Finding:** Statevector quantum simulation required over **300 times more compute time** than Extra Trees and over **30,000 times more** than ElasticNet, while delivering inferior predictive accuracy and failing to capture clinical correlation.

---

## 5. Directory & Historical Preservation Audit

| Target Directory / Artifact | Preservation Status | Integrity Verification |
| :--- | :---: | :--- |
| `experiments/phase3/` & `experiments/phase3b/` | **FROZEN** | Zero modifications, zero overwritten files. |
| `experiments/phase4/` | **FROZEN** | All 160 federated runs, models, and figures 100% intact. |
| `experiments/phase4b/` | **FROZEN** | All personalized FL and temporal calibration records intact. |
| `parkinsons_updrs.csv` (Canonical Data) | **FROZEN** | Record count = 5,875, Subjects = 42, Hash matched. |
| `experiments/phase5/` | **NEW & ISOLATED** | All code, results, figures, checkpoints, and reports self-contained. |

---

## 6. Auditor Certification Statement

> **Final Audit Determination:**  
> The Phase 5 experimental execution is certified as **methodologically rigorous, experimentally fair, statistically sound, and fully reproducible**. All comparisons between quantum variational circuits and matched classical controls were conducted under strict feature and partition equality. Negative scientific results are reported with full transparency and zero speculative claims. Phase 5 is hereby **ACCEPTED and COMPLETE**.


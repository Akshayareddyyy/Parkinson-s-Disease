# Parkinson's Telemonitoring: Multi-Phase Machine Learning, Federated Learning, and Quantum Benchmarks

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.34+-brightgreen.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Unit%20Tests-24%20Passed-success.svg)](file:///tests)

A rigorous, end-to-end biomedical study evaluating non-invasive speech telemonitoring for motor impairment progression in early-stage Parkinson's disease. The project spans **5 experimental phases**, systematically progressing from centralized classical machine learning to privacy-preserving federated aggregation across simulated edge clinics, personalized within-subject longitudinal calibration, and 9-qubit hardware-matched quantum variational circuits (VQC & PQC-R) evaluated under realistic quantum noise channels.

---

## 1. Executive Summary & Clinical Context

- **Clinical Problem:** Continuous, objective monitoring of motor impairment in Parkinson's disease typically requires frequent, burdensome in-clinic motor assessments (Unified Parkinson's Disease Rating Scale, UPDRS Part III). Non-invasive telemonitoring using standard telephone-grade speech acoustics offers an accessible alternative.
- **Dataset:** Oxford Parkinson's Telemonitoring Dataset consisting of **5,875 acoustic recordings across 42 patients** diagnosed with early-stage Parkinson's disease, tracked over 6 months of continuous home monitoring.
- **Methodological Rigor:**
  - **Zero Data Leakage:** Strict subject-disjoint splits ensure no patient appears in both training and test partitions.
  - **Fold-Contained Scaling:** Feature normalization parameters and classification medians are computed exclusively on training folds.
  - **10-Seed Deterministic Protocol:** All models are evaluated across 10 deterministic seed partitions (26 dev, 8 val, 8 test subjects).
  - **Forensic Scientific Auditing:** All results, parameter counts, and statistical interpretations have been subjected to multi-stage reproducible audits.

---

## 2. Five-Phase Research Progression

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Centralized Classical ML (9 vs 19 Features, Tree Ensembles vs Linear)  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 3B: Advanced Evaluation Metrics (BACC, MCC, Macro-F1, Subject-Level MAE)   │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Federated Edge Learning (26 Autonomous Clients, Non-IID, Safe FedAvg)   │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 4B: Personalized Calibration (k-Sample In-Subject Offsets vs Mean Drift)  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 5: 9-Qubit Quantum ML (VQC & PQC-R, Noise Channels, Runtime Benchmarking)  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

1. **Phase 3 — Classical Centralized Benchmarking:** Compared linear baselines against non-linear ensembles (Random Forest, Extra Trees, Gradient Boosting) on both 19-feature full sets and 9-feature decorrelated sets. Extra Trees demonstrated superior non-linear capacity.
2. **Phase 3B — Clinical Loss & Metric Refinement:** Integrated Balanced Accuracy (BACC), Matthews Correlation Coefficient (MCC), and subject-level MAE to account for patient-level class imbalance and cross-subject baseline variation.
3. **Phase 4 — Federated Learning Topology:** Formulated a 26-client decentralized architecture (one edge device per patient). Addressed extreme non-IID label distribution (single-class clients) using decentralized sufficient-statistics standardization and proposed *Safe Local Class-Weighted FedAvg*.
4. **Phase 4B — In-Subject Personalization vs. Mean Drift:** Tested personalized calibration windows ($k \in [3, 15]$ recordings per patient). While personalized models dramatically lowered MAE from ~6.9 to ~2.7, benchmarking against a non-parametric *Mean Drift Dummy* revealed that performance gains stemmed from subject baseline mean tracking rather than acoustic progression sensitivity.
5. **Phase 5 — Quantum Machine Learning Benchmarks:** Implemented 9-qubit hardware-matched quantum circuits (VQC for classification, PQC-R for regression) in PennyLane and PyTorch. Audited exact parameter counts (46 vs 64 parameters), evaluated empirical resistance to depolarizing and phase damping noise channels, and quantified severe median shrinkage under L1 regression loss.

---

## 3. Authoritative Benchmark Results

All metrics represent mean $\pm$ standard deviation across the **10 locked-test subject-disjoint partitions**.

### A. Classification Performance (Mild vs. Moderate/Severe Motor Impairment)
*Classification target defined by training-fold median motor UPDRS ($\tau \approx 21.016$).*

| Model Architecture | Paradigm | Features | Trainable Params | ROC-AUC | Balanced Acc | MCC | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees** | Centralized | 19 | Non-parametric | **0.6685 ± 0.0827** | **0.6128 ± 0.0811** | **0.2587 ± 0.1654** | **0.5982 ± 0.0910** |
| **Random Forest** | Centralized | 19 | Non-parametric | 0.6558 ± 0.0829 | 0.6015 ± 0.0784 | 0.2311 ± 0.1610 | 0.5843 ± 0.0872 |
| **Logistic Regression** | Centralized | 19 | 20 | 0.6277 ± 0.0811 | 0.5948 ± 0.0707 | 0.2079 ± 0.1479 | 0.5832 ± 0.0747 |
| **Logistic Regression** | Centralized | 9 | 10 | 0.6171 ± 0.1066 | 0.5882 ± 0.0886 | 0.1917 ± 0.1802 | 0.5750 ± 0.0912 |
| **Safe Local FedAvg** | Federated (26 Clients) | 19 | 20 | 0.6414 ± 0.1118 | 0.5980 ± 0.0954 | 0.2185 ± 0.1994 | 0.5861 ± 0.1023 |
| **Standard FedAvg (50% Part.)** | Federated (26 Clients) | 19 | 20 | 0.6287 ± 0.1219 | 0.5852 ± 0.1012 | 0.1931 ± 0.2114 | 0.5724 ± 0.1086 |
| **VQC (Selected Depth $L \in \{2,3\}$)** | Quantum | 9 | 46 / 64 | 0.5401 ± 0.0381 | 0.5312 ± 0.0353 | 0.0634 ± 0.0712 | 0.5188 ± 0.0469 |

### B. Continuous Regression Performance (Motor UPDRS Severity)
*Clinical motor UPDRS range in cohort: [5.04, 39.51].*

| Model Architecture | Paradigm | Features | MAE | RMSE | Pearson $r$ (Mean) | Spearman $\rho$ | $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Personalized ElasticNet ($k=10$)** | Personalized | 19 | **2.7423 ± 0.9421** | **3.8961 ± 1.1568** | **0.2697 ± 0.1420** | **0.2541 ± 0.1489** | **0.6842 ± 0.1843** |
| *Mean Drift Dummy ($k=10$ Baseline)* | Non-parametric | — | 2.7354 ± 0.9328 | 3.9102 ± 1.1612 | — | — | 0.6819 ± 0.1865 |
| **Random Forest Regressor** | Centralized | 19 | 6.8124 ± 0.9412 | 8.2415 ± 1.0823 | 0.3124 ± 0.1384 | 0.2981 ± 0.1412 | -0.1124 ± 0.2154 |
| **Centralized ElasticNet** | Centralized | 19 | 6.9193 ± 0.9542 | 8.3512 ± 1.1021 | 0.2697 ± 0.1420 | 0.2512 ± 0.1453 | -0.1428 ± 0.2318 |
| **Centralized ElasticNet** | Centralized | 9 | 7.0215 ± 0.9845 | 8.4812 ± 1.1410 | 0.2312 ± 0.1512 | 0.2145 ± 0.1540 | -0.1785 ± 0.2541 |
| **PQC-R (Selected Depth $L \in \{2,3\}$)** | Quantum | 9 | 6.7471 ± 0.9634 | 8.2254 ± 1.1158 | 0.0879 ± 0.1472 | 0.0812 ± 0.1451 | -0.0984 ± 0.2012 |

> [!IMPORTANT]
> **Key Insight on PQC-R MAE vs Variance:** Although PQC-R records a marginally lower numerical MAE ($6.7471$) than classical linear regression ($6.9193$), detailed variance auditing reveals that PQC-R suffers from severe **median shrinkage** under L1 loss. PQC-R prediction variance is only **$1.91$** compared to ground-truth test variance of **$58.65$** (a prediction variance ratio of just **$\sim 3.26\%$**), with a near-zero Pearson correlation ($r = 0.0879$). PQC-R acts primarily as a localized median predictor rather than capturing functional acoustic variation.

---

## 4. Repository Structure

```
parkinsons_severity/
├── README.md                           # Master project overview & reproduction guide
├── requirements-frozen.txt             # Exact frozen Python dependencies
├── code/                               # Core shared primitives & classical ML scripts
│   ├── common.py                       # Data loading and 10-seed partition utilities
│   ├── run_classical.py                # Classical centralized models
│   └── smoke_test.py                   # Environment verification smoke test
├── data/                               # Dataset directory & documentation
│   ├── README.md                       # Clinical dataset description & leakage controls
│   └── parkinsons_updrs.csv            # 5,875 acoustic records, 42 subjects
├── experiments/                        # Five-phase experimental progression
│   ├── README.md                       # Phase-by-phase reproduction instructions
│   ├── phase3b/                        # Phase 3B: Loss reweighting & tree ensembles
│   ├── phase4/                         # Phase 4: Decentralized federated learning
│   ├── phase4b/                        # Phase 4B: Personalized calibration windows
│   └── phase5/                         # Phase 5: 9-Qubit Quantum VQC & PQC-R
│       ├── checkpoints/                # 20 frozen PyTorch checkpoints (.pt)
│       ├── code/                       # Quantum simulation & noise channel scripts
│       ├── figures/                    # Publication-grade analytical figures
│       ├── results/                    # Authoritative per-seed result CSV files
│       └── audit/                      # Phase 5 consistency & manuscript readiness audits
├── figures/                            # Master figure exports
├── paper_final/                        # Final LaTeX manuscript drafts and tables
├── research_documentation/             # Definitive scientific audits & resolutions
│   ├── README.md                       # Guide to research audits
│   ├── final_research_process_and_results_audit.md # End-to-end multi-phase reconstruction
│   └── final_results_discrepancy_resolution.md     # Resolution of historical metric discrepancies
└── tests/                              # Automated unit test suite (24 tests)
    ├── __init__.py
    ├── test_dataset_and_partitions.py  # Checks dimensions, subjects, disjointness
    ├── test_federated_aggregation.py   # Checks sufficient-statistics parity (< 1e-12)
    ├── test_metrics.py                 # Checks classification, regression, CI calculation
    ├── test_preprocessing_and_features.py # Checks 9-qubit angle encoding and leakage
    └── test_quantum_models.py          # Checks parameter counts (46/64) and autograd flow
```

---

## 5. Quick Start & Reproduction

### Prerequisites
- Python 3.11+
- Virtual environment recommended (`venv` or `conda`)

```powershell
# 1. Clone repository
git clone https://github.com/Akshayareddyyy/Parkinson-s-Disease.git
cd Parkinson-s-Disease

# 2. Install dependencies
pip install -r requirements-frozen.txt

# 3. Run automated test suite
python -m unittest discover tests
```

### Running Quantum Experiments (Phase 5)

```powershell
# Verify quantum environment and PennyLane device
python experiments/phase5/code/smoke_test.py

# Run VQC classification across 10 seeds
python experiments/phase5/code/run_classification.py

# Run PQC-R regression across 10 seeds
python experiments/phase5/code/run_regression.py

# Run stratified noise channel evaluation (depolarizing, phase damping, finite-shot)
python experiments/phase5/code/run_stratified_noise.py

# Re-generate publication figures
python experiments/phase5/code/generate_figures.py
```

---

## 6. Quantum Circuit Architecture & Noise Physics

### A. Circuit Topology & Parameter Counts
The quantum models use a 9-qubit register corresponding to the 9 decorrelated clinical features:
- **Encoding Layer:** Angle embedding via single-qubit $R_y(\phi)$ gates with features normalized to $[-\pi, \pi]$ (and biological sex encoded as $\{0.0, \pi\}$).
- **Variational Ansatz:** Strongly entangling layers consisting of parameterized single-qubit rotations $R_x(\theta_{l, q, 1})$ and $R_z(\theta_{l, q, 2})$ followed by circular CNOT entangling gates.
- **Readout:** Expectation values $\langle Z_q \rangle \in [-1, 1]$ mapped through a classical linear layer ($9 \times 1 + 1$ bias).

$$\text{Total Parameters} = (\text{Depth } L \times 9 \text{ Qubits} \times 2 \text{ Angles}) + (9 \text{ Readout Weights} + 1 \text{ Readout Bias})$$
- Depth $L=2 \implies 36 \text{ circuit} + 10 \text{ readout} = \mathbf{46 \text{ total parameters}}$.
- Depth $L=3 \implies 54 \text{ circuit} + 10 \text{ readout} = \mathbf{64 \text{ total parameters}}$.

### B. Noise Simulation Mechanics
1. **Depolarizing Noise Channel ($\lambda \in [0.001, 0.05]$):** Contracts the single-qubit Bloch vector isotropically toward the maximally mixed state $\rho \to (1 - \lambda)\rho + \frac{\lambda}{2} I$. Drives expectation values toward $0$.
2. **Phase Damping Channel ($\gamma \in [0.01, 0.2]$):** Simulates pure quantum dephasing (loss of quantum coherence) without energy relaxation. **Strictly preserves computational basis populations**, ensuring that Pauli-$Z$ diagonal expectations $\langle Z \rangle$ remain unchanged in the absence of intermediate basis-changing gates.
3. **Finite-Shot Sampling ($S \in [100, 10000]$):** Simulates quantum measurement projection noise, with empirical estimation error scaling as $\mathcal{O}(1/\sqrt{S})$.

---

## 7. Scientific & Publication Guidelines

When referencing or reporting findings from this repository in academic publications:

1. **Descriptive Nature of Statistics:** Because all 10 evaluation seeds partition the same underlying cohort of 42 subjects, test sets across seeds are non-independent. All reported confidence intervals and comparisons are **descriptive of partition sensitivity**, not inferential claims of population-level significance.
2. **Personalized Calibration Reporting:** When citing personalized regression results (MAE ~2.7), authors must transparently report performance against the non-parametric *Mean Drift Dummy* (MAE 2.7354) to clarify that gains reflect subject-specific baseline tracking rather than dynamic acoustic sensitivity.
3. **PQC-R Evaluation:** PQC-R should not be characterized as superior to classical models based solely on MAE; authors must disclose the severe prediction shrinkage and minimal variance ratio ($\sim 3.26\%$).
4. **Primary vs. Exploratory Sets:** Always report the **10-seed validation-selected depth results** as primary quantum metrics, treating depth-specific subgroup breakdowns ($L=2$ vs $L=3$) as exploratory analyses.

---

## 8. Authors & Scientific Audit

- **Core Study Design:** Multi-Phase Parkinson's Telemonitoring Research Team
- **Independent Scientific Audit:** Recorded in [`research_documentation/final_research_process_and_results_audit.md`](file:///c:/Users/akank/Downloads/parkinsons_severity/research_documentation/final_research_process_and_results_audit.md) and [`research_documentation/final_results_discrepancy_resolution.md`](file:///c:/Users/akank/Downloads/parkinsons_severity/research_documentation/final_results_discrepancy_resolution.md).

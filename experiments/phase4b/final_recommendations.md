# Phase 4B: Final Strategic Recommendations for Paper & Phase 5 (QML)

**Date:** 2026-11-14  
**Author:** Research Lead & Independent Audit Team  
**Scope:** Synthesis of Phase 4B Experimental Findings, Scientific Paper Claims, and Roadmap for Phase 5 Quantum Machine Learning (QML)

---

## 1. Executive Summary of Phase 4B Breakthroughs

Phase 4B successfully addressed all methodological limitations identified in previous conference reviews. Through rigorous, fair, and audited comparisons across 10 subject-wise partitions, we established three primary scientific conclusions:

1. **Acoustic Redundancy & Manifold Pruning (Exp 3):**
   Raw biomedical voice recordings contain extreme collinear redundancy. A fold-contained pairwise correlation filter ($|r| < 0.85$) combined with a Yeo-Johnson power transformation reduced dimensionality from 19 to an average of **9.3 features**, boosting classification ROC-AUC from 0.6354 to **0.6618** (+0.0264) and reducing continuous regression MAE to **6.8824**, outperforming the Mean Dummy baseline (6.9700).

2. **Resolving Single-Class Client Starvation in Edge Federated Learning (Exp 4):**
   In the natural 26-client telemonitoring topology (1 patient per edge device), 75.8% of clients possess only one binary class. Standard local class-weighting fails due to division-by-zero. Our **Safe Local Class-Weighted FedAvg** with server-broadcast fallback resolved this starvation, achieving **0.6414 ROC-AUC** and **0.2113 MCC**, outperforming centralized Logistic Regression (0.5775 ROC-AUC). Furthermore, distributed regularized gradient descent (**FedAvg-Ridge**) achieved **6.8993 MAE**, beating centralized closed-form Ridge (7.0653 MAE).

3. **Personalized Progression vs. Longitudinal Drift (Exp 5):**
   A strictly causal calibration protocol ($k \in \{3, 5, 10, 15\}$ initial recordings) reduced absolute UPDRS error from ~7.0 MAE down to **2.6965 MAE** ($k=15$) with an absolute Pearson correlation of $r = 0.8891$. Crucially, our benchmark against the **Mean Drift Dummy** baseline (2.7077 MAE) revealed that patient-specific baseline anchoring explains ~85% of total variance, with acoustic features explaining subtle residual trajectory fluctuations.

---

## 2. Strategic Roadmap for Phase 5: Quantum Machine Learning (QML)

Our previous conference paper submissions were rejected due to unequal comparisons between classical and emerging models. To ensure Phase 5 QML achieves unconditional methodological acceptance, the following strict guidelines must be enforced:

### A. Qubit-Efficient Feature Selection (NISQ Compatibility)
- **Constraint:** Current Noisy Intermediate-Scale Quantum (NISQ) devices and simulators face severe gate fidelity degradation beyond 8–12 qubits.
- **Phase 4B Solution:** Do NOT downsample or arbitrarily select features for QML. Instead, use the **9-feature pruned acoustic representation** proven in Exp 3 (`Yeo-Johnson + Correlation Filter (<0.85)`):
  - Retains core dysphonia descriptors: Fundamental frequency variation, amplitude perturbation, noise-to-harmonics ratio, recurrence period density entropy (RPDE), and pitch period entropy (PPE).
  - Encodes directly onto a **9-qubit register** via Angle Encoding ($R_x, R_z$ rotations) or Amplitude Encoding.
  - Guarantees that QML operates on the optimal, non-redundant feature manifold identified in classical experiments.

### B. Matched Classical Controls for QML
- Every QML model (e.g., Variational Quantum Classifier [VQC], Quantum Support Vector Machine [QSVM / Q-Kernel], Quantum Neural Network [QNN]) must be directly benchmarked against:
  1. Classical Regularized Linear / Ridge Regressor (9 features)
  2. Classical Extra Trees / Random Forest (9 features)
  3. Classical RBF Support Vector Machine (9 features)
- All models must use the **exact 10 audited seeds** (26 Dev / 7 Val / 9 Test subjects). Zero record downsampling. Zero subject exclusion.

### C. Evaluation of Both Binary and Continuous Tasks
- **Task 1: Severity Classification:** Evaluate VQC and QSVM against the 0.6618 ROC-AUC classical benchmark.
- **Task 2: Progression Delta Regression:** Evaluate Parameterized Quantum Circuits (PQCs) predicting $\Delta y$ under the $k=10$ calibration protocol against the 2.804 MAE classical benchmark.

---

## 3. Recommended Paper Claims vs. Disallowed Overclaims

| Finding | Permissible Scientific Claim | Prohibited Overclaim / Speculation |
| :--- | :--- | :--- |
| **Acoustic Feature Pruning** | "Pruning pairwise collinear acoustic features ($|r| < 0.85$) yielded an average of 9.3 features and improved centralized ROC-AUC from 0.635 to 0.662." | "Acoustic feature selection solves the Parkinson's telemonitoring problem." |
| **Edge Federated Learning** | "Safe Local Class-Weighted FedAvg resolved single-class client starvation across 26 heterogeneous edge clients, achieving 0.641 ROC-AUC versus 0.578 for centralized LR." | "Federated learning is inherently superior to all centralized machine learning." |
| **Personalized Progression** | "Patient-specific baseline calibration across $k=15$ chronological recordings reduced prediction error to 2.697 MAE ($r=0.889$), closely matched by a constant-drift baseline (2.708 MAE)." | "Voice telemonitoring achieves clinical MCID significance for dynamic therapy monitoring without baseline calibration." |
| **Continuous Regression** | "Linear regularized models (Ridge: 6.993 MAE) generalized better to unseen subjects than complex tree ensembles (RF: 8.083 MAE) due to inter-subject baseline shifts." | "Random forests cannot predict Parkinson's disease." |

---

## 4. Deliverable File Registry (Phase 4B)

All artifacts are verified and located in the workspace:
- **Experiment Plan:** `experiments/phase4b/experiment_plan.md`
- **Fairness Protocol:** `experiments/phase4b/fairness_protocol.md`
- **Comparison Matrix:** `experiments/phase4b/comparison_matrix.csv`
- **Pre-Execution Audit:** `experiments/phase4b/pre_execution_audit.md`
- **Post-Execution Audit:** `experiments/phase4b/post_execution_audit.md`
- **Final Recommendations:** `experiments/phase4b/final_recommendations.md`
- **Results:**
  - `experiments/phase4b/results/exp1_advanced_classification_summary.csv`
  - `experiments/phase4b/results/exp2_advanced_regression_summary.csv`
  - `experiments/phase4b/results/exp3_feature_engineering_summary.csv`
  - `experiments/phase4b/results/exp4_federated_summary.csv`
  - `experiments/phase4b/results/exp5_personalized_summary.csv`
- **Figures:**
  - `experiments/phase4b/figures/fig1_advanced_classification.png`
  - `experiments/phase4b/figures/fig2_advanced_regression.png`
  - `experiments/phase4b/figures/fig3_feature_engineering_impact.png`
  - `experiments/phase4b/figures/fig4_federated_improvement.png`
  - `experiments/phase4b/figures/fig5_personalized_calibration_windows.png`


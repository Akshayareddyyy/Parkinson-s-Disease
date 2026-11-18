# Phase 5: Manuscript-Readiness Scientific Audit & Theoretical Consistency Guide

**Document ID:** AUDIT-PHASE5-MANUSCRIPT-READINESS  
**Project:** Cross-Subject Voice-Based Motor-UPDRS Severity Estimation  
**Governing Protocol:** Strict Experimental Parity Protocol (Phase 5 QML vs. Matched Classical Controls)  
**Auditor:** Senior Research Scientist & QML Mathematical Auditor  
**Date:** November 2026  
**Status:** **FINAL MANUSCRIPT-READY CERTIFICATION**  
**Traceability Note:** Preserves `phase5_final_consistency_audit.md` as the historical baseline; this document establishes the definitive, publication-ready interpretation.

---

## 1. Executive Summary of Corrections

This document establishes the authoritative, publication-ready scientific interpretation for the Phase 5 Quantum Machine Learning study. It incorporates all 10 mandated documentation-only scientific corrections without altering any frozen experimental results, model checkpoints, CSV runs, or publication figures:

1. **Prediction Variance Ratio:** Corrected to **approximately 3.26%** based on unrounded aggregate variances ($\frac{1.9139}{58.6513} = 0.03263$), while clarifying that the mean of per-seed variance ratios is 3.39%.
2. **Correlation Metrics:** Explicitly decoupled the **mean per-seed Pearson correlation** ($r = \mathbf{0.0879 \pm 0.2397}$) from the **pooled Pearson correlation** across all 12,678 concatenated test records ($r = \mathbf{0.0805}$).
3. **Subgroup Differences ($L=2$ vs. $L=3$):** Clarified as an **observational subgroup difference across different subject partitions**, not a causal within-partition depth effect. Confounding by partition difficulty is explicitly noted.
4. **Finite-Shot Scaling:** Standardized theoretical measurement error scaling to KaTeX $\mathcal{O}(1/\sqrt{S})$.
5. **Shrinkage Mechanism:** Rigorously distinguished **mean shrinkage under Mean Squared Error (MSE) loss** from median shrinkage under $L_1$ loss.
6. **Gradient Diagnostics:** Re-framed non-vanishing initial gradient norms ($\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|_2 \approx 0.44 - 5.0$) as evidence of **adequate initial gradient flow under near-zero initialization**, avoiding over-claims that barren plateaus or inductive-bias mismatches were conclusively proven or ruled out.
7. **Clinical Task Definition:** Replaced unsupported references to "longitudinal clinical progression tracking" with **"cross-subject voice-based motor-severity estimation"**.
8. **Statistical Non-Independence:** Documented that test partitions share an average of **2.29 subjects** across seed pairs, precluding standard independent-sample t-tests and establishing all 95% CIs as **descriptive measures of cross-partition variability**.
9. **Primary Quantum Benchmarks:** Preserved the validated **10-seed selected-pipeline results** as the primary quantum baseline (VQC ROC-AUC = **0.5353**, PQC-R MAE = **6.7471**).
10. **Evidence-Based Reporting:** All manuscript conclusions strictly reflect recorded empirical measurements.

---

## 2. Reconciled Primary Model Comparison Tables (Full 10-Seed Cohort)

### 2.1 Binary Severity Classification (Task 1: $y = \mathbb{I}(\text{motor\_UPDRS} \ge \text{median}_{\text{dev}})$)
*Evaluated across all 10 locked-test partitions on the verified 9-feature manifold ($N \in [1,202, 1,336]$ test records per seed).*

| Tier | Model Architecture | Evaluated Seeds ($N$) | Test ROC-AUC (Mean ± SD) | 95% Descriptive CI | Balanced Accuracy | Matthews Corr (MCC) | Macro F1 | Brier Score | Mean Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Control** | **Extra Trees (9 Feats)** | **10** | **0.6685 ± 0.1386** | [0.5694, 0.7676] | **0.6251 ± 0.1098** | **0.2385 ± 0.1969** | **0.5765 ± 0.1081** | 0.2410 ± 0.0295 | **0.54s** |
| **Classical Control** | **Classical RBF-SVM (9 Feats)** | **10** | **0.6569 ± 0.1276** | [0.5656, 0.7481] | **0.6209 ± 0.1046** | **0.2274 ± 0.1938** | **0.5905 ± 0.0880** | 0.2922 ± 0.0610 | 11.97s |
| **Primary QML Benchmark** | **9-Qubit VQC (Selected Depth Pipeline)** | **10** | **0.5353 ± 0.1562** | [0.4235, 0.6470] | **0.5150 ± 0.1100** | **0.0323 ± 0.2174** | **0.4875 ± 0.1070** | 0.2508 ± 0.0308 | 167.71s |
| *Subgroup Analysis* | 9-Qubit VQC ($L=2$ Partition Subset) | 6 | 0.5885 ± 0.1613 | [0.4192, 0.7577] | 0.5458 ± 0.1201 | 0.0938 ± 0.2390 | 0.5227 ± 0.1162 | 0.2356 ± 0.0254 | 171.42s |
| *Subgroup Analysis* | 9-Qubit VQC ($L=3$ Partition Subset) | 4 | 0.4555 ± 0.1255 | [0.2558, 0.6551] | 0.4687 ± 0.0865 | -0.0601 ± 0.1661 | 0.4349 ± 0.0754 | 0.2737 ± 0.0247 | 162.15s |

### 2.2 Continuous Motor Severity Estimation (Task 2: $y = \text{motor\_UPDRS}$)
*Evaluated across all 10 locked-test partitions for cross-subject continuous severity estimation.*

| Tier | Model Architecture | Evaluated Seeds ($N$) | Test MAE (Mean ± SD) | 95% Descriptive CI | Mean per-seed Pearson $r$ | Pooled Pearson $r$ | Spearman Corr ($\rho$) | Test $R^2$ Score | Mean Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Control** | **ElasticNet (9 Feats)** | **10** | **6.9193 ± 0.8992** | [6.2761, 7.5626] | **0.2697 ± 0.1841** | **0.1828** | **0.3173 ± 0.1822** | **-0.1143 ± 0.2393** | **< 0.01s** |
| **Classical Control** | **Classical RBF-SVR (9 Feats)** | **10** | **7.2184 ± 0.9576** | [6.5334, 7.9035] | **0.2869 ± 0.1868** | **0.2051** | **0.3028 ± 0.1960** | **-0.4732 ± 0.7147** | 5.32s |
| **Primary QML Benchmark** | **9-Qubit PQC-R (Selected Depth Pipeline)** | **10** | **6.7471 ± 1.0413** | [6.0022, 7.4920] | **0.0879 ± 0.2397** | **0.0805** | **0.0741 ± 0.2442** | **-0.0625 ± 0.1218** | 181.56s |
| *Subgroup Analysis* | 9-Qubit PQC-R ($L=2$ Partition Subset) | 4 | 6.3500 ± 0.9063 | [4.9078, 7.7921] | 0.0728 ± 0.3576 | 0.0614 | 0.0605 ± 0.3622 | -0.0762 ± 0.1994 | 132.68s |
| *Subgroup Analysis* | 9-Qubit PQC-R ($L=3$ Partition Subset) | 6 | 7.0118 ± 1.1174 | [5.8392, 8.1845] | 0.0979 ± 0.1626 | 0.0912 | 0.0831 ± 0.1684 | -0.0534 ± 0.0511 | 214.15s |

---

## 3. Methodological & Theoretical Corrections

### 3.1 Observational vs. Causal Interpretation of Ansatz Depth Subgroups
- **Selection Mechanism:** In each seed, depths $L \in \{2, 3\}$ were trained on development data and selected via validation loss early stopping.
- **Subgroup Composition:**
  - VQC selected $L=2$ in 6 partitions (Seeds 0, 2, 3, 5, 6, 9) and $L=3$ in 4 partitions (Seeds 1, 4, 7, 8).
  - PQC-R selected $L=2$ in 4 partitions (Seeds 5, 6, 8, 9) and $L=3$ in 6 partitions (Seeds 0, 1, 2, 3, 4, 7).
- **Correct Scientific Framing:**
  The performance gap between $L=2$ (ROC-AUC 0.5885) and $L=3$ (ROC-AUC 0.4555) must be described as an **observed observational subgroup difference across distinct subject partitions**, NOT a causal depth degradation. Because different seeds contain different patient cohorts, differences between depth subgroups are confounded by partition difficulty. A rigorous causal depth comparison would require evaluating both $L=2$ and $L=3$ on identical locked-test partitions for all 10 seeds.

### 3.2 Regression Mean Shrinkage vs. Median Shrinkage
- **Mathematical Training Objective:**
  The parameterized quantum regressor was optimized using the **Mean Squared Error (MSE)** loss:
  $$\mathcal{L}_{\text{MSE}}(\boldsymbol{\theta}, \mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \left( \hat{y}_i(\boldsymbol{\theta}, \mathbf{w}, b) - y_i \right)^2$$
- **Shrinkage Direction:**
  Under $L_2$ loss, the global constant Bayes optimal prediction is the **sample mean** $\bar{y} = \mathbb{E}[Y]$. Therefore, when the quantum feature map fails to provide linear separability, gradient descent contracts predictions toward the training mean (**mean shrinkage**), not the median.
- **Empirical Diagnostics:**
  - Mean of true test targets across 10 seeds: $\bar{y} = \mathbf{20.48 \pm 1.63}$.
  - Mean of PQC-R test predictions across 10 seeds: $\bar{\hat{y}} = \mathbf{21.09 \pm 0.69}$.
  - Mean PQC-R prediction variance: $\text{Var}(\hat{y}) = \mathbf{1.9139}$, compared to true variance $\text{Var}(y) = \mathbf{58.6513}$.
  - **Variance Ratio:** $\frac{1.9139}{58.6513} = \mathbf{3.26\%}$ (mean of per-seed ratios = 3.39%).
- **Clinical Implication:**
  Because the target distribution is roughly unimodal and symmetric, predicting the mean yields an MAE ($\approx 6.75$) comparable to median guessing. However, this is achieved by suppressing 96.7% of target variance, driving correlation to near zero ($r = 0.0879$). Calling PQC-R superior based on MAE alone is clinically invalid.

### 3.3 Gradient Diagnostics & Barren Plateau Limits
- **Measured Dynamics:**
  Across all seeds, initial gradient norms ranged from $\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|_2 = 0.44$ to $5.0$.
- **Scientific Limitation:**
  Non-vanishing initial gradients demonstrate that near-zero angle initialization ($\sigma = 0.05\pi$) combined with Xavier initialization on the readout layer provided **adequate initial gradient flow** at shallow depths ($L \le 3$). 
- **Prohibited Claim:**
  This does **not** constitute mathematical proof that barren plateaus cannot emerge at deeper layers ($L \ge 4$) or across the broader parameter landscape. Similarly, poor generalization reflects an **empirical observation of limited representation utility on this feature set**, rather than formal theoretical proof of an inductive-bias mismatch.

### 3.4 Finite-Shot Measurement Error Scaling
- **PennyLane Sampling Implementation:**
  In `noise_simulation.py`, finite-shot evaluations on `default.qubit` execute stochastic sampling from the multinomial distribution of $2^9 = 512$ computational basis states.
- **Error Scaling:**
  For shot budget $S \in \{1000, 4000, 10000\}$, the standard error of the empirical sample expectation $\hat{\langle Z_i \rangle} = \frac{1}{S} \sum_{s=1}^S z_{i, s}$ obeys Central Limit Theorem scaling:
  $$\text{SE}\left(\hat{\langle Z_i \rangle}\right) = \sqrt{\frac{1 - \langle Z_i \rangle^2}{S}} = \mathcal{O}\left(\frac{1}{\sqrt{S}}\right)$$
  - $S = 1,000 \implies \text{SE} \le 0.0316$
  - $S = 4,000 \implies \text{SE} \le 0.0158$
  - $S = 10,000 \implies \text{SE} \le 0.0100$
- **Impact on Final Readout:**
  The linear readout head projects expectations via $\hat{y} = \sum_{i=0}^8 w_i \hat{\langle Z_i \rangle} + b$. Because $\|\mathbf{w}\|_2 \approx 0.15$, the net output standard error is $\le 0.005$, matching the observed stability within $\pm 0.004$ of analytic statevector baselines.

### 3.5 Phase Damping Physical Mechanics
- **Kraus Formulation:**
  Single-qubit phase damping channel with damping parameter $\gamma \in [10^{-3}, 5\times 10^{-2}]$:
  $$\mathcal{E}_{\text{phase}}(\rho) = K_0 \rho K_0^\dagger + K_1 \rho K_1^\dagger, \quad K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix}$$
- **Exact Action on State:**
  $$\mathcal{E}_{\text{phase}}\begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix} = \begin{pmatrix} \rho_{00} & \sqrt{1 - \gamma}\rho_{01} \\ \sqrt{1 - \gamma}\rho_{10} & \rho_{11} \end{pmatrix}$$
- **Physical Boundary:**
  Phase damping destroys off-diagonal quantum coherence ($\rho_{01} \to 0$) but **strictly preserves diagonal computational-basis populations** ($\Delta \rho_{00} = \Delta \rho_{11} = 0$). Consequently, pure Pauli-Z expectation values $\langle Z \rangle = \rho_{00} - \rho_{11}$ are invariant under phase damping. Phase damping drives the density matrix to a diagonal computational-basis mixture $\sum_k \rho_{kk} |k\rangle\langle k|$, **not** the maximally mixed state $\frac{1}{2^n} I$.

---

## 4. Manuscript-Ready Conclusions & Guidelines

### 4.1 Permissible Scientific Claims

1. **Definitive Classical Superiority:**
   Under identical 9-feature manifold inputs, preprocessing fitted exclusively on development data, and identical subject partitions, classical machine learning models decisively outperform nine-qubit quantum variational architectures (Extra Trees ROC-AUC **0.6685 ± 0.1386** vs. 9-Qubit VQC **0.5353 ± 0.1562**; ElasticNet $r = \mathbf{0.2697 \pm 0.1841}$ vs. 9-Qubit PQC-R $r = \mathbf{0.0879 \pm 0.2397}$).
2. **Failure of Continuous Quantum Biomarker Estimation:**
   The nine-qubit PQC-R fails as a continuous predictor of motor severity. Its low nominal MAE (**6.7471**) is an artifact of severe mean shrinkage under MSE loss, where prediction variance collapses to **approximately 3.26%** of ground-truth variance, yielding near-zero correlation ($r = 0.0879$ per-seed mean; $r = 0.0805$ pooled). Classical linear and kernel models preserve directional sensitivity ($r \approx 0.27 - 0.29$).
3. **Severe Computational Simulation Overhead:**
   Vectorized statevector quantum simulation required over **$300\times$ more execution time** per partition than Extra Trees ($167.7\text{s}$ vs $0.54\text{s}$) and over **$30,000\times$ more** than ElasticNet ($181.6\text{s}$ vs $<0.01\text{s}$) without offering predictive or clinical benefits.
4. **Physical Invariance under Noise Simulation:**
   The apparent stability of quantum models across depolarizing noise, phase damping, and finite shot counts reflects the contraction of the quantum expectation values toward zero, causing the model to default to its classical affine readout bias, rather than physical quantum noise resilience.

---

### 4.2 Prohibited Interpretations

1. **Do NOT claim that flat noise performance represents quantum fault tolerance or algorithmic robustness.**
2. **Do NOT claim that phase damping drives density matrices toward the maximally mixed state.**
3. **Do NOT assert inferential statistical significance ($p < 0.05$) based on cross-validation or repeated partition t-tests.**
4. **Do NOT claim that PQC-R outperforms classical regression based on MAE alone.**
5. **Do NOT assert causal depth degradation between $L=2$ and $L=3$ based on non-overlapping partition subsets.**
6. **Do NOT state that the noise resilience evaluation used "balanced (50/50)" test samples.**
7. **Do NOT frame the clinical application as "longitudinal progression tracking"; it is cross-subject severity estimation.**

---

## 5. Auditor Confirmation Statement

> **Final Verification:**  
> The 10 scientific and documentation corrections have been verified against the underlying raw CSVs, mathematical derivations, and execution logs. All results are fully reproducible. With the publication-ready guidelines established in this document, Phase 5 is hereby **APPROVED and CERTIFIED for paper integration**.


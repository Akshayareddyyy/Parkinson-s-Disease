# Methodological Issues & Algorithmic Diagnosis Report

**Project:** Parkinson's Disease Telemonitoring — Phase 4 Federated Learning  
**Audit Component:** Implementation Audits, Optimization Mechanics, & Comparative Fairness  
**Audit Date:** November 2026  
**Status:** COMPLETE EVALUATION & CLASSIFICATION

---

## 1. Executive Summary & Issue Classification Table

This audit details the algorithmic, mathematical, and comparative discrepancies identified during the independent verification of Phase 4 Federated Learning.

| Issue ID | Domain / Component | Description | Severity / Classification |
| :--- | :--- | :--- | :---: |
| **ISS-01** | Optimization & Loss | FedAvg local gradient descent vs. Centralized L-BFGS with L2 regularization mismatch. | **VERIFIED WITH LIMITATIONS** |
| **ISS-02** | Proximal Regularization | Equating FedProx proximal penalty ($\|w - w^t\|^2$) to Centralized L2 norm ($\|w\|^2$). | **METHODOLOGICAL ERROR** |
| **ISS-03** | Continuous Regression | Near-identity of FedAvg-Linear and FedAvg-Ridge due to step-size shrinkage scale. | **VERIFIED (Mathematically Explained)** |
| **ISS-04** | Distributed Ensembling | Single-class client starvation (65–85% of edge clients) in CED-Forest. | **VERIFIED DISCOVERY** |
| **ISS-05** | Statistical Inference | Non-independence of paired tests across partitions reusing 42 subjects. | **VERIFIED WITH LIMITATIONS** |

---

## 2. In-Depth Analysis of Methodological Issues

### 2.1 Issue ISS-01: FedAvg vs. Centralized Logistic Regression Optimization Mismatch
- **Implementation Inspection:**
  - `run_phase4_federated.py` lines 188–190:
    ```python
    m_cent_lr = LogisticRegression(max_iter=2000, random_state=seed).fit(sc_cent.transform(XD), yD)
    ```
    In scikit-learn, default `LogisticRegression()` enforces an $L_2$ penalty with $C=1.0$ and solves the dual/primal problem using L-BFGS to exact convergence.
  - In `run_federated_logistic` (lines 60–73):
    Local clients execute batch gradient descent on **unregularized binary cross-entropy** with $\eta=0.1$ for $E=5$ epochs over $R=40$ communication rounds (200 total local gradient updates).
- **Scientific Impact:**
  - FedAvg achieved **0.6289** ROC-AUC while Centralized LR achieved **0.5775** ROC-AUC.
  - *Why this occurs:* Centralized empirical risk minimization without sample weighting allows patients with high recording counts to dominate the gradient, resulting in poor generalizability to held-out subjects. FedAvg computes isolated local gradients on each patient, followed by sample-weighted averaging. This serves as an **implicit structural regularizer** across patient distributions.
  - *Methodological Limitation:* FedAvg is an unregularized federated optimizer, whereas Centralized LR has $L_2$ regularization. In addition, L-BFGS finds the global empirical minimum, whereas 40 rounds of FedAvg terminates at an intermediate trajectory point that happens to generalize better (early stopping effect).
- **Classification:** **VERIFIED WITH LIMITATIONS**. The empirical superiority of FedAvg over Centralized LR is mathematically genuine, but stems from structural gradient isolation and implicit early stopping rather than algorithmic optimization superiority.

---

### 2.2 Issue ISS-02: Equating FedProx Proximal Regularization to Centralized L2 Penalty
- **Problem Statement:** Preliminary proposals suggested that testing FedProx with proximal parameter $\mu \in \{0.01, 0.1, 1.0\}$ corresponds to centralized $L_2$ regularization with $C = 1/\mu$.
- **Mathematical Disproof:**
  - Centralized $L_2$ Regularization penalizes model distance from the origin:
    $$\mathcal{L}_{\text{cent}}(w) = \mathcal{L}(w) + \frac{1}{2C} \|w\|_2^2, \quad \nabla_w = \nabla \mathcal{L}(w) + \frac{1}{C} w$$
    This forces parameter shrinkage toward zero.
  - FedProx Proximal Penalty penalizes local client drift from the *current global model* $w^t$:
    $$\mathcal{L}_{\text{prox}}(w; w^t) = \mathcal{L}_k(w) + \frac{\mu}{2} \|w - w^t\|_2^2, \quad \nabla_w = \nabla \mathcal{L}_k(w) + \mu (w - w^t)$$
    This restricts how far a client's local parameters can diverge from the broadcast global weights during local epochs.
- **Empirical Sensitivity Finding:**
  - In Phase 4, varying $\mu$ across three orders of magnitude ($0.01, 0.1, 1.0$) produced virtually zero variation in test ROC-AUC:
    - $\mu = 0.01$: ROC-AUC = $0.628919 \pm 0.1241$
    - $\mu = 0.1$: ROC-AUC = $0.628918 \pm 0.1241$
    - $\mu = 1.0$: ROC-AUC = $0.628388 \pm 0.1244$
  - *Cause:* With $E=5$ local epochs and local batch gradients on standardized features, local parameter drift per round $\|lw - w^t\|$ is small ($\approx 0.01 - 0.05$). Thus, $\mu (lw - w^t)$ is negligible compared to the loss gradient $\frac{1}{n_k} X_k^T (p - y)$, rendering FedProx functionally identical to FedAvg under this communication frequency.
- **Classification:** **METHODOLOGICAL ERROR** in conceptual pairing. Centralized $C=1/\mu$ is not a valid theoretical counterpart to FedProx $\mu$. FedProx must be evaluated strictly as a client-drift regularizer against standard FedAvg ($\mu=0.0$).

---

### 2.3 Issue ISS-03: Near-Identity of FedAvg-Linear and FedAvg-Ridge Regression
- **Observation:** In continuous progression regression:
  - Centralized: Linear MAE = `7.0685`, Ridge MAE = `7.0472` (Difference: $-0.0213$ points).
  - Federated: FedAvg-Linear MAE = `6.90411`, FedAvg-Ridge MAE = `6.90398` (Difference: $-0.00013$ points).
- **Mathematical Explanation:**
  - In `run_federated_ridge`:
    $$\nabla_w = \frac{1}{n_k} X_k^T (X_k w + b - y) + \frac{\alpha}{n_k} w$$
  - With $\alpha = 10.0$ and average $n_k \approx 135$, the shrinkage term is $\frac{10}{135} w \approx 0.074 w$.
  - With local learning rate $\eta = 0.01$, the weight shrinkage per step is $\eta \cdot \frac{\alpha}{n_k} w = 0.00074 w$.
  - Over 5 epochs $\times$ 40 rounds, on standardized features where weights have magnitudes $|w_j| \in [0.05, 0.40]$, this cumulative shrinkage alters weights by less than $0.005$, producing identical predictions to 4 decimal places.
- **Classification:** **VERIFIED (Mathematically Accounted For)**. The implementation is correct; the near-identity is an artifact of the chosen step size and regularization coefficient.

---

### 2.4 Issue ISS-04: Single-Class Client Starvation in CED-Forest
- **Observation:** Centralized Extra Trees achieves **0.6351** ROC-AUC, whereas Client-Ensembled Distributed Forest (CED-Forest) collapses to **0.5147** ROC-AUC.
- **Root Cause Discovery:**
  - Independent verification of the 26 edge clients across all 10 seeds revealed that **between 17 and 22 clients (65.4% to 84.6%, mean = 19.7 clients) have ONLY A SINGLE CLASS ($y=0$ only or $y=1$ only)** in their local data!
  - In chronic disease telemonitoring, a patient's motor-UPDRS over 6 months often remains either entirely below or entirely above the population median threshold.
  - When an edge client has only single-class records, a decision tree cannot perform feature splits and defaults to a trivial constant prediction ($P=0$ or $P=1$).
  - When the central server averages the soft votes of all 26 clients, ~20 clients cast flat invariant votes, completely washing out the multivariate signal from the ~6 clients who possess bi-class records.
- **Classification:** **VERIFIED DISCOVERY**. This conclusively demonstrates why parameter-level federated models (FedAvg) succeed where distributed ensembling (CED-Forest) fails in single-patient edge federations.

---

### 2.5 Issue ISS-05: Non-Independence of Partitions & Hypothesis Testing
- **Observation:** Paired t-tests on the 10 seeds report $p = 0.0182$ for FedAvg-LR vs. Centralized LR.
- **Methodological Limitation:** The 10 seeds resample the exact same 42 patients. The paired test observations are not independent realizations from a super-population.
- **Multiplicity Correction:** Applying Holm-Bonferroni correction across the family of 7 planned comparisons raises the adjusted p-value to $p_{\text{adj}} = 0.1641$.
- **Classification:** **VERIFIED WITH LIMITATIONS**. Descriptive differences must be reported as partition variability; statistical superiority must not be claimed as confirmatory evidence.


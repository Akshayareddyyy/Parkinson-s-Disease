# Recommended Paper Claims & Scientific Framing Guidelines

**Project:** Parkinson's Disease Telemonitoring — Phase 4 Federated Learning  
**Document:** Manuscript Claim Classification & Reviewer-Proof Scientific Framing  
**Date:** November 2026  
**Status:** BINDING PUBLICATION RECOMMENDATIONS

---

## 1. Executive Summary & Reviewer Guidance

To guarantee acceptance in competitive peer-reviewed venues and eliminate the risk of rejection due to over-claiming, all manuscript assertions regarding Phase 4 Federated Learning must be classified and framed strictly according to this guide.

### Claims Classification Framework
- **GREEN (Scientifically Defensible & Verified):** Strong empirical support; exact methodology; fully reproducible.
- **YELLOW (Verified With Limitations / Qualified Claim):** Empirically sound, but requires explicit disclosure of assumptions, non-confirmatory statistical framing, or topology-specific scoping.
- **RED (Unsupported / Rejected Claim):** Empirically refuted, theoretically mismatched, or scientifically invalid. **Must NOT appear in the manuscript.**

---

## 2. Claim Classification Matrix

| Claim Topic | Raw Hypothesis / Common Pitfall | Recommended Scientific Framing | Classification |
| :--- | :--- | :--- | :---: |
| **FedAvg vs. Centralized LR** | *"FedAvg is statistically superior to centralized logistic regression (p < 0.05)."* | *"In natural edge topologies, FedAvg achieves higher test ROC-AUC (0.629 vs. 0.577) than centralized unweighted ERM by enforcing structural patient-level gradient regularization; however, after multiple-comparison correction across repeated partitions, this reflects descriptive partition dispersion rather than confirmatory superiority."* | **YELLOW** (Qualified) |
| **FedProx Regularization** | *"FedProx significantly improves over FedAvg and corresponds to centralized L2 regularization with C = 1/μ."* | *"Varying the FedProx proximal parameter μ across three orders of magnitude (0.01 to 1.0) yielded indistinguishable performance (ROC-AUC ~0.629), demonstrating that local parameter drift is already constrained under standard local epoch budgets. FedProx proximal penalties restrict drift from global weights and must not be conflated with centralized L2 weight shrinkage."* | **RED** (Refuted) $\to$ **YELLOW** (Re-framed) |
| **Decentralized Preprocessing** | *"Our framework provides provable privacy via decentralized standardization."* | *"Decentralized standardization via sufficient statistics aggregation achieves exact algebraic parity with scikit-learn StandardScaler (numerical error < 1e-13) without transmitting raw audio signals. However, sharing raw moments does not provide formal differential privacy guarantees."* | **YELLOW** (Qualified) |
| **Federated Regression** | *"Federated Ridge regression significantly beats centralized Ridge regression."* | *"FedAvg-Ridge achieved lower test MAE (6.904 vs. 7.047) and higher Pearson correlation (r = 0.288 vs. 0.247) on held-out test subjects, but the paired difference (95% CI: [-0.415, +0.129], p = 0.264) indicates the improvement is within expected partition variability."* | **YELLOW** (Qualified) |
| **Distributed Decision Trees** | *"Federated decision trees cannot work for Parkinson's disease telemonitoring."* | *"Client-Ensembled Distributed Forests (CED-Forest) collapsed to near chance (ROC-AUC ~0.51) due to single-class starvation in extreme single-patient edge nodes (where 65–85% of clients have only one class). This finding highlights the failure of naive soft-voting under extreme physiological non-IID fragmentation, rather than a universal limitation of federated tree algorithms."* | **GREEN** (Defensible Discovery) |
| **Client Subsampling** | *"Subsampling clients degrades model utility."* | *"Subsampling 50% of edge clients per round halved total communication overhead (from 325 KB to 162.5 KB) while preserving strong predictive utility (ROC-AUC 0.618 vs. 0.629), with every client receiving at least 12 training rounds."* | **GREEN** (Defensible) |

---

## 3. Explicit Draft Text for Manuscript Sections

### 3.1 Abstract / Results Summary
> *"When deploying federated learning across 26 autonomous telemonitoring devices, FedAvg-LR achieved a mean test ROC-AUC of 0.6289 ± 0.1241 across 10 subject-wise partitions, compared to 0.5775 ± 0.1066 for unweighted centralized empirical risk minimization. This empirical advantage arises because patient-level gradient averaging acts as an implicit regularizer against over-represented individual acoustic baselines. In continuous motor-UPDRS progression regression, FedAvg-Ridge achieved a test MAE of 6.9040 ± 0.8675 points and Pearson correlation r = 0.2878. Conversely, distributed decision tree ensembling collapsed to near-chance (ROC-AUC 0.5147 ± 0.0647) due to single-class client starvation (mean 19.7 of 26 clients possessing only one class label), demonstrating that parameter-level optimization is robust to extreme physiological non-IID partitioning where distributed soft-voting fails."*

### 3.2 Discussion: Statistical & Partitioning Limitations
> *"We emphasize that the 10 subject-wise cross-validation partitions reuse the same 42 patient cohorts. Consequently, paired differences cannot be interpreted as confirmatory evidence from independent clinical populations. After applying Holm-Bonferroni multiple-comparison correction across the evaluation family, paired differences between FedAvg and centralized baselines are descriptive of partition dispersion (p_adj = 0.1641 for ROC-AUC; p_adj = 0.7734 for Ridge MAE). Furthermore, while decentralized standardization matches centralized StandardScaler to machine precision (< 1e-13) without transmitting raw speech signals, sufficient statistics exchange does not provide formal differential privacy guarantees."*

---

## 4. Banned Claims & Mandatory Corrections Checklist

- [x] **BANNED:** Calling client-weighted soft-voting of decision trees "FedAvg".  
  **MANDATORY:** Must be labeled **Client-Ensembled Distributed Forest (CED-Forest)** or **Distributed Soft-Voting Ensemble**.
- [x] **BANNED:** Claiming FedProx proximal penalty is mathematically equivalent to centralized $L_2$ regularized Logistic Regression ($C = 1/\mu$).  
  **MANDATORY:** Must state that FedProx penalizes local client drift from round-start weights ($\|w - w^t\|^2$), not parameter norm ($\|w\|^2$).
- [x] **BANNED:** Claiming formal privacy or differential privacy without adding calibrated noise mechanisms.  
  **MANDATORY:** Must describe decentralized standardization as data-local sufficient statistics aggregation.
- [x] **BANNED:** Claiming statistical superiority based on raw paired t-test $p < 0.05$ without disclosing partition subject reuse.  
  **MANDATORY:** Disclose subject reuse and report Holm-adjusted $p$-values.


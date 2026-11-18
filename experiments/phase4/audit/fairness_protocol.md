# Phase 4: Strict Fair and Equal Experimental Comparison Protocol

**Project:** Parkinson's Disease Telemonitoring — Federated Learning Study  
**Document:** Experimental Fairness & Scientific Parity Protocol  
**Scope:** Phase 4 Federated Learning (FL) Benchmarked Against Phase 3 / Phase 3B Centralized Baselines  
**Status:** MANDATORY GOVERNING SPECIFICATION

---

## 1. Rationale & Rejection Feedback Context

Previous conference submissions received critical reviewer rejections concerning **unequal, non-rigorous, or mismatched experimental comparisons**. Specifically, reviewers highlighted:
1. *Mismatched data partitioning:* Comparing federated models on subject-grouped splits against centralized models evaluated on random row-level splits.
2. *Unequal training sets:* Subsampling or pruning records for federated rounds while centralized baselines trained on full datasets (or vice versa).
3. *Cross-task conflation:* Comparing continuous regression error against classification accuracy, or contrasting zero-shot absolute severity against baseline-calibrated progression delta.
4. *Inaccurate algorithm naming:* Describing client-weighted soft-voting of decision trees as "FedAvg", which implies parameter-level gradient/weight averaging.
5. *Unequal feature spaces:* Comparing low-dimensional models against full-feature baselines without matched controls.

This protocol establishes **non-negotiable parity constraints** for Phase 4 Federated Learning. **Experimental fairness strictly supersedes pursuing artificially higher accuracy.**

---

## 2. Principle 1: Identical Dataset and Subject Partitions

All centralized and federated experiments must execute on the exact same dataset and partition structure:
- **Dataset:** UCI Parkinson's Telemonitoring Dataset (`data/parkinsons_updrs.csv`, 5,875 rows, 42 subjects).
- **Cryptographic Hash:** SHA-256 = `927b23aebd30d1f11022b32c7da90a6f209122834e4c28ffe60146047663db0c`.
- **Subject Partitions:** Exactly the 10 audited subject-wise partitions from Phase 3 and Phase 3B (`common.make_sets(df, seed)`):
  - **Development Cohort ($\mathcal{D}$):** Exactly 26 subjects (~62% of subjects, $N \ge 3,517$ records).
  - **Validation Cohort ($\mathcal{V}$):** Exactly 7 subjects (~17% of subjects, $N \approx 1,090$ records).
  - **Locked-Test Cohort ($\mathcal{T}$):** Exactly 9 subjects (~21% of subjects, $N \approx 1,268$ records).
- **Disjoint Partition Invariant:**
  $$\mathcal{D} \cap \mathcal{V} = \emptyset, \quad \mathcal{D} \cap \mathcal{T} = \emptyset, \quad \mathcal{V} \cap \mathcal{T} = \emptyset$$
- **Rule:** In Phase 4, the 26 development subjects form the client federation pool. The 7 validation subjects serve strictly as the global server-side validation set for round monitoring and early stopping. The 9 locked-test subjects are held out completely and evaluated centrally on the final aggregated global model once per seed.

---

## 3. Principle 2: Equal Training Data (100% Development Utilization)

- **Full Data Requirement:** Every eligible development recording must be allocated to a federated client and utilized during federated training.
- **Zero Subsampling:** No downsampling, no arbitrary record truncation, no dropped subjects.
- **Client Accounting:** For any client allocation scheme with $K$ clients:
  $$\sum_{k=1}^K N_k = N_{\mathcal{D}} \quad (100\% \text{ of development records accounted for})$$
  where $N_k$ is the number of local records at client $k$.

---

## 4. Principle 3: Client Topology & Simulation Scenarios

To investigate both natural physiological heterogeneity and regional clinical networks, Phase 4 evaluates two standardized client topologies:

### 4.1 Topology 1: One Client per Patient ($K=26$ Autonomous Devices)
- **Clinical Mapping:** Remote patient telemonitoring where each patient's home device functions as a decentralized edge client.
- **Data Distribution:** Natural, extreme physiological non-IID data distribution (each client possesses exclusively one patient's longitudinal voice recordings, $n_k \in [101, 168]$).
- **Client Participation:** Evaluated under full participation ($\text{fraction} = 1.0$, all 26 clients per round) and partial participation ($\text{fraction} = 0.5$, 13 clients sampled uniformly at random per round).

### 4.2 Topology 2: Multi-Patient Clinical Centers ($K \in \{3, 5, 6\}$ Clinics)
- **Clinical Mapping:** Regional hospitals or clinical research centers aggregating multi-patient cohorts.
- **Assignment Strategies:**
  1. *Natural Assignment:* Consecutive subject grouping representing regional patient recruitment.
  2. *Label-Balanced Benchmark Assignment:* Greedy serpentine allocation to benchmark the empirical penalty of label imbalance.

---

## 5. Principle 4: Matched Model Comparisons & Transparent Naming

For every federated algorithm, an exact matched centralized counterpart must be trained on the identical $N_{\mathcal{D}}$ development records:

| Federated Method | Matched Centralized Baseline | Aggregation Mechanism | Architectural Disclosure |
| :--- | :--- | :--- | :--- |
| **FedAvg-LR** | Centralized Logistic Regression | Exact parameter-level averaging ($w = \sum \frac{n_k}{N} w_k$) | Linear/Logistic models support exact mathematical FedAvg. |
| **FedProx-LR** | Centralized $L_2$-Regularized LR | Parameter-level averaging with proximal penalty ($\mu \in \{0.01, 0.1, 1.0\}$) | Regularized linear model mitigating local client drift. |
| **FedAvg-Ridge** | Centralized Ridge Regression | Parameter-level coefficient averaging | Continuous progression regression under parameter aggregation. |
| **Client-Ensembled Distributed Forest (CED-RF / CED-ET)** | Centralized Random Forest / Extra Trees | Server-side sample-weighted soft-voting of client trees | **DECISION TREES CANNOT BE FEDERATED AT THE PARAMETER LEVEL.** Splitting rules and tree topologies cannot be averaged element-wise. Describing client tree voting as "FedAvg" is scientifically invalid; it must be labeled **Distributed Ensemble (CED)**. |

---

## 6. Principle 5: Decentralized Preprocessing Without Data Leakage

Centralized pipelines typically fit a `StandardScaler` across the entire development set. In federated learning, moving raw data to a central server violates privacy. Phase 4 implements **Decentralized Sufficient Statistics Aggregation**:
1. Each client $k$ computes local summary statistics:
   $$n_k = |X_k|, \quad S_{1, k} = \sum_{i=1}^{n_k} x_{k, i}, \quad S_{2, k} = \sum_{i=1}^{n_k} x_{k, i}^2$$
2. The server aggregates these scalars across clients:
   $$N = \sum_{k=1}^K n_k, \quad \mu_{\text{global}} = \frac{1}{N}\sum_{k=1}^K S_{1, k}, \quad \sigma_{\text{global}}^2 = \frac{1}{N}\sum_{k=1}^K S_{2, k} - \mu_{\text{global}}^2$$
3. Global mean $\mu_{\text{global}}$ and standard deviation $\sigma_{\text{global}}$ are broadcasted to clients to standardize local features.
- **Fairness Guarantee:** Yields mathematical equivalence to centralized scaling without transferring a single patient data point or leaking validation/test data.

---

## 7. Principle 6: Hyperparameter & Computational Resource Parity

- **Standardized FL Training Schedule:**
  - Communication Rounds: $R = 40$ rounds.
  - Local Epochs: $E \in \{1, 5\}$ local SGD passes per round.
  - Learning Rate: Initial $\eta = 0.1$ with decay schedule.
  - Proximal Parameter: $\mu \in \{0.0, 0.01, 0.1, 1.0\}$.
- **Communication Overhead Accounting:**
  - For parameter-level federated linear models:
    $$\text{Bytes Per Round} = 2 \times |S| \times (d + 1) \times 8 \text{ bytes}$$
    where $|S|$ is the number of active clients in the round and $d=19$ features.
  - For distributed tree ensembles:
    $$\text{Bytes Transmitted} = \sum_{k=1}^K \text{SerializedSize}(\text{LocalForest}_k)$$
  - All communication overheads must be logged and reported alongside accuracy metrics.

---

## 8. Principle 7: Complete and Identical Metric Suite

### 8.1 Classification Suite
- Receiver Operating Characteristic Area Under Curve (ROC-AUC)
- Precision-Recall Area Under Curve (PR-AUC)
- Balanced Accuracy ($\frac{\text{Sens} + \text{Spec}}{2}$)
- Sensitivity (Recall) & Specificity
- Matthews Correlation Coefficient (MCC)
- Macro-Averaged F1-Score ($F_1$)
- Full Confusion Matrix ($\text{TN}, \text{FP}, \text{FN}, \text{TP}$)

### 8.2 Regression Suite
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination ($R^2$)
- Pearson Correlation Coefficient ($r$)
- Spearman Rank Correlation ($\rho$)
- Mean Per-Subject MAE & Inter-Subject Error Standard Deviation

---

## 9. Final Research Acceptance Rule

A federated method may be reported as competitive or viable only if:
1. It is evaluated on the exact 10 locked-test partitions with zero test data in federation.
2. It is benchmarked directly against its matched centralized counterpart on the identical task.
3. Computational communication overhead and local client variance are fully disclosed.
4. Differences in performance are reported descriptively without false claims of privacy guarantees (e.g., differential privacy must not be claimed unless formally parameterized and added).


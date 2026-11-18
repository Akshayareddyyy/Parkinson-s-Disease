# Phase 3B: Scientific Methodology & Pipeline Audit Report

**Project:** Parkinson's Disease Telemonitoring — Motor Progression & Severity Prediction  
**Dataset:** UCI Parkinson's Telemonitoring Dataset (5,875 recordings, 42 unique subjects)  
**Audit Scope:** Methodology, Data Leakage Prevention, Target Construction, Feature Space, & Cross-Validation Architecture  
**Verification Date:** November 2026  
**Status:** PASS (Strict Compliance with Scientific Validation Protocols)

---

## 1. Executive Summary & Core Scientific Commitments

Phase 3B expands upon the validated Phase 3 centralized classical machine learning experiments by exploring scientifically rigorous model improvements. This audit verifies that all experimental designs adhere strictly to established machine learning best practices for biomedical and clustered tabular data.

### Core Validation Checklist
1. **Preservation of Benchmark Results:** The Phase 3 baseline results (Extra Trees: Mean ROC-AUC = 0.629, Mean Balanced Accuracy = 0.607, Mean MCC = 0.205) are preserved unaltered. All Phase 3 prediction logs, per-seed matrices, and statistical corrections in `results/phase3_*` remain intact and read-only.
2. **Full Development Dataset Utilization:** Zero subsampling is permitted. For all 10 random seeds ($0-9$), models are trained on 100% of eligible development records ($N \ge 3,517$ per seed), satisfying the non-negotiable project mandate.
3. **Strict Subject-Wise Disjoint Partitions:** In every partition, the 42 subjects are strictly separated into Development (26 subjects, ~62%), Validation (7 subjects, ~17%), and Locked-Test (9 subjects, ~21%) cohorts. Zero subject overlap exists across splits:
   $$\mathcal{D} \cap \mathcal{V} = \emptyset, \quad \mathcal{D} \cap \mathcal{T} = \emptyset, \quad \mathcal{V} \cap \mathcal{T} = \emptyset$$
4. **Absolute Data Leakage Elimination:** All data scalers, power transformers, feature selectors, correlation filters, and baseline statistics are fitted strictly inside training folds. Neither validation nor locked-test records or labels ever influence preprocessing or model selection.
5. **Separation of Clinical Tasks:** The distinction between binary severity classification (screening at threshold $\tau$) and continuous disease progression regression ($motor\_UPDRS \in [5.0377, 39.5110]$) is strictly maintained with specialized objective functions and evaluation metrics.

---

## 2. Dataset Integrity & Feature Space Verification

### 2.1 Cryptographic Dataset Verification
- **File:** `data/parkinsons_updrs.csv`
- **SHA-256 Checksum:** `927b23aebd30d1f11022b32c7da90a6f209122834e4c28ffe60146047663db0c`
- **Row Count:** 5,875 records
- **Subject Count:** 42 unique subjects (Subject ID integers: 1 through 42)
- **Missing / Infinite Values:** Exactly 0 missing values; 0 infinite values.

### 2.2 Target Formulation & Range Truth
- **Continuous Target (`motor_UPDRS`):**
  - Minimum: `5.0377`
  - Maximum: `39.5110` (Subject 21 at `test_time = 186.88`)
  - Mean: `21.2995`
  - Standard Deviation: `8.1292`
  - Median: `20.8710`
  *(Note on historical discrepancy: A preliminary proposal mentioned a maximum of 39.388; verification confirms the actual exact maximum is 39.5110. `total_UPDRS` ranges from 7.000 to 54.992.)*
- **Binary Target Formulation:**
  For each random seed $s \in \{0, \dots, 9\}$, the classification threshold $\tau_s$ is frozen strictly as the median `motor_UPDRS` of the development partition $\mathcal{D}_s$:
  $$\tau_s = \text{median}(\{y_i : i \in \mathcal{D}_s\})$$
  $$y_{i, \text{binary}} = \mathbf{1}[y_i \ge \tau_s]$$
  Locked-test labels are computed using the frozen development threshold $\tau_s$, completely preventing target distribution leakage.

### 2.3 Permitted Predictors & Excluded Columns
- **Permitted Predictors (19 Features):**
  1. `age` (integer, demographic)
  2. `sex` (binary: 0 = male, 1 = female)
  3. `test_time` (float, days from baseline recording)
  4. `Jitter(%)` (multidimensional voice jitter)
  5. `Jitter(Abs)`
  6. `Jitter:RAP`
  7. `Jitter:PPQ5`
  8. `Jitter:DDP`
  9. `Shimmer` (voice shimmer measure)
  10. `Shimmer(dB)`
  11. `Shimmer:APQ3`
  12. `Shimmer:APQ5`
  13. `Shimmer:APQ11`
  14. `Shimmer:DDA`
  15. `NHR` (noise-to-harmonics ratio)
  16. `HNR` (harmonics-to-noise ratio)
  17. `RPDE` (recurrence period density entropy)
  18. `DFA` (detrended fluctuation analysis)
  19. `PPE` (pitch period entropy)
- **Strictly Excluded Columns:**
  - `subject#`: Excluded from predictor matrix $X$; retained exclusively as grouping vector $groups$ for clustered cross-validation.
  - `total_UPDRS`: Excluded entirely from all models to prevent clinical target leakage (since $total\_UPDRS = motor\_UPDRS + mental\_UPDRS$).

---

## 3. Methodological Design of Phase 3B Experiments

```
Phase 3B Experimental Architecture
├── Exp 1: Subject-Balanced Learning (w_ij = N_tr / (K_tr * n_j))
├── Exp 2: Expanded Classical Suite (CatBoost, XGBoost, HistGB, RF, ET, SVR, ElasticNet, Ridge, PLS)
├── Exp 3: Preprocessing & Fold-Contained Feature Selection (Standard, Robust, Yeo-Johnson; ANOVA, MI, L1, RFE)
├── Exp 4: Nested GroupKFold Cross-Validation (5 Outer Folds x 4 Inner Folds)
└── Exp 5: Subject-Aware Normalization & Personalized Prediction (Scenario A: Zero-Shot vs Scenario B: Baseline Onboarding k=5,10)
```

### 3.1 Experiment 1: Subject-Balanced Learning (Sample Weighting)
Biomedical cohort datasets often display patient recording imbalance (here, records per subject range from 101 to 168). Unweighted training risks over-representing highly recorded individuals.
- **Formulation:** For each training fold with $N_{\text{train}}$ total recordings across $K_{\text{train}}$ unique subjects:
  $$w_{ij} = \frac{N_{\text{train}}}{K_{\text{train}} \cdot n_j}$$
  where $n_j$ is the number of development recordings for subject $j$.
- **Validation Constraints:**
  - $\sum_{i \in \text{train}} w_{ij} = N_{\text{train}}$ (preserves effective training sample size).
  - Every individual training subject contributes equal total weight: $\sum_{i \in \text{subject } j} w_{ij} = \frac{N_{\text{train}}}{K_{\text{train}}}$.
  - Weights are computed strictly on training indices; validation and test sets are strictly unweighted ($w = 1$).

### 3.2 Experiment 2: Comprehensive Classical Algorithm Suite
Evaluates whether modern gradient boosting architectures (XGBoost, CatBoost) or regularized linear models improve upon tree ensembles.
- **Regression Suite:**
  - Trivial Baselines: Dummy Mean regressor ($\hat{y} = \bar{y}_{\text{train}}$), Simple Linear `test_time` trend.
  - Regularized Linear: Ridge ($\alpha \in \{0.1, 1.0, 10.0, 100.0\}$), ElasticNet ($l_1 \in \{0.2, 0.5, 0.8\}$), PLS Regression ($n_{\text{comp}} \in \{3, 5, 8\}$).
  - Kernel SVR: RBF SVR ($C \in \{1.0, 10.0\}$, $\epsilon \in \{0.1, 0.5\}$).
  - Ensembles: Random Forest, Extra Trees, HistGradientBoosting ($n_{\text{est}} \in \{150, 200\}$).
  - Gradient Boosters: XGBoost (`n_estimators=150`, `learning_rate=0.05`, `max_depth=4`), CatBoost (`iterations=250`, `depth=5`, `l2_leaf_reg=3.0`).
- **Classification Suite:** Evaluated with identical algorithms for binary severity prediction.

### 3.3 Experiment 3: Preprocessing, Transformations & Fold-Contained Feature Selection
Evaluates non-linear feature transformations and sparse feature subsets to mitigate collinearity among acoustic dysphonia metrics.
- **Scalers & Transformations:**
  1. `StandardScaler`: Zero mean, unit variance.
  2. `RobustScaler`: Median centering, interquartile range scaling (robust to acoustic measurement outliers).
  3. `PowerTransformer (Yeo-Johnson)`: Normalizing non-linear power transform accommodating positive and negative values.
- **Feature Selection Mechanisms (Pipeline-Encapsulated):**
  1. `ANOVA F-Score`: Univariate feature scoring (Top 10 features).
  2. `Mutual Information`: Non-linear dependency estimation (Top 60th percentile).
  3. `L1-Regularized Sparsity`: Sparse selection via `Lasso(alpha=0.05)` and `L1-LogisticRegression(C=0.1)`.
  4. `Recursive Feature Elimination (RFE)`: Stepwise elimination using tree-based importance (Top 10 features).
  5. `Training-Fold Correlation Filter`: Dropping collinear features where $|r| > 0.85$ based strictly on the training fold covariance matrix.

### 3.4 Experiment 4: Nested GroupKFold Cross-Validation
Eliminates hyperparameter selection bias through nested cross-validation:
- **Outer Loop:** 5-fold `GroupKFold` partitioning the 26 development subjects into 5 outer folds (outer evaluation subjects $\approx 5$ subjects, completely withheld from training and inner selection).
- **Inner Loop:** 4-fold `GroupKFold` partitioning the 21 outer training subjects into inner training and inner validation folds.
- **Verification:** Inner tuning optimizes parameters over inner folds; the optimal model is refitted on all 21 outer training subjects and evaluated once on the unseen outer evaluation subjects.

### 3.5 Experiment 5: Subject-Aware Normalization & Personalized Calibration
Formulates two clinically distinct paradigms in telemonitoring:
- **Scenario A: Global Zero-Shot Prediction:**
  - Evaluates generalization to an entirely new, uncalibrated screening patient.
  - Predictors are standardized using global cohort statistics: $z = (x - \mu_{\text{cohort}}) / \sigma_{\text{cohort}}$.
- **Scenario B: Subject-Calibrated Personalized Prediction:**
  - Evaluates longitudinal monitoring of an onboarded patient.
  - The first $k \in \{5, 10\}$ chronological recordings (`test_time` sorted) establish subject baseline statistics:
    $$\mu_{s, k} = \frac{1}{k}\sum_{t=1}^k x_{s, t}, \quad \sigma_{s, k} = \sqrt{\frac{1}{k-1}\sum_{t=1}^k (x_{s, t} - \mu_{s, k})^2}$$
    (With numerical safety: $\sigma_{s, k} = \max(\sigma_{s, k}, 10^{-6})$).
  - Subsequent recordings ($t > k$) are normalized relative to patient baseline:
    $$z_{s, t} = \frac{x_{s, t} - \mu_{s, k}}{\sigma_{s, k}}$$
  - Evaluates both absolute future severity ($y_{s, t}$) and disease progression delta ($\Delta y_{s, t} = y_{s, t} - y_{s, \text{baseline}}$).
  - **Leakage Prevention:** Training and locked-test cohorts undergo identical simulation protocols independently.

---

## 4. Evaluation Metrics & Statistical Standards

### 4.1 Regression Metrics
- **Mean Absolute Error (MAE):** $\frac{1}{N}\sum |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Pearson Correlation ($r$) & Spearman Rank Correlation ($\rho$):** Measuring linear and monotonic tracking.
- **Subject-Level Metrics:** Mean Per-Subject MAE ($\frac{1}{K}\sum_{j=1}^K \text{MAE}_j$) and inter-subject variance ($\text{std}(\text{MAE}_j)$).

### 4.2 Classification Metrics
- **ROC-AUC & PR-AUC:** Area under receiver operating and precision-recall curves.
- **Balanced Accuracy:** $\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$.
- **Matthews Correlation Coefficient (MCC):** Balanced correlation metric for binary classifications.
- **Subject-Macro Metrics:** Explicitly reported across subjects with valid bi-class labels.

### 4.3 Statistical Interpretation & Non-Confirmatory Rule
Because the 10 cross-validation seeds reuse the same 42 subjects across randomized subject-wise groupings, observations across seeds are correlated. Therefore:
- Standard errors and standard deviations across the 10 seeds are reported strictly as **descriptive partition variability**, not as proof of independent hypothesis confirmation.
- Any observed differences are stated objectively as "numerically higher/lower" rather than claiming statistical significance without appropriate clustered multi-level modeling.

---

## 5. Audit Conclusion

All 5 Phase 3B experiments satisfy the highest standards of scientific reproducibility and data leakage prevention. All pipelines are fully automated and verified against the UCI Parkinson's benchmark.


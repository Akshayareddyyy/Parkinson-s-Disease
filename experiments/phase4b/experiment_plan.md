# Phase 4B: Scientifically Valid Predictive Performance Improvement — Master Plan

**Project:** Parkinson's Disease Telemonitoring — Motor Progression & Severity Prediction  
**Dataset:** UCI Parkinson's Telemonitoring Dataset (5,875 recordings, 42 unique subjects)  
**Execution Environment:** Windows 11, Python 3.11.7, Scikit-learn, XGBoost, CatBoost, SciPy, NumPy, Pandas, Matplotlib  
**Audit Status:** APPROVED WITH METHODOLOGICAL CORRECTIONS (Pre-Execution Phase)  
**Governing Rule:** Scientific Validity & Strict Parity Supersede Forcing Higher Accuracy

---

## 1. Executive Summary & Existing Reference Benchmarks

Phase 4B explores scientifically rigorous avenues for improving predictive accuracy across centralized machine learning, federated learning, feature representations, and personalized longitudinal monitoring.

### 1.1 Existing Audited Reference Benchmarks
All comparisons will be benchmarked directly against the audited results from Phase 3, Phase 3B, and Phase 4:
1. **Centralized Binary Classification:** Extra Trees (Phase 3B Exp 1 Ordinary):
   - **Test ROC-AUC:** `0.6384` $\pm$ `0.1053`
   - **Test Balanced Accuracy:** `0.6155` $\pm$ `0.0896`
   - **Test MCC:** `0.2203` $\pm$ `0.1616`
2. **Federated Binary Classification:** FedAvg-LR ($K=26$ Edge Clients, Phase 4):
   - **Test ROC-AUC:** `0.6289` $\pm$ `0.1241`
   - **Test Balanced Accuracy:** `0.5974` $\pm$ `0.0986`
   - **Test MCC:** `0.1847` $\pm$ `0.1930`
3. **Centralized Continuous Regression:** ElasticNet (Phase 3B Exp 2):
   - **Test MAE:** `6.9635` $\pm$ `0.9438` points
   - **Test RMSE:** `7.9388` $\pm$ `0.9522` points
   - **Pearson $r$:** `0.2586`
   - *Baselines:* Mean Dummy Baseline MAE = `6.9700`, Test_Time Trend MAE = `6.9956`.
4. **Federated Continuous Regression:** FedAvg-Ridge ($K=26$, Phase 4):
   - **Test MAE:** `6.9040` $\pm$ `0.8675` points
   - **Test RMSE:** `7.8497` $\pm$ `0.9459` points
   - **Pearson $r$:** `0.2878`
5. **Personalized Delta Prediction ($\Delta y = y_t - y_{\text{base}}$):** Random Forest ($k=10$, Phase 3B Exp 5):
   - **Test MAE:** `2.8039` $\pm$ `0.5105` points
   - **Test RMSE:** `3.8307` $\pm$ `0.6941` points
   - Extra Trees ($k=5$): Test MAE = `2.9602` $\pm$ `0.4152` points.

---

## 2. Integrated Methodological Corrections

Per scientific review requirements, the following methodological enhancements are strictly integrated:
1. **Isolated Execution:** All code, data, logs, and figures are strictly confined to `experiments/phase4b/`.
2. **Class-Weighted Federated Objectives:**
   - Single-class clients cannot compute valid local class weights (division by zero). Phase 4B implements:
     a) *Globally-Calculated Class-Weighted FedAvg:* Development-wide class weights ($w_c = N / (2 N_c)$) broadcasted to all clients to safely counter class imbalance.
     b) *Safe Local Class-Weighted FedAvg:* Uses local class weights where both classes are present, falling back to global class weights for single-class clients.
3. **Hyperparameter Selection for FedProx:**
   - FedProx proximal parameter $\mu \in \{0.0, 0.01, 0.1, 1.0, 5.0, 10.0\}$ is treated as a validation-selected hyperparameter evaluated on the 7 server validation subjects ($\mathcal{V}$). FedAvg is included explicitly as the $\mu = 0.0$ baseline reference.
4. **Calibration Metrics & Zero Test Leakage:**
   - Probability calibration (`CalibratedClassifierCV` via Platt sigmoid scaling) is fitted strictly inside development folds.
   - Classification metric suite is expanded to include **Brier Score** (`brier_score_loss`) and **Log Loss** (`log_loss`) to evaluate probability calibration directly.
5. **Stable Acoustic Interactions:**
   - All engineered ratios incorporate safety offsets (`Jitter / (Shimmer + 1e-6)`, `NHR / (HNR + 1e-6)`) to guarantee numerical stability. All transformers are fitted inside training folds.
6. **Strict Chronological Calibration Invariant:**
   - For every patient, recordings are sorted by `test_time`. The first $k$ records form the baseline; future records ($t > k$) are evaluated.
7. **Explicit Record Accounting:**
   - Every logged row explicitly distinguishes $N_{\text{train}}$, $N_{\text{calib}}$, $N_{\text{val}}$, and $N_{\text{test}}$.
8. **Documented Comparable Tuning Budgets:**
   - Hyperparameter grid sizes and evaluation budgets are documented and controlled across all models.

---

## 3. Experimental Architecture

```
Phase 4B Investigation Suite
├── Exp 1: Advanced Centralized Classification (ET, RF, CatBoost, XGBoost, HistGB, RBF-SVM, LR)
│         + Probability Calibration (Platt) + Brier Score & Log Loss Reporting
├── Exp 2: Advanced Continuous Regression (ElasticNet, Ridge, Huber, ET, RF, CatBoost, XGBoost, SVR)
│         + Matched Baseline Comparators (Mean & Test-Time)
├── Exp 3: Acoustic Feature Engineering & Transformations (Stable Interaction Terms, Power Transforms, Fold Selection)
├── Exp 4: Federated Learning Improvement (Global Class-Weighted FedAvg, Cosine Annealing, Validation-Tuned FedProx)
└── Exp 5: Personalized Longitudinal Delta Prediction (Windows k in {3, 5, 10, 15}, Z-score vs Robust Scaling)
```

---

## 4. Detailed Experimental Protocols

### 4.1 Experiment 1: Advanced Centralized Classification
- **Evaluated Algorithms (7 Models):**
  1. Extra Trees Classifier (Grid: `n_estimators` $\in [150, 250]$, `max_depth` $\in [8, \text{None}]$, `class_weight='balanced'`)
  2. Random Forest Classifier (Grid: `n_estimators` $\in [150, 250]$, `max_depth` $\in [6, 8]$, `class_weight='balanced'`)
  3. CatBoost Classifier (Grid: `iterations` $\in [200, 350]$, `depth` $\in [4, 6]$, `auto_class_weights='Balanced'`)
  4. XGBoost Classifier (Grid: `n_estimators` $\in [150, 250]$, `max_depth` $\in [3, 5]$, `learning_rate` $\in [0.03, 0.08]$)
  5. HistGradientBoosting Classifier (Grid: `max_iter` $\in [150, 250]$, `l2_regularization` $\in [0.1, 1.0]$, `class_weight='balanced'`)
  6. RBF Support Vector Classifier (Grid: $C \in [0.5, 2.0]$, $\gamma \in [\text{'scale'}, 0.05]$, `class_weight='balanced'`)
  7. Regularized Logistic Regression (Grid: $C \in [0.01, 0.1, 1.0, 10.0]$, penalty $\in [l_1, l_2]$, `class_weight='balanced'`)
- **Probability Calibration:** Models are wrapped with `CalibratedClassifierCV(method='sigmoid', cv='prefit')` fitted on inner validation folds without test contamination.
- **Metrics Logged:** ROC-AUC, PR-AUC, Balanced Accuracy, Sensitivity, Specificity, F1-macro, MCC, Brier Score, Log Loss, Confusion Matrix.

### 4.2 Experiment 2: Advanced Centralized Regression
- **Evaluated Algorithms (8 Models):**
  1. ElasticNet Regressor ($\alpha \in [0.01, 0.1, 1.0, 10.0]$, $l_1 \in [0.2, 0.5, 0.8]$)
  2. Ridge Regressor ($\alpha \in [0.1, 1.0, 10.0, 50.0]$)
  3. Huber Regressor ($\epsilon \in [1.2, 1.35, 1.75]$, $\alpha \in [0.01, 0.1]$)
  4. Extra Trees Regressor (`n_estimators` $\in [150, 250]$, `min_samples_leaf` $\in [2, 5]$)
  5. Random Forest Regressor (`n_estimators` $\in [150, 250]$, `max_depth` $\in [6, 8]$)
  6. CatBoost Regressor (`iterations` $\in [250, 400]$, `depth` $\in [4, 6]$)
  7. XGBoost Regressor (`n_estimators` $\in [150, 250]$, `max_depth` $\in [3, 5]$)
  8. RBF Support Vector Regression ($C \in [1.0, 5.0]$, $\epsilon \in [0.05, 0.1]$)
- **Baselines:** Mean Dummy Baseline ($\hat{y} = \bar{y}_{\text{train}}$) and Test-Time Trend ($\hat{y} = \beta_0 + \beta_1 \cdot \text{test\_time}$).
- **Metrics Logged:** MAE, RMSE, $R^2$, Pearson $r$, Spearman $\rho$, Per-Subject Mean MAE and SD.

### 4.3 Experiment 3: Acoustic Feature Engineering & Transformations
- **Acoustic Interaction Terms (Numerically Stable):**
  - Jitter-to-Shimmer ratio: `Jitter(%) / (Shimmer + 1e-6)`
  - Harmonic-to-Noise inverse: `NHR / (HNR + 1e-6)`
  - Non-linear entropy interaction: `PPE * RPDE`
  - Periodicity-Entropy interaction: `DFA * PPE`
- **Transformation Pipelines:**
  - RobustScaler (median centering, IQR scaling)
  - Yeo-Johnson PowerTransformer
  - QuantileTransformer (`output_distribution='normal'`, `n_quantiles=500`)
- **Selection Mechanisms:**
  - Correlation Filter ($|r| < 0.85$ strictly on training fold covariance)
  - Mutual Information (Top 60th percentile)
  - L1 Sparsity (Lasso/Logistic)

### 4.4 Experiment 4: Federated Learning Improvement
- **Topology 1 (26 Edge Devices) & Topology 2 (5 Regional Clinics):**
- **Evaluated Optimizers:**
  1. *Standard FedAvg-LR:* Baseline reference ($\mu = 0.0$, constant $\eta = 0.1$).
  2. *Cosine Annealing FedAvg-LR:* Dynamic schedule $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / R))$.
  3. *Globally Class-Weighted FedAvg-LR:* Local gradients weighted by global class prevalence $w_0, w_1$ to resolve single-class client starvation.
  4. *Safe Local Class-Weighted FedAvg-LR:* Local class weights for bi-class clients, falling back to global weights for single-class clients.
  5. *Validation-Tuned FedProx-LR:* $\mu \in \{0.0, 0.01, 0.1, 1.0, 5.0, 10.0\}$ tuned on 7 server validation subjects ($\mathcal{V}$).
  6. *FedAvg-Ridge Regression:* Continuous regression with cosine annealing schedule.
- **Matched Centralized Controls:** Evaluated on the identical data splits.

### 4.5 Experiment 5: Personalized Longitudinal Delta Prediction
- **Calibration Windows Evaluated:** $k \in \{3, 5, 10, 15\}$ initial chronological recordings.
- **Patient-Specific Normalization Schemes:**
  1. Patient Z-score: $z = (x - \mu_{s, k}) / \sigma_{s, k}$
  2. Patient Robust Median/IQR: $z = (x - \text{median}_{s, k}) / \max(\text{IQR}_{s, k}, 1e-6)$
  3. Patient Min-Max: $z = (x - x_{\min, s, k}) / \max(x_{\max, s, k} - x_{\min, s, k}, 1e-6)$
- **Models:** Extra Trees, Random Forest, Ridge, ElasticNet.
- **Strict Accounting:** Separate logging of $N_{\text{calib}}$ and $N_{\text{test, eval}}$.

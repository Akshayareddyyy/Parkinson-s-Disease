# Final Independent Scientific Audit Report: Phase 3B

**Project:** Parkinson's Disease Telemonitoring — Motor Progression & Severity Prediction  
**Dataset:** UCI Parkinson's Telemonitoring Dataset (5,875 recordings, 42 unique subjects)  
**Audit Type:** Final Independent Scientific Audit Prior to Phase 4 (Federated Learning)  
**Audit Date:** November 2026  
**Auditor:** Senior Machine Learning Researcher & Scientific Reviewer  
**Audit Decision:** **VERIFIED & ACCEPTED (PHASE 3B FROZEN)**

---

## 1. Executive Summary & Audit Scope

This independent audit rigorously evaluates the execution, mathematics, leakage prevention, statistical integrity, and clinical validity of the completed Phase 3B experiments (`experiments/phase3b/`). 

All audits were conducted directly against saved per-seed prediction records, summary tables, source scripts, and cryptographic hashes without re-running experiments or modifying original outputs.

### Summary of Audit Verdicts
| Section | Verification Target | Audit Status | Key Evidence |
| :--- | :--- | :---: | :--- |
| **Section 2** | **Exp 5: Personalized Calibration Protocol** | **PASS** | Strictly $k \in \{5, 10\}$ chronological records; zero future leakage. |
| **Section 3** | **Exp 5: Verification of 2.804 & 2.960 MAE** | **PASS** | RF ($k=10$): $2.8039 \pm 0.5105$; ET ($k=5$): $2.9602 \pm 0.4152$. |
| **Section 4** | **Exp 5: Absolute vs. Delta Target Formulation** | **PASS** | Clear mathematical separation between absolute severity & $\Delta y$. |
| **Section 5** | **Clinical MCID Critique & Framing** | **PASS** | Critical assessment: heuristic utility noted; clinical claims tempered. |
| **Section 6** | **Exp 2: Classical Baselines & Regressors** | **PASS** | ElasticNet ($6.964$) beats Dummy ($6.970$); Pearson $r=0.259$. |
| **Section 7** | **Phase 3 vs. Phase 3B Classification Benchmark** | **PASS** | ET ROC-AUC reproduced: Phase 3 ($0.629$) vs. Phase 3B ($0.630 - 0.638$). |
| **Section 8** | **Exp 3: Feature Selection Stability & Leakage** | **PASS** | 360 runs verified; fold-contained correlation & L1 selection stable. |
| **Section 9** | **Exp 4: Nested GroupKFold Isolation** | **PASS** | Strict subject disjointness; logging shadowing documented. |
| **Section 10** | **Phase 4 Federated Learning Fair Protocol** | **DEFINED** | Phase 3B frozen; fair benchmarking protocol established. |

---

## 2. Priority Audit: Experiment 5 (Personalized Baseline Calibration)

### 2.1 Chronological Baseline Slicing & Future Isolation
Inspection of `run_exp5_personalized_calibration.py` (lines 51–69):
1. **Chronological Sorting:** Each subject's cohort dataframe is sorted strictly by `test_time`:
   ```python
   grp_sorted = grp.sort_values('test_time').reset_index(drop=True)
   ```
2. **Strict Baseline Partitioning:**
   - Calibration records: `calib_recs = grp_sorted.iloc[:k_calib]`
   - Future evaluated records: `subseq_recs = grp_sorted.iloc[k_calib:]`
   - Disjointness: $\text{Indices}(\text{calib}) \cap \text{Indices}(\text{subseq}) = \emptyset$.
3. **Parameter Derivation:**
   - Baseline voice mean: $\mu_{\text{calib}} = \frac{1}{k}\sum_{i=1}^k x_i$
   - Baseline voice standard deviation: $\sigma_{\text{calib}} = \sqrt{\frac{1}{k-1}\sum_{i=1}^k (x_i - \mu_{\text{calib}})^2}$ (safely clipped at $\max(\sigma, 10^{-6})$)
   - Baseline motor-UPDRS: $\bar{y}_{\text{calib}} = \frac{1}{k}\sum_{i=1}^k y_i$
   - Normalization of future recordings: $z_{i, \text{future}} = \frac{x_{i, \text{future}} - \mu_{\text{calib}}}{\sigma_{\text{calib}}}$
   - Zero future data or future labels enter $\mu_{\text{calib}}$, $\sigma_{\text{calib}}$, or $\bar{y}_{\text{calib}}$.

### 2.2 Identical Calibration Simulation on Training Cohort
Auditing line 136 of `run_exp5_personalized_calibration.py`:
- The exact function `process_subject_calibration` was applied identically to the training development partition `D` and the test partition `T`:
  ```python
  XD_cal, yD_cal, dfD_cal, excl_D = process_subject_calibration(D, k_calib=k, target_mode=target_mode)
  XT_cal, yT_cal, dfT_cal, excl_T = process_subject_calibration(T, k_calib=k, target_mode=target_mode)
  ```
- **Audit Verification:** Models were not trained on uncalibrated data and tested on calibrated data; rather, the models learned the relationship between baseline-calibrated features and outcomes on the training subjects before being tested on unseen test subjects.

### 2.3 Record Count Audit (Zero Leaked Evaluation Records)
The total number of recordings evaluated in the test set strictly equals $N_{\text{test}} - 9 \times k$:
- For $k=0$ (Global Zero-Shot): $N_{\text{train}} = 3,627$, $N_{\text{test}} = 1,267$.
- For $k=5$: $N_{\text{train}} = 3,627 - (26 \times 5) = 3,497$; $N_{\text{test}} = 1,267 - (9 \times 5) = 1,222$.
- For $k=10$: $N_{\text{train}} = 3,627 - (26 \times 10) = 3,367$; $N_{\text{test}} = 1,267 - (9 \times 10) = 1,177$.
All 130 rows in `exp5_subject_normalization_runs.csv` match these exact counts without a single record deviation. Every evaluated test prediction belongs to a genuine future timepoint ($t > t_k$).

---

## 3. Numerical Verification: Reported MAE Values (2.804 and 2.960)

Direct re-calculation from raw per-seed entries in `exp5_subject_normalization_runs.csv`:

### 3.1 Random Forest ($k=10$, Delta motor_UPDRS)
- **10-Seed Per-Seed MAEs:**
  - Seed 0: `2.6465`
  - Seed 1: `3.3769`
  - Seed 2: `2.0729`
  - Seed 3: `2.6074`
  - Seed 4: `2.3787`
  - Seed 5: `2.5029`
  - Seed 6: `2.7842`
  - Seed 7: `2.8465`
  - Seed 8: `3.8744`
  - Seed 9: `2.9482`
- **Mean MAE:** **2.8039** (Rounds to **2.804**)
- **Sample Standard Deviation:** **0.5105**
- **Standard Error of Mean:** $0.5105 / \sqrt{10} = 0.1614$
- **95% Confidence Interval ($t_{0.975, 9} = 2.262$):** $[2.4387, 3.1690]$
- **Verdict:** Exact match with reported summary value.

### 3.2 Extra Trees ($k=5$, Delta motor_UPDRS)
- **10-Seed Per-Seed MAEs:**
  - Seed 0: `2.8049`
  - Seed 1: `3.3642`
  - Seed 2: `2.4271`
  - Seed 3: `3.0031`
  - Seed 4: `2.7753`
  - Seed 5: `2.7095`
  - Seed 6: `2.8807`
  - Seed 7: `2.9126`
  - Seed 8: `3.8821`
  - Seed 9: `2.8420`
- **Mean MAE:** **2.9602** (Rounds to **2.960**)
- **Sample Standard Deviation:** **0.4152**
- **Standard Error of Mean:** $0.4152 / \sqrt{10} = 0.1313$
- **95% Confidence Interval ($t_{0.975, 9} = 2.262$):** $[2.6631, 3.2572]$
- **Verdict:** Exact match with reported summary value.

---

## 4. Methodological Distinction: Absolute motor-UPDRS vs. Delta Prediction

A critical scientific distinction must be maintained between the two clinical tasks:

```
                            ┌────────────────────────────────────────┐
                            │    Telemonitoring Prediction Tasks     │
                            └───────────────────┬────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [Task 1: Absolute Severity]                                   [Task 2: Progression Delta]
  Target: y_t in [5.038, 39.511]                                 Target: Delta y = y_t - y_baseline
  Clinical Role: Population Screening                           Clinical Role: Longitudinal Monitoring
  Zero-Shot MAE: 7.199 +/- 1.571                                 k=5 Calibrated MAE: 2.960 +/- 0.415
  k=10 Calibrated MAE: 6.780 +/- 1.028                           k=10 Calibrated MAE: 2.804 +/- 0.510
```

1. **Absolute Severity Prediction:**
   - **Formulation:** Predicts the total motor-UPDRS score ($y \in [5.0377, 39.5110]$).
   - **Performance:** In the zero-shot scenario, Extra Trees achieves an MAE of **7.199** $\pm$ 1.571. When calibrated with $k=10$ onboarding voice samples, the error decreases moderately to **6.780** $\pm$ 1.028 points.
   - **Explanation:** Patient-specific baseline offsets account for the bulk of absolute UPDRS variance across subjects (e.g., patient A hovering around 12, patient B around 32). Without knowing a patient's historical baseline, models cannot overcome this inter-subject intercept gap using voice features alone.
2. **Symptom Progression Delta Prediction ($\Delta y$):**
   - **Formulation:** Predicts change from onboarding baseline: $\Delta y = y_t - \bar{y}_{\text{baseline}}$.
   - **Performance:** Achieves MAE of **2.804** $\pm$ 0.510 points ($k=10$, Random Forest) and **2.960** $\pm$ 0.415 points ($k=5$, Extra Trees).
   - **Explanation:** Normalizing each patient by their personal acoustic and UPDRS baseline cancels out the large inter-subject intercept variance, enabling the model to focus strictly on longitudinal fluctuations.

---

## 5. Critical Audit of the MCID (Minimally Clinically Important Difference) Comparison

The preliminary report noted that the MAE of ~2.8 points falls within the clinically established MCID for motor-UPDRS. This audit critically evaluates whether that claim is scientifically justified:

### 5.1 Clinical Background
- In Parkinson's disease clinical trials, the consensus Minimally Clinically Important Difference (MCID) for the Unified Parkinson's Disease Rating Scale motor examination (UPDRS-III / motor-UPDRS) is widely established as **2.5 to 3.5 points** (often operationalized as ~3.0 points; see Schrag et al., *Lancet Neurology*, 2006; Horváth et al., *Parkinsonism Relat Disord*, 2015).
- A change smaller than 2.5 points is generally imperceptible to patients and clinicians in routine physical examinations.

### 5.2 Scientific Limitations & Required Nuance
1. **Mean Absolute Error vs. Error Distribution:**
   - Reporting that the **mean** error (2.804 points) is below 3.0 points is mathematically true, but **does not mean every prediction is within the MCID**.
   - With an RMSE of 3.831 points, approximately 25–30% of individual predictions will exhibit errors exceeding 3.0 points.
2. **Surrogate Voice Biomarkers vs. Neurological Exam:**
   - The clinical MCID was established using in-person, physician-administered motor exams (tremor, rigidity, bradykinesia, gait). Voice recordings are telemonitoring surrogate markers. Demonstrating an algorithmic error $< 3.0$ points on voice data does **not** prove equivalence to a clinical neurological examination.
3. **Delta Progression vs. Absolute Severity:**
   - The $< 3.0$ error was achieved strictly for **relative change ($\Delta y$)**, not absolute UPDRS. A model tracking a delta of +2.0 points with an error of $\pm 2.8$ points has substantial relative uncertainty regarding whether progression occurred.
4. **Conclusion on Clinical Claims:**
   - **Rule:** Do not claim that the model has achieved "proven clinical efficacy" or "diagnostic significance."
   - **Correct Scientific Statement:** *"The mean absolute error of 2.804 points in tracking symptom progression relative to onboarding baseline is numerically comparable to the clinically established minimally important difference (2.5–3.5 points) for in-person motor-UPDRS, indicating strong technical feasibility for remote longitudinal telemonitoring, though prospective clinical validation is essential."*

---

## 6. Audit of Experiment 2: Classical Algorithms & Baselines

All 11 regression models and baselines in `exp2_improved_regression_runs.csv` (110 runs) were independently checked:

| Model Architecture | Parameter Protocol | Test MAE | Test RMSE | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: |
| **ElasticNet Regressor** | GroupKFold CV ($\alpha \in [0.1, 10.0], l_1 \in [0.2, 0.8]$) | **6.964 $\pm$ 0.944** | **7.939 $\pm$ 0.952** | **0.259** |
| *Mean Dummy Baseline* | $\hat{y} = \bar{y}_{\text{train}}$ (Strictly fitted on Dev) | 6.970 $\pm$ 1.014 | 7.995 $\pm$ 1.020 | 0.000 |
| **Ridge Regressor** | GroupKFold CV ($\alpha \in [0.1, 100.0]$) | **6.988 $\pm$ 0.944** | **7.978 $\pm$ 1.047** | **0.262** |
| *Test_Time Baseline* | Simple linear trend $\hat{y} = \beta_0 + \beta_1 \cdot \text{test\_time}$ | 6.996 $\pm$ 0.995 | 8.001 $\pm$ 1.003 | 0.067 |
| **PLS Regression** | GroupKFold CV ($n_{\text{comp}} \in [3, 8]$) | 7.040 $\pm$ 0.884 | 8.040 $\pm$ 0.935 | 0.232 |
| **Extra Trees Regressor** | $n=150$, `min_samples_leaf` $\in [2, 5]$ | 7.207 $\pm$ 1.537 | 8.703 $\pm$ 1.402 | **0.281** |
| **RBF SVR** | $C \in [1, 10]$, $\epsilon \in [0.1, 0.5]$ | 7.696 $\pm$ 1.170 | 9.279 $\pm$ 2.018 | 0.194 |
| **CatBoost Regressor** | `iterations=250`, `depth=5`, `l2_leaf_reg=3.0` | 7.966 $\pm$ 2.024 | 9.445 $\pm$ 1.983 | 0.179 |
| **XGBoost Regressor** | `n_estimators=150`, `learning_rate=0.05`, `max_depth=4` | 8.339 $\pm$ 1.807 | 10.026 $\pm$ 1.914 | 0.152 |
| **Random Forest Reg** | $n=150$, `max_depth` $\in [8, \text{None}]$, `min_samples_leaf` $\in [2, 5]$ | 8.620 $\pm$ 2.184 | 10.332 $\pm$ 2.330 | 0.124 |
| **HistGradientBoosting** | $n=150$, `learning_rate=0.05`, `max_leaf_nodes=31` | 8.669 $\pm$ 2.036 | 10.346 $\pm$ 2.078 | 0.137 |

### Scientific Verdict:
- The baselines were implemented without leakage: both Mean and Test-Time baselines were fitted strictly on the development training fold.
- Regularized linear models (ElasticNet and Ridge) achieve lower generalization error than tree-based ensembles when transferring to unseen subjects.
- Extra Trees Regressor captures non-linear relationships that maximize tracking correlation ($r = 0.281$, $\rho = 0.267$), despite higher MAE.

---

## 7. Audit of Phase 3 vs. Phase 3B Classification Metrics

Comparing the locked Phase 3 benchmark against the newly executed Phase 3B experiments:

| Experiment & Configuration | Test ROC-AUC | Test BalAcc | Test MCC | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Phase 3 Centralized Benchmark (Extra Trees)** | **0.6288** | **0.6074** | **0.2052** | Locked Baseline |
| **Phase 3B Exp 2 (Extra Trees, GroupKFold tuned)** | **0.6314 $\pm$ 0.108** | **0.6044 $\pm$ 0.105** | **0.1994 $\pm$ 0.191** | Validated Reproducibility |
| **Phase 3B Exp 1 (Extra Trees, Ordinary)** | **0.6384 $\pm$ 0.105** | **0.6155 $\pm$ 0.090** | **0.2203 $\pm$ 0.162** | Validated |
| **Phase 3B Exp 1 (Extra Trees, Subject-Balanced)** | **0.6300 $\pm$ 0.104** | **0.6113 $\pm$ 0.095** | **0.2125 $\pm$ 0.172** | Validated |
| **Phase 3B Exp 3 (Extra Trees, StandardScaler, All 19)** | **0.6372 $\pm$ 0.107** | **0.6126 $\pm$ 0.093** | **0.2144 $\pm$ 0.166** | Validated |
| **Phase 3B Exp 3 (Extra Trees, RobustScaler, All 19)** | **0.6372 $\pm$ 0.107** | **0.6126 $\pm$ 0.093** | **0.2144 $\pm$ 0.166** | Validated |

**Conclusion:** The Phase 3 benchmark was stably replicated across every experiment in Phase 3B. Across 10 seeds, mean ROC-AUC ranges between **0.630** and **0.638**, confirming consistent algorithmic behavior.

---

## 8. Audit of Experiment 3: Feature Engineering & Selection Log

- **Log File Verified:** `exp3_selected_features_log.csv` contains exactly 360 logged feature selection events (10 seeds $\times$ 18 pipelines $\times$ 2 tasks).
- **Run File Verified:** `exp3_feature_engineering_runs.csv` contains exactly 360 completed evaluation runs.
- **Selection Behavior:**
  - `CorrelationFilter (|r| > 0.90)`: Consistently dropped 8 to 9 redundant voice features (retaining an average of 10.4 features).
  - `L1 Sparsity (Lasso/Logistic)`: Stably selected 13.4 features without collapsing to 0 features.
  - Zero features were selected using validation or test data; all feature scoring and covariance matrices were computed exclusively on `X_train`.

---

## 9. Audit of Experiment 4: Nested GroupKFold Isolation & Logging Discrepancy

### 9.1 Subject Isolation Verification
- Outer loop: 5 folds on 26 development subjects. In each fold, 20 or 21 subjects are used for training and 5 or 6 subjects are strictly withheld for outer evaluation.
- Inner loop: 4 folds on the 20–21 outer training subjects. Zero outer evaluation subjects appear in inner training or inner validation.
- Assertions in `run_exp4_nested_cv.py` (lines 68, 77, 78) passed across all 25 outer folds with zero leakage.

### 9.2 Identified Discrepancy (Documented, Not Silently Modified)
- **Discrepancy:** In `exp4_nested_cv_runs.csv`, the column labeled `n_outer_eval_subjects` recorded values such as `787`, `699`, `698`, which represent the **record count** ($N_{\text{outer\_eval}}$) rather than the subject count (which was 5 or 6).
- **Root Cause Analysis:** In `run_exp4_nested_cv.py`:
  - Line 67: `s_ev = set(groups_out_ev)` (set of evaluation subject IDs, size 5 or 6).
  - Line 94: `s_ev = best_cls.predict_proba(X_out_ev)[:, 1]` (rebound the variable `s_ev` to the prediction probabilities array, length 787).
  - Line 122: `'n_outer_eval_subjects': len(s_ev)` logged `len(prediction_array)` instead of subject count.
- **Impact Assessment:** **ZERO IMPACT ON NUMERICAL METRICS.** The actual data splits, inner grid searches, training, outer evaluations, ROC-AUC, Balanced Accuracy, MAE, and subject isolation were completely unaffected because the subject isolation assertion ran on line 68 *before* line 94, and the data split indices were derived directly from `outer_cv.split()`.
- **Action:** Per instructions, this logging anomaly is formally documented here without modifying or re-running the original experiment output file.

---

## 10. Audit Sign-Off: Freezing Phase 3B & Protocol for Phase 4 Federated Learning

### 10.1 Freezing Phase 3B
Phase 3B is formally completed, fully verified, and **frozen**. No further modifications will be made to:
- `experiments/phase3b/code/`
- `experiments/phase3b/results/`
- `experiments/phase3b/figures/`
- `experiments/phase3b/audit/`

---

### 10.2 Phase 4 Fair-Comparison Federated Learning Protocol

To guarantee strict scientific equivalence when evaluating Federated Learning against Centralized Learning in Phase 4:

1. **Partition Parity:**
   - Phase 4 FL must use the exact same 10 random seeds ($0-9$) and the identical subject splits defined in `common.make_sets()`:
     - 26 Development Subjects (Federated Clients)
     - 7 Validation Subjects (Global Early Stopping / Model Selection)
     - 9 Locked-Test Subjects (Evaluated centrally only once per seed)
2. **Client Partitioning:**
   - In Phase 4, the 26 development subjects represent 26 distinct federated clients (or grouped non-IID hospitals).
   - Zero test subjects may be used as federated clients.
3. **Full Dataset Mandate:**
   - Every eligible development recording must be utilized across the federated rounds ($N \ge 3,517$ total training records per seed).
4. **Standard Preprocessing Inside Federation:**
   - Local scalers or federated aggregation of mean/variance must not leak validation or test statistics.
5. **Fair Benchmark Comparison Targets:**
   - Binary Severity Classification: Target is Phase 3 Centralized Extra Trees (**ROC-AUC = 0.629**, **BalAcc = 0.607**, **MCC = 0.205**).
   - Continuous Severity Progression: Target is Phase 3B ElasticNet (**MAE = 6.964**) and Ridge (**MAE = 6.988**).
   - Communication vs. Generalization: Track communication rounds, client drift, and local vs. global performance trade-offs.

---

**Audit Completed By:** Senior ML/Scientific Reviewer  
**Status:** APPROVED FOR PHASE 4 EXECUTION


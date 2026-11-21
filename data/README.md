# Oxford Parkinson's Telemonitoring Dataset

## 1. Dataset Overview

This directory contains the telemonitoring dataset used across all experimental phases of the Parkinson's motor severity prediction study.

- **File Name:** `parkinsons_updrs.csv`
- **Original Source:** A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig (2009), *Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests*, IEEE Transactions on Biomedical Engineering. Available at the UCI Machine Learning Repository.
- **Cohort:** 42 patients diagnosed with early-stage Parkinson's disease (recruited within the first five years of clinical onset).
- **Study Duration:** Six months of continuous, non-invasive home acoustic telemonitoring.
- **Total Records:** 5,875 biomedical voice samples recorded through standard telephone networks.

---

## 2. Feature Definitions & Taxonomy

The dataset contains 22 columns: 4 patient/session metadata fields, 2 clinical targets, and 16 non-invasive biomedical acoustic voice measures.

| Column | Type | Unit / Range | Clinical Meaning / Acoustic Definition |
| :--- | :--- | :--- | :--- |
| `subject#` | Integer | [1, 42] | Unique patient identifier. |
| `age` | Integer | [36, 85] | Subject age at study enrollment (years). |
| `sex` | Binary | {0: Male, 1: Female} | Biological sex of subject. |
| `test_time` | Float | [-4.26, 215.49] | Time elapsed from baseline clinic visit (fractional days). |
| `motor_UPDRS` | Float | [5.04, 39.51] | Clinician-scored motor Unified Parkinson's Disease Rating Scale. **Primary Target**. |
| `total_UPDRS` | Float | [7.00, 54.99] | Total UPDRS score (motor + cognitive/mood + ADL). **Excluded as target/feature to prevent leakage.** |
| `Jitter(%)` | Float | [0.0008, 0.0999] | Percentage of cycle-to-cycle variation in fundamental frequency ($F_0$). |
| `Jitter(Abs)` | Float | [0.000002, 0.00045] | Absolute cycle-to-cycle variation in fundamental frequency (seconds). |
| `Jitter:RAP` | Float | [0.0003, 0.057] | Relative Amplitude Perturbation ($F_0$ variability across 3 cycles). |
| `Jitter:PPQ5` | Float | [0.0003, 0.049] | 5-point Period Perturbation Quotient ($F_0$ variability across 5 cycles). |
| `Jitter:DDP` | Float | [0.001, 0.17] | Average absolute difference of differences between consecutive cycles ($3 \times \text{RAP}$). |
| `Shimmer` | Float | [0.003, 0.26] | Local shimmer (relative amplitude perturbation between adjacent sound cycles). |
| `Shimmer(dB)` | Float | [0.026, 2.11] | Shimmer expressed on a logarithmic decibel scale. |
| `Shimmer:APQ3` | Float | [0.0016, 0.16] | 3-point Amplitude Perturbation Quotient. |
| `Shimmer:APQ5` | Float | [0.0019, 0.17] | 5-point Amplitude Perturbation Quotient. |
| `Shimmer:APQ11` | Float | [0.0027, 0.27] | 11-point Amplitude Perturbation Quotient. |
| `Shimmer:DDA` | Float | [0.0048, 0.49] | Average absolute difference between consecutive cycle amplitudes ($3 \times \text{APQ3}$). |
| `NHR` | Float | [0.00028, 0.748] | Noise-to-Harmonics Ratio (incoherent spectral noise vs. periodic energy). |
| `HNR` | Float | [1.65, 37.88] | Harmonics-to-Noise Ratio (dB) (ratio of periodic signal energy to noise). |
| `RPDE` | Float | [0.15, 0.97] | Recurrence Period Density Entropy (measure of vocal fold vibration dynamical complexity). |
| `DFA` | Float | [0.51, 0.87] | Detrended Fluctuation Analysis (long-range statistical persistence of acoustic signal). |
| `PPE` | Float | [0.02, 0.73] | Pitch Period Entropy (impairment in voluntary control of pitch). |

---

## 3. Preprocessing, Leakage Prevention, & Exclusions

### A. Data Leakage Prevention
1. **Target Leakage (`total_UPDRS`):** `total_UPDRS` is strictly excluded from all feature sets (`FEATS_19` and `FEATS_9`). `motor_UPDRS` constitutes the primary clinical target.
2. **Identifier & Longitudinal Leakage:** `subject#` and `test_time` are excluded from model input tensors during classification and regression, avoiding memorization of individual subject baselines.
3. **Partition-Contained Scaling:** All feature scalers (`StandardScaler`, `MinMaxScaler`) are fitted **strictly on the training development partition (`X_dev`)** and applied out-of-fold to validation (`X_val`) and test sets (`X_test`). Under no circumstances are global dataset statistics computed across test sets.

### B. Feature Subsets
- **19-Feature Full Classical Set (`FEATS_19`):** Includes all 16 acoustic features plus patient demographics (`age`, `sex`, `test_time` when modeling longitudinal drift in Phase 3/4).
- **9-Feature Quantum-Matched Manifold (`FEATS_9`):**
  Due to 9-qubit hardware constraints in Phase 5, features were pruned based on clinical decorrelation:
  `['age', 'sex', 'test_time', 'Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']`.
  Here, redundant variants (e.g., APQ3/5/11, RAP, PPQ5, DDP, DDA) were dropped to eliminate near-collinear variables ($r > 0.95$).

---

## 4. Subject-Disjoint Partitioning Protocol

To avoid patient leakage where samples from the same individual appear in both training and evaluation folds, a strict **Subject-Disjoint Split** is enforced across 10 deterministic random seeds (seeds 0 through 9):

- **Total Cohort:** 42 unique subjects.
- **Development Partition (`D`):** Exactly 26 subjects (~61.9% of cohort, ~3,640 records). Used exclusively for training and inner cross-validation.
- **Validation Partition (`V`):** Exactly 8 subjects (~19.0% of cohort, ~1,120 records). Used for checkpoint selection and hyperparameter tuning.
- **Test Partition (`T`):** Exactly 8 subjects (~19.0% of cohort, ~1,115 records). Held completely locked until final evaluation.

### Binary Classification Threshold Definition
To prevent target threshold leakage, the threshold separating "mild" from "moderate/severe" motor impairment is calculated as the **median `motor_UPDRS` of the development partition (`D`) strictly within each seed fold**:
$$\tau_{\text{seed}} = \text{Median}\left(\{y_i \mid i \in \mathcal{D}_{\text{seed}}\}\right) \approx 21.016$$
Samples with $y_i \ge \tau$ are assigned label $1$, and $y_i < \tau$ assigned label $0$.

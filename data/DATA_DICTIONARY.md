# Biomedical Acoustic Data Dictionary

This document provides the definitive data dictionary for the **Oxford Parkinson's Telemonitoring Dataset** (`data/parkinsons_updrs.csv`), recording all 22 attributes, their verified physical units, observed numerical distributions across the 5,875 samples, and their specific experimental roles across the study.

---

## 1. Column Specifications & Attribute Taxonomy

| Column Name | Data Type | Physical / Measurement Unit | Observed Range `[Min, Max]` | Observed Distribution `Mean ± SD` | Experimental Role | Verified Acoustic / Clinical Definition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `subject#` | `int64` | Identifier | `[1, 42]` | $21.49 \pm 12.37$ | Grouping ID | Unique integer assigned to each of the 42 individuals with early-stage Parkinson's disease. **Excluded from feature matrices to prevent subject memorization.** |
| `age` | `int64` | Years | `[36, 85]` | $64.80 \pm 8.82$ | Predictor | Patient chronological age at study enrollment. Included in 19-feature set and 9-feature quantum manifold. |
| `sex` | `int64` | Binary code | `{0, 1}` | $0.32 \pm 0.47$ (31.8% Female) | Predictor | Biological sex of subject (`0`: Male, `1`: Female). Mapped to orthogonal quantum angles $\{0.0, \pi\}$ in Phase 5. |
| `test_time` | `float64` | Fractional Days | `[-4.26, 215.49]` | $92.86 \pm 53.45$ | Predictor / Temporal | Time elapsed since baseline in-clinic examination. Negative values indicate baseline pre-trial recordings. |
| `motor_UPDRS` | `float64` | Clinical Score | `[5.04, 39.51]` | $21.30 \pm 8.13$ | **Primary Target** | Clinician-scored motor UPDRS (Part III), assessing speech, facial expression, tremor, rigidity, and gait. Binarized at dev median for classification. |
| `total_UPDRS` | `float64` | Clinical Score | `[7.00, 54.99]` | $29.02 \pm 10.70$ | **Excluded** | Comprehensive UPDRS score (Parts I, II, and III). **Strictly excluded from all feature sets to prevent trivial target leakage ($r \approx 0.947$).** |
| `Jitter(%)` | `float64` | Percentage | `[0.0008, 0.1000]` | $0.0062 \pm 0.0056$ | Predictor | Cycle-to-cycle perturbation in fundamental frequency ($F_0$), expressed as a fraction of the average period. Included in 9-feature manifold. |
| `Jitter(Abs)` | `float64` | Seconds | `[0.0000, 0.0004]` | $0.00004 \pm 0.00004$ | Predictor (19-feat) | Absolute cycle-to-cycle perturbation in fundamental frequency in microseconds/seconds. |
| `Jitter:RAP` | `float64` | Dimensionless ratio | `[0.0003, 0.0575]` | $0.0030 \pm 0.0031$ | Predictor (19-feat) | Relative Amplitude Perturbation: average absolute difference between a period and the average of it and its two neighbors. |
| `Jitter:PPQ5` | `float64` | Dimensionless ratio | `[0.0004, 0.0696]` | $0.0033 \pm 0.0037$ | Predictor (19-feat) | Five-point Period Perturbation Quotient: average absolute difference between a period and the average of it and its four nearest neighbors. |
| `Jitter:DDP` | `float64` | Dimensionless ratio | `[0.0010, 0.1726]` | $0.0090 \pm 0.0094$ | Predictor (19-feat) | Average absolute difference of differences between consecutive cycles ($3 \times \text{RAP}$). |
| `Shimmer` | `float64` | Dimensionless ratio | `[0.0031, 0.2686]` | $0.0340 \pm 0.0258$ | Predictor | Local cycle-to-cycle amplitude perturbation between adjacent sound waveforms. Included in 9-feature manifold. |
| `Shimmer(dB)` | `float64` | Decibels (dB) | `[0.0260, 2.1070]` | $0.3110 \pm 0.2303$ | Predictor (19-feat) | Local shimmer expressed on a base-10 logarithmic decibel scale: $20 \log_{10}(A_{i+1} / A_i)$. |
| `Shimmer:APQ3` | `float64` | Dimensionless ratio | `[0.0016, 0.1627]` | $0.0172 \pm 0.0132$ | Predictor (19-feat) | Three-point Amplitude Perturbation Quotient: relative amplitude variability across a 3-cycle sliding window. |
| `Shimmer:APQ5` | `float64` | Dimensionless ratio | `[0.0019, 0.1670]` | $0.0201 \pm 0.0167$ | Predictor (19-feat) | Five-point Amplitude Perturbation Quotient: relative amplitude variability across a 5-cycle sliding window. |
| `Shimmer:APQ11` | `float64` | Dimensionless ratio | `[0.0025, 0.2755]` | $0.0275 \pm 0.0200$ | Predictor (19-feat) | Eleven-point Amplitude Perturbation Quotient: relative amplitude variability across an 11-cycle sliding window. |
| `Shimmer:DDA` | `float64` | Dimensionless ratio | `[0.0048, 0.4880]` | $0.0515 \pm 0.0397$ | Predictor (19-feat) | Average absolute difference between consecutive cycle amplitudes ($3 \times \text{APQ3}$). |
| `NHR` | `float64` | Dimensionless ratio | `[0.0003, 0.7483]` | $0.0321 \pm 0.0597$ | Predictor | Noise-to-Harmonics Ratio: ratio of non-periodic inharmonic acoustic noise energy to periodic harmonic signal energy. Included in 9-feature manifold. |
| `HNR` | `float64` | Decibels (dB) | `[1.66, 37.88]` | $21.68 \pm 4.29$ | Predictor | Harmonics-to-Noise Ratio: ratio of periodic signal energy to noise energy expressed in decibels. Included in 9-feature manifold. |
| `RPDE` | `float64` | Dimensionless $[0, 1]$ | `[0.1510, 0.9661]` | $0.5415 \pm 0.1010$ | Predictor | Recurrence Period Density Entropy: nonlinear measure of vocal fold vibration dynamical complexity and periodicity deviation. Included in 9-feature manifold. |
| `DFA` | `float64` | Dimensionless scaling exponent | `[0.5140, 0.8656]` | $0.6532 \pm 0.0709$ | Predictor | Detrended Fluctuation Analysis: stochastic long-range self-similarity scaling exponent characterizing vocal acoustic turbulent noise. Included in 9-feature manifold. |
| `PPE` | `float64` | Dimensionless entropy | `[0.0220, 0.7317]` | $0.2196 \pm 0.0915$ | Predictor (19-feat) | Pitch Period Entropy: nonlinear measure of dysphonia quantifying pitch instability and voluntary fundamental frequency control impairment. |

---

## 2. Experimental Feature Subsets

```
All 22 Columns in parkinsons_updrs.csv
├── Identifiers & Targets (3 Columns):
│   ├── subject#        --> Used strictly for subject-wise partitioning and edge client allocation.
│   ├── motor_UPDRS     --> Primary continuous regression and binarized classification target.
│   └── total_UPDRS     --> EXCLUDED to prevent trivial target leakage (r = 0.947).
└── Predictors (19 Features - FEATS_19):
    ├── Demographics & Timing: age, sex, test_time
    ├── 5 Jitter Features:   Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP
    ├── 6 Shimmer Features:  Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA
    ├── 2 Harmonic Ratios:   NHR, HNR
    └── 3 Nonlinear Measures: RPDE, DFA, PPE
```

### The 9-Feature Manifold (`FEATS_9`)
To match 9-qubit hardware register limits in Phase 5 without arbitrary dimensionality reduction, feature selection pruned highly collinear acoustic variants ($r > 0.95$) to preserve distinct physical acoustic dimensions:
1. `age` (Demographic)
2. `sex` (Demographic)
3. `test_time` (Longitudinal timing)
4. `Jitter(%)` (Frequency perturbation)
5. `Shimmer` (Amplitude perturbation)
6. `NHR` (Incoherent noise ratio)
7. `HNR` (Harmonic signal clarity)
8. `RPDE` (Dynamical complexity entropy)
9. `DFA` (Long-range turbulence persistence)

---

## 3. Data Leakage Controls & Clinical Caveats

1. **Target Contamination:** `total_UPDRS` is never included in training matrices. Because it strongly correlates with `motor_UPDRS`, including it would create trivial predictive accuracy that does not generalize to clinical telemonitoring.
2. **Subject Independence:** In-clinic telemonitoring models must generalize to unobserved patients. Splitting records randomly across rows leads to data leakage where the model memorizes individual patient vocal pitch baselines. All evaluation enforces subject-disjoint splits across the 42 subjects.
3. **Statistical vs. Clinical Validity:** Statistical checks (e.g. range and completeness validation) confirm the mechanical integrity of the CSV file. They do not substitute for clinical diagnostic validation.

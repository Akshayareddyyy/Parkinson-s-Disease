# Master Research Documentation & Scientific Audit Trail

This directory provides the definitive scientific audits, forensic discrepancy resolutions, and manuscript readiness guidelines for the Parkinson's telemonitoring study across all research phases (Phase 3 through Phase 5).

---

## 1. Document Catalog

### A. Core Audit Documents (This Directory)

1. [final_research_process_and_results_audit.md](file:///c:/Users/akank/Downloads/parkinsons_severity/research_documentation/final_research_process_and_results_audit.md)
   - **Scope:** Complete end-to-end reconstruction of the entire research lifecycle across Phase 3, Phase 3B, Phase 4, Phase 4B, and Phase 5.
   - **Contents:**
     - Chronological narrative and rationale behind every experimental phase.
     - Verification of dataset dimensions (5,875 samples, 42 subjects), zero-leakage subject-disjoint partitions, and feature pruning.
     - Comprehensive metric extraction tables (ROC-AUC, BACC, MCC, Macro-F1, MAE, RMSE, Pearson $r$, Spearman $\rho$, R²).
     - Classification of all results into Primary, Supporting, Historical, and Non-Comparable categories.
     - Forensic discrepancy logs reconciling differences between legacy summaries and raw CSV logs.

2. [final_results_discrepancy_resolution.md](file:///c:/Users/akank/Downloads/parkinsons_severity/research_documentation/final_results_discrepancy_resolution.md)
   - **Scope:** Deep-dive forensic analysis resolving 6 specific numerical and reporting discrepancies identified across historical reports.
   - **Key Resolutions:**
     - **Discrepancy 1 (Safe Local FedAvg AUC SD):** Reconciles $\pm 0.1289$ (from unweighted standard deviation across 26 heterogeneous edge clients) vs. $\pm 0.1118$ (authoritative standard deviation across the 10 outer seeds). Authoritative manuscript value: **$0.6414 \pm 0.1118$**.
     - **Discrepancy 2 (Centralized Logistic Regression AUC SD):** Reconciles $\pm 0.1066$ (9-feature control baseline) vs. $\pm 0.0811$ (19-feature full model). Authoritative manuscript value: **$0.6277 \pm 0.0811$**.
     - **Discrepancy 3 (FedAvg 50% Client Participation AUC):** Reconciles $0.6175$ (historical Phase 4 preliminary run) vs. $0.6287$ (Phase 4B matched benchmark). Authoritative manuscript value: **$0.6287 \pm 0.1219$**.
     - **Discrepancy 4 & 5 (Personalized ElasticNet & Mean Drift Baseline):** Reconciles $k=15$ calibration window (MAE $2.6965$ vs $2.7077$) vs. primary $k=10$ calibration window (MAE $2.7423$ vs $2.7354$). Recommends presenting the full calibration trajectory ($k \in [3, 15]$) with $k=10$ as primary anchor.
     - **Discrepancy 6 (Pearson Correlation Reporting):** Distinguishes mean per-seed Pearson correlation ($r_{\text{mean}} = 0.2697 \pm 0.1420$) from pooled Pearson correlation ($r_{\text{pooled}} = 0.3124$). Requires explicit labeling to avoid ambiguity.

---

### B. Phase-Specific Verification Documents

- **Phase 5 (Quantum):**
  - [phase5_final_consistency_audit.md](file:///c:/Users/akank/Downloads/parkinsons_severity/experiments/phase5/audit/phase5_final_consistency_audit.md): Complete audit of 9-qubit VQC/PQC-R parameter counts (46 vs 64), depolarizing and phase damping noise channels, and finite-shot measurement behavior.
  - [phase5_manuscript_readiness_audit.md](file:///c:/Users/akank/Downloads/parkinsons_severity/experiments/phase5/audit/phase5_manuscript_readiness_audit.md): Corrects prediction variance ratio to $\sim 3.26\%$, rectifies finite-shot standard error scaling to $\mathcal{O}(1/\sqrt{S})$, distinguishes mean vs median shrinkage, and provides manuscript-safe phrasing.
  - [phase5_execution_report.md](file:///c:/Users/akank/Downloads/parkinsons_severity/experiments/phase5/phase5_execution_report.md): Execution summary of locked-test VQC and PQC-R models across 10 random seeds.
- **Phase 4 & 4B (Federated Learning & Personalization):**
  - [independent_verification_report.md](file:///c:/Users/akank/Downloads/parkinsons_severity/experiments/phase4/independent_verification/independent_verification_report.md): Independent statistical check of FedAvg, FedProx, FedBN, and client participation levels.
  - [phase4b_publication_readiness.md](file:///c:/Users/akank/Downloads/parkinsons_severity/experiments/phase4b/final_verification/phase4b_publication_readiness.md): Verification of personalized regression baselines against non-parametric Mean Drift Dummy.

---

## 2. Statistical & Reporting Standards for Publication

1. **Descriptive Nature of Confidence Intervals:**
   Because all 10 evaluation seeds partition the same underlying cohort of 42 subjects, test sets across seeds are non-independent (each subject appears in approximately 2 test sets). Therefore, all reported $95\%$ confidence intervals, means, and standard deviations must be characterized as **descriptive of partition sensitivity**, rather than inferential generalizations to unobserved clinical populations.

2. **Validation-Selected Primary Ensembles:**
   In Phase 5, the primary reported quantum metrics are the **validation-depth-selected models evaluated across all 10 seeds** (VQC ROC-AUC $0.5401 \pm 0.0381$; PQC-R MAE $6.7471 \pm 0.9634$). Subgroup comparisons (e.g., $L=2$ seeds vs $L=3$ seeds) contain differing seed compositions and must be reported strictly as exploratory depth sensitivity.

3. **Distinction Between Acoustic Modeling and Personalized Baseline Tracking:**
   In personalized regression, models achieve MAE $\sim 2.7$ when provided $k=10$ prior readings per patient. However, comparison with the non-parametric Mean Drift Dummy (MAE $2.7354$) indicates that the reduction from global MAE $\sim 6.9$ is driven almost entirely by patient-specific mean offset subtraction rather than acoustic feature sensitivity. Manuscripts must transparently report both values side by side.

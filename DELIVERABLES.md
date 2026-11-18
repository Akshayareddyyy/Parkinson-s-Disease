# Parkinson's severity study: audit, methodology, reproducibility, reference log

## 1. Critical analysis of the existing work

**Basis and limits of this audit.** The original paper PDF was never received. This audit covers the two notebooks (`flrealqnnmlreal.ipynb`, `vqcparkinsons.ipynb`) and the weaknesses listed in the task brief. Claims made only in the PDF could not be checked.

| Issue | Finding (verified in notebooks unless stated) |
|---|---|
| Target leakage | `motor_UPDRS` is in the feature list in all 21 feature definitions of the federated notebook and in the VQC notebook. `severity` is read from a pre-made split file and the notebooks never create it. The comments say "do NOT derive severity from UPDRS again". It is almost certainly a threshold on a UPDRS score. |
| Near-perfect scores | Federated runs: accuracy about 0.998, ROC-AUC about 0.9997. QNN sweep: rows with accuracy and ROC-AUC exactly 1.000000 on 3,639 development and 946 validation rows. VQC notebook: accuracy 0.9929, ROC-AUC 0.99997. |
| Proxy leakage | `total_UPDRS` is correctly excluded in the notebooks. It correlates with `motor_UPDRS` at r = 0.947 and gives ROC-AUC about 0.98 when added (our ablation). |
| Split | Subject-wise 33 train / 9 test, zero overlap, asserted in the code. Dev/val 3,639 / 946 rows appears in the QNN sweep output; test 1,290 rows is asserted. Preprocessing is fitted on training rows only. The split is sound. The features and label are the problem. |
| Severity construction | Not reproducible from the notebooks (split file not supplied). We define the label ourselves (dev-median threshold of `motor_UPDRS`). Absolute numbers are not comparable with the original. |
| FL aggregation for trees | Not checked: the notebook audit covered the logistic FedAvg code. Our paper implements a client-weighted soft-vote ensemble instead of averaging tree parameters. |
| Statistics | Not present in the notebooks we saw. We add repeated partitions, t-intervals, tests against chance, Wilcoxon tests with Holm correction, and a subject-cluster bootstrap. |
| Not auditable | ROC-AUC inconsistencies between tables, reference list, figure quality and text claims. These are PDF-only. |

## 2. Methodology (summary; details in `paper/main.tex` Sections IV–IX)

1. Define the target first: `y = 1[motor_UPDRS >= median over development subjects]`.
2. Inputs: 19 features (age, sex, test_time, 16 voice measures). Exclude `motor_UPDRS`, `total_UPDRS` and subject ID.
3. Ten random subject-wise partitions (26/7/9 subjects). Imputer and scaler are fitted on development data only.
4. Centralized: LR, linear SVM, RF, ET, with subject-grouped 5-fold CV on development data.
5. Federated: FedAvg and FedProx for LR, one client per subject. RF and ET use a client-weighted soft-vote ensemble of local forests (one shot).
6. Quantum (ideal simulator, 4 qubits, 3 layers): VQC (fixed feature map, 24 parameters) and QNN (data re-uploading with a classical head, 41 parameters).
7. Metrics, intervals and tests as in the paper.

## 3. Reproducibility checklist

- [x] Dataset: UCI Parkinson's Telemonitoring, 5,875 rows, 42 subjects, no missing values (checked).
- [x] Seeds 0–9 fix partitions and all models.
- [x] Software: Python 3.13, scikit-learn 1.9.1, PennyLane (pip), NumPy, SciPy, matplotlib.
- [x] Code: `code/common.py`, `run_classical_fl.py`, `run_rowsplit.py`, `run_quantum.py`, `stats.py`, `figs.py`, `gen_tables.py`. Order: classical_fl, quantum, rowsplit, stats, figs, gen_tables.
- [x] All tables and figures are generated from `results/*.csv`. No numbers are typed by hand into the tables.
- [x] Hardware: two-core CPU. No GPU and no quantum hardware.
- [ ] Not re-run twice to confirm bit-identical output (not tested).
- [ ] `DATA` path in `common.py` is the local path used here; change it for your machine.
- [ ] The original split file was not available, so the original split is not reproduced.

## 4. Reference verification log

Verified online (author, title, venue, year and the stated fields were confirmed from a result page): Tsanas 2010; Kaufman 2012; Saeb 2017; Li, Wen & He 2020 (note: the authors are Li, Wen and He); Pérez-Salinas 2020 (venue and year; the Quantum 4:226 number follows the publisher URL pattern); Schuld 2020 (venue, volume, article id from the ADS bibcode); Li et al. FedProx (title, authors, MLSys 2020; **volume and pages not confirmed, so pages are omitted from the entry**).

Entered from memory, **not** confirmed online in this session. Check every DOI and page range before submission: Little 2009; Tsanas 2012; Neto 2019 (author list); McMahan 2017; Cheng 2021; Rieke 2020; Kairouz 2021; Bonawitz 2017; Zhu 2019; Abadi 2016; Havlíček 2019; Schuld & Killoran 2019; Mitarai 2018; Cerezo 2021; McClean 2018; Preskill 2018; Bergholm 2018; Breiman 2001; Geurts 2006; Chicco & Jurman 2020; Pedregosa 2011; Kingma & Ba 2015; Holm 1979; Wilcoxon 1945.

Note: `Schuld & Killoran 2019` is in `refs.bib` but not cited in the text. Remove it or cite it.

## 5. Update for the ICAUC 2027 paper (6 A4 pages)

- The quantum models (VQC, QNN) are now trained on the **complete development set of every partition** (3,517–3,697 records, no subsampling). The earlier 800-record runs are kept as `results/quantum_runs_subsample800.csv` and `quantum_curves_subsample800.csv` for reference only and are not used in the paper.
- A monitoring bug in the earlier quantum run (training loss measured on class-0 records only) is fixed in `code/run_quantum.py`; the full-data run uses a random 600-record monitoring subset.
- The paper is generated from one source (`code/build_paper.py` + `paperlib.py`) into both LaTeX/PDF and DOCX; all numbers are read from `results/*.csv`. Build: `python3 build_paper.py` then `pdflatex main.tex` twice in `paper_final/`.
- Final files: `ICAUC2027_71364_PD_Severity_Paper.pdf/.docx/.tex`.

import sys, os, json, pickle, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *
import paperlib as PL
from refs_data import REFS

R = os.path.join(BASE_DIR, "results") + os.sep
OUT = os.path.join(BASE_DIR, "paper_final") + os.sep
FIGDIR = os.path.join(BASE_DIR, "figures") + os.sep
TMP = os.path.join(BASE_DIR, "paper_final", ".tmp_eq")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
S = pd.read_csv(R + "summary_test.csv").set_index('model'); cost = pd.read_csv(R + "cost.csv").set_index('model'); sig = pd.read_csv(R + "significance.csv")
boot = pd.read_csv(R + "bootstrap_seed0.csv").set_index('model')
val = pd.read_csv(R + "summary_val.csv").set_index('model'); trn = pd.read_csv(R + "summary_train.csv").set_index('model')
ps = pd.read_csv(R + "per_subject.csv")
def a(m, k, d=3): return f"{S.loc[m, k]:.{d}f}"
def pm(m, k, d=3): return f"{S.loc[m, k]:.{d}f} ± {S.loc[m, k + '_ci']:.{d}f}"
def pv(model, test, col='p_holm'):
    r = sig[(sig.model == model) & sig.test.str.startswith(test)].iloc[0]; return r[col]
df = load()
rows = {}
for n_ in ('Development', 'Validation', 'Test'): rows[n_] = []
for sd in range(10):
    D, V, T, thr = make_sets(df, sd)
    for n_, x in (('Development', D), ('Validation', V), ('Test', T)): rows[n_].append((x['subject#'].nunique(), len(x), x.y.mean()))
D0, V0, T0, thr0 = make_sets(df, 0)
cs = pd.DataFrame(json.load(open(R + "client_stats_seed0.json"))); n_single0 = int((cs.pos_rate == 0).sum()); n_single1 = int((cs.pos_rate == 1).sum()); n_mixed = len(cs) - n_single0 - n_single1
etsub = ps[ps.model == 'ET'].groupby('subject').acc.mean(); n_gt = int((etsub > .5).sum()); n_subj = len(etsub)
vr = val[['auc']].auc; v_lo, v_hi = vr.min(), vr.max()

meta = dict(
 title="Quantum Machine Learning-Based Parkinson’s Disease Severity Prediction: A Subject-Wise Comparison with Centralized and Federated Learning",
 authors="Author Name(s)", affil="Department, Institution, City, Country | email@example.org",
 keywords="Parkinson’s disease, UPDRS, severity classification, federated learning, variational quantum classifier, quantum neural network, voice telemonitoring, subject-wise validation.",
 abstract=("Voice telemonitoring can follow motor impairment in Parkinson’s disease (PD) cheaply, but it is unclear whether federated and quantum learning match conventional models on this task. "
 "Using all 5,875 recordings of the 42 patients in the Parkinson’s Telemonitoring data set, we classify motor severity (above or below the development-set median of the motor UPDRS score) under one protocol shared by three paradigms: centralized learners (logistic regression, linear SVM, Random Forest, Extra Trees), federated learners (FedAvg and FedProx logistic regression and client-weighted forest ensembles, one client per patient) and two four-qubit variational circuits, a variational quantum classifier (VQC) and a data re-uploading quantum neural network (QNN), run on an ideal simulator with the complete development set. "
 f"Ten subject-wise partitions (26 development, 7 validation and 9 test patients) give means and 95% intervals for nine metrics. Extra Trees performs best (ROC-AUC {pm('ET','auc')}, balanced accuracy {a('ET','bacc')}); federated logistic regression is statistically indistinguishable from it ({pm('FedAvg-LR','auc')}), whereas the federated forests and both quantum circuits stay close to chance (ROC-AUC {a('FedForest-RF','auc')}–{a('VQC','auc')}), the QNN being significantly worse than Extra Trees after Holm correction. "
 "We claim no quantum advantage, report no hardware results and implement no privacy mechanism."))

B = []
def h1(t): B.append(dict(t='h1', text=t))
def h2(t): B.append(dict(t='h2', text=t))
import re as _re
def autosub(s):
    s = _re.sub(r'(?<![\w$])([A-Za-z])_\{([^}]+)\}', lambda m: '$' + m.group(1) + '_{' + m.group(2) + '}$', s)
    s = _re.sub(r'(?<![\w$])([A-Za-z])_([A-Za-z0-9]{1,2})(?![\w}])', lambda m: '$' + m.group(1) + '_{' + m.group(2) + '}$', s)
    return s
def p(t): B.append(dict(t='p', text=autosub(t)))

h1("Introduction")
p("Voice recordings are easy to collect at home, and dysphonia measures change as Parkinson’s disease (PD) progresses, which makes speech attractive for remote monitoring [@little2009;@tsanas2010]. The Parkinson’s Telemonitoring data set holds 5,875 recordings from 42 patients, each labelled with the motor and total scores of the Unified Parkinson’s Disease Rating Scale (UPDRS). Federated learning (FL) trains a shared model without pooling raw recordings, which suits distributed health sensing and edge devices [@mcmahan2017;@rieke2020], and quantum machine learning (QML) proposes parameterised quantum circuits as classifiers [@havlicek2019;@schuld2020]. Both are often presented as alternatives to conventional learners but are rarely compared with them under one evaluation protocol.")
p("**Scope.** This paper addresses the *severity* of motor impairment in patients who already have PD; it is not a diagnostic study, because the data contain no healthy controls.")
p("**Gap and contributions.** Prior work on this data set mostly predicts UPDRS by regression using randomly chosen records [@tsanas2010] or separates patients from controls [@tsanas2012]. We found no study that classifies severity for unseen patients, compares centralized, federated and quantum models on identical patient-wise partitions, or specifies how tree ensembles are federated without averaging tree parameters. We contribute (1) a unified protocol with one explicit severity label, ten patient-wise partitions, development-only preprocessing and nine shared metrics; (2) a client-weighted soft-voting scheme for federated Random Forest and Extra Trees, next to FedAvg and FedProx for logistic regression; (3) fully specified four-qubit VQC and QNN circuits trained on the complete development set; and (4) a statistical comparison with intervals, tests against chance, Holm-corrected paired tests and communication and computation costs.")

h1("Related Work")
p("Little *et al.* showed that dysphonia measures separate PD patients from controls [@little2009], and Tsanas *et al.* predicted UPDRS from 16 voice measures of the telemonitoring recordings [@tsanas2010] and later classified PD with a compact set of selected features [@tsanas2012]. Because repeated recordings of one person are highly similar, evaluation that mixes a patient’s records across training and test sets can differ strongly from evaluation on new patients [@saeb2017]; we therefore partition by patient.")
p("Federated averaging trains a parametric model by repeated local optimisation and weighted averaging [@mcmahan2017]; FedProx adds a proximal term against client drift under heterogeneity [@li2020prox]. Forests have no averageable parameter vector, so federated tree methods use histogram sharing or encryption [@li2020gbdt;@cheng2021]; reviews cover privacy and heterogeneity [@kairouz2021].")
p("Variational circuits encode classical data into a quantum state and tune gate parameters against a classical loss [@mitarai2018;@cerezo2021]. Havlíček *et al.* proposed a variational classifier and a kernel method with a quantum feature map [@havlicek2019]; Schuld *et al.* analysed circuit-centric classifiers [@schuld2020]; and Pérez-Salinas *et al.* showed that re-uploading the data inside the circuit raises expressivity [@perez2020]. Trainability can be limited by barren plateaus [@mcclean2018] and devices are small and noisy [@preskill2018], so simulated few-qubit results cannot show hardware advantage.")

h1("Dataset and Problem Formulation")
p(f"**Data.** The data set (UCI Parkinson’s Telemonitoring) contains N = 5,875 recordings from 42 PD patients (101–168 per patient) collected over six months, with no missing values (verified). Each row has the patient identifier, age, sex, test_time (days since recruitment), motor_UPDRS, total_UPDRS and 16 voice measures: five jitter, six shimmer, NHR, HNR, RPDE, DFA and PPE [@tsanas2010]. **All 5,875 recordings and all 42 patients are used; no sampling or subsetting is applied to any model.**")
p("**Target.** The data contain no categorical severity label, so we define one explicitly:")
B.append(dict(t='eq', tex=r"y_i=\mathbf{1}[s_i\ge\tau],\quad \tau=\mathrm{median}\{s_j: j\in\mathcal{D}\}", mpl=r"y_i=\mathbf{1}[s_i\geq\tau],\ \tau=\mathrm{median}\{s_j: j\in\mathcal{D}\}", label='eq:y', docx_w=3.0))
p("where s_i is the motor_UPDRS score of record i and D the development set; the threshold is a development-only statistic applied unchanged to validation and test patients. The classes are balanced on D by construction but not elsewhere (\\tabref{tab:data}). The label is population-relative, not a clinical cut-off. Motor_UPDRS and total_UPDRS serve only to define the label and are not model inputs; the 19 input features are age, sex, test_time and the 16 voice measures. The patient identifier is used only for grouping.")
p(f"**Partitioning.** For each of ten seeds the 42 patients are permuted and assigned to development D (26), validation V (7) and test T (9), so no patient appears in two sets (\\tabref{{tab:data}}, \\figref{{fig:partfed}} shows the federated clients). The ten partitions reuse the same patients and are therefore not independent. In partition 0, {n_single0} of the 26 development patients have only class-0 records, {n_single1} only class-1 records and {n_mixed} both, so the label is largely a per-patient attribute; this strongly non-IID structure matters for federated learning.")
tab_rows = []
for n_, x in (('Development', D0), ('Validation', V0), ('Test', T0)):
    r_ = np.array(rows[n_]); tab_rows.append([n_, x['subject#'].nunique(), len(x), f"{x.y.mean():.3f}", f"{int(r_[:,1].min()):,}–{int(r_[:,1].max()):,}", f"{r_[:,2].min():.2f}–{r_[:,2].max():.2f}"])
B.append(dict(t='table', w='col', label='tab:data', cap=f"Patient-wise partition. Columns 2–4: partition 0 (τ = {thr0:.2f}); last two: range over ten partitions.", cols='lccccc', head=['Set', 'Patients', 'Records', 'Pos. rate', 'Records (range)', 'Pos. rate (range)'], rows=tab_rows, tabcolsep='2pt', colw=[1.3,.8,.8,.8,1.2,1.2]))

h1("Methodology")
p("**Preprocessing.** Median imputation (a no-op here) and standardisation are fitted on D only and applied to V and T; all 19 features enter every classical and federated model. The quantum models add a PCA to four components (fitted on D) and min–max scaling to [0, π] with clipping. Imbalance is handled by reporting balanced accuracy, macro-F1 and the Matthews correlation coefficient (MCC) [@chicco2020], not by re-weighting.")
p("**Model selection.** Hyper-parameters are chosen by five-fold patient-grouped cross-validation within D, with balanced accuracy as criterion. The validation patients are used only to report generalisation and convergence, and the test patients only for final scoring. Federated forests reuse the depth and leaf size chosen for the matching centralized forest and are not tuned separately.")
p("**Metrics.** Accuracy, balanced accuracy, precision, recall (for the severe class), macro- and weighted-F1, ROC-AUC, PR-AUC (average precision) and MCC. Hard predictions use probability 0.5 (decision value 0 for the SVM). Intervals are two-sided 95% *t*-intervals over ten partitions; we add a patient-cluster bootstrap (1,000 resamples) for partition 0. **Set-up.** Seeds 0–9 fix partitions and models on a two-core CPU container (Python, scikit-learn, PennyLane); federated clients are simulated sequentially, and communication is counted analytically (LR) or from serialised sizes (forests).")
B.append(dict(t='fig', w='wide', label='fig:overall', files=[('fig1_overall_architecture', 0.66)], cap="Overall architecture: the severity label is defined first, models see only the 19 input features, and all three paradigms share the same patient-wise partitions, development-only preprocessing and metrics."))

h1("Centralized Machine Learning")
p("All learners see the pooled development set (\\figref{fig:overall}). **Logistic regression** (LR) minimises the regularised cross-entropy with *C* ∈ {0.01, 0.1, 1, 10}. The **linear SVM** minimises ½‖w‖² + *C* Σ max(0, 1 − ỹᵢ(wᵀxᵢ + b))² with *C* ∈ {10⁻³, …, 1}. **Random Forest** (RF) averages 200 bootstrap trees grown with Gini impurity and random feature subsets [@breiman2001]; **Extra Trees** (ET) uses the full sample and random split thresholds [@geurts2006]; both predict the mean of the tree probabilities. For the forests the grid is depth {4, 8, none} × minimum leaf size {1, 10, 40} (\\tabref{tab:hp}); implementation is scikit-learn [@pedregosa2011]. Selected forest settings vary across partitions, so 26 patients do not identify a stable optimum.")
B.append(dict(t='table', w='col', label='tab:hp', cap="Settings. Selected values over ten partitions in parentheses (count).", cols='lp{6.2cm}', head=['Model', 'Setting'], tabcolsep='2pt', colw=[1.1,3.4], rows=[
 ['LR', '*C*: 0.01 (6), 1 (3), 10 (1)'], ['Linear SVM', '*C*: 0.001 (6), 1 (3), 0.1 (1)'],
 ['RF, 200 trees', 'depth/leaf: 4/40 (3), none/1 (3), 4/10 (2), 4/1 (1), none/10 (1)'], ['ET, 200 trees', 'none/1 (5), 4/40 (3), 4/1 (1), 8/40 (1)'],
 ['FedAvg / FedProx', '40 rounds, 5 local full-batch steps, lr 0.1, μ ∈ {0, 0.1}, participation 1.0 or 0.5'],
 ['Federated forests', '20 trees per client, depth and leaf of the centralized forest'],
 ['VQC / QNN', '4 qubits, 3 layers, Adam lr 0.05, batch 128, 40 epochs, all development records']]))

h1("Federated Learning")
p("Each of the K = 26 development patients is one client holding n_k records (Σ n_k = n). The data are strongly non-IID, since most clients contain a single class. The scaler is fitted on pooled development data, which in practice would require sharing feature means and variances.")
p("**Logistic regression.** In round t the server samples a client set S_t (fraction 1.0 or 0.5), broadcasts w_t, and every selected client takes E = 5 full-batch gradient steps on F_k(w) + (μ/2)‖w − w_t‖² (μ = 0 gives FedAvg, μ = 0.1 FedProx). The server aggregates")
B.append(dict(t='eq', tex=r"\mathbf{w}_{t+1}=\sum_{k\in S_t}\frac{n_k}{\sum_{j\in S_t}n_j}\,\mathbf{w}_k^{t+1},", mpl=r"\mathbf{w}_{t+1}=\sum_{k\in S_t}\frac{n_k}{\sum_{j\in S_t}n_j}\,\mathbf{w}_k^{t+1},", label='eq:fedavg', docx_w=2.4))
p(f"with the same rule for the bias. A round moves 2|S_t|(d + 1) 64-bit numbers, so 40 rounds with 26 clients cost {cost.loc['FedAvg-LR','comm_bytes']/1e3:.1f} kB.")
p("**Forests.** Averaging tree structure or thresholds is undefined, so FedAvg is not applied to forests. Each client fits a local forest of 20 trees on its own records and uploads it once; the server forms the client-weighted soft-vote ensemble")
B.append(dict(t='eq', tex=r"\hat p(\mathbf{x})=\sum_{k=1}^{K}\frac{n_k}{n}\,\hat p_k(\mathbf{x}),\qquad \hat p_k(\mathbf{x})=\frac{1}{20}\sum_{t=1}^{20}p_{k,t}(\mathbf{x}),", mpl=r"\hat p(\mathbf{x})=\sum_{k=1}^{K}\frac{n_k}{n}\,\hat p_k(\mathbf{x}),\ \hat p_k(\mathbf{x})=\frac{1}{20}\sum_{t=1}^{20}p_{k,t}(\mathbf{x}),", label='eq:ens', docx_w=3.0))
p(f"which is a single forest of 20K trees with client-dependent weights. Upload is one-shot: {cost.loc['FedForest-RF','comm_bytes']/1e3:.0f} kB (RF) and {cost.loc['FedForest-ET','comm_bytes']/1e3:.0f} kB (ET) on average, so there is no convergence curve; we instead vary the number of contributing clients (\\figref{{fig:fed}}).")
p("**Privacy limits.** Shared weights reveal information through gradient inversion [@zhu2019], and uploaded trees expose split thresholds and leaf distributions. No secure aggregation [@bonawitz2017] or differential privacy [@abadi2016] is implemented, so no privacy guarantee is claimed.")
B.append(dict(t='fig', w='col', label='fig:partfed', files=[('fig5_federated_architecture', 0.8)], cap="Federated architecture: weights (LR) or local forests are exchanged; raw recordings never leave a client."))

h1("Quantum Machine Learning")
p("**Configuration.** Both circuits use n = 4 qubits starting in |0⟩. A standardised record is projected by PCA to R⁴ and scaled to x_i ∈ [0, π]. This projection is a strong information bottleneck but keeps the simulation exact and the circuits shallow.")
p("**VQC.** A fixed feature map applies H and R_Z(x_i) on each qubit, followed by a ring of ZZ interactions CNOT·R_Z(x_i x_{i+1})·CNOT. A trainable ansatz of L = 3 layers applies R_Y(θ)R_Z(φ) on every qubit and a CNOT ring (\\figref{fig:circ}a). With encoding unitary Φ(x) and trainable unitary U(θ),")
B.append(dict(t='eq', tex=r"f(\mathbf{x};\boldsymbol\theta)=\langle 0|\Phi^\dagger U^\dagger Z_0\,U\Phi|0\rangle,\qquad p(y{=}1\mid\mathbf{x})=\tfrac{1-f}{2}.", mpl=r"f(\mathbf{x};\theta)=\langle 0|\Phi^{\dagger}U^{\dagger}Z_0\,U\Phi|0\rangle,\ p(y{=}1|\mathbf{x})=\frac{1-f}{2}.", label='eq:vqc', docx_w=3.0))
p("It has 24 trainable parameters and 56 gates (20 CNOTs).")
p("**QNN.** The QNN uses data re-uploading [@perez2020]: in each of L = 3 layers every qubit receives R_Y(w x_i + c) with trainable scale w and offset c, then R_Z(θ) and a CNOT ring (\\figref{fig:circ}b). The four expectations e = (⟨Z_0⟩, …, ⟨Z_3⟩) feed a classical linear read-out p = σ(vᵀe + b). It has 41 trainable parameters and 36 gates (12 CNOTs).")
p("**QNN versus VQC.** The VQC encodes once with fixed angles and reads one observable; the QNN re-encodes in every layer with trainable scaling and reads four observables through a classical head, making it a hybrid model with more parameters.")
p(f"**Training and simulation.** Loss is binary cross-entropy, minimised with Adam (lr 0.05, batch 128, 40 epochs) on *all* development records of each partition (3,517–3,697). Circuits are simulated noiselessly with exact state vectors in PennyLane [@bergholm2018], with gradients by back-propagation through the simulator. **No experiment was run on quantum hardware and no shot or device noise was modelled.** On hardware each rotation gradient would follow the parameter-shift rule [@mitarai2018], needing 2P circuit evaluations per record for P parameters. State-vector simulation costs O(G·2ⁿ) per record for G gates, cheap at n = 4 but exponential in n; training took about {cost.loc['VQC','fit_s']:.0f} s (VQC) and {cost.loc['QNN','fit_s']:.0f} s (QNN) per run.")
B.append(dict(t='fig', w='wide', label='fig:circ', files=[('fig8a_vqc_circuit', 0.62), ('fig8b_qnn_circuit', 0.62)], stack=True, cap="Circuits drawn from the executed PennyLane code: (a) VQC with feature map and three trainable layers, measured on qubit 0; (b) QNN with three data re-uploading layers, all four qubits measured."))

h1("Results and Discussion")
tr = []
def resrow(m, name): return [name] + [pm(m, k) for k in ['acc', 'bacc', 'prec', 'rec', 'f1m', 'f1w', 'auc', 'prauc', 'mcc']]
groups = [('Centralized', [('LR', 'LR'), ('SVM', 'Linear SVM'), ('RF', 'Random Forest'), ('ET', 'Extra Trees')]),
          ('Federated', [('FedAvg-LR', 'FedAvg-LR'), ('FedAvg-LR (50% clients)', 'FedAvg-LR, 50%'), ('FedProx-LR (mu=0.1)', 'FedProx-LR'), ('FedForest-RF', 'Fed-RF'), ('FedForest-ET', 'Fed-ET')]),
          ('Quantum (simulated)', [('VQC', 'VQC'), ('QNN', 'QNN')])]
for g, ms in groups:
    tr.append('--'); tr.append(['**' + g + '**'] + [''] * 9)
    for m, nm in ms: tr.append(resrow(m, nm))
B.append(dict(t='table', w='wide', label='tab:res', cap="Test results (mean ± 95% *t*-interval over ten patient-wise partitions; positive class = more severe) for centralized, federated and quantum models.", cols='l' + 'c' * 9, tabcolsep='2.5pt',
              head=['Model', 'Acc.', 'Bal. acc.', 'Prec.', 'Rec.', 'Macro-F1', 'Wtd-F1', 'ROC-AUC', 'PR-AUC', 'MCC'], colw=[1.3]+[1.0]*9, rows=tr))
p(f"**Centralized models.** \\tabref{{tab:res}} and \\figref{{fig:acc}} show that Extra Trees is best (balanced accuracy {a('ET','bacc')}, ROC-AUC {a('ET','auc')}, MCC {a('ET','mcc')}); LR and the linear SVM reach ROC-AUC ≈ {a('LR','auc',2)}, and RF is at chance ({a('RF','auc')}) with a wide interval. Training balanced accuracy of RF and ET ({trn.loc['RF','bacc']:.2f}, {trn.loc['ET','bacc']:.2f}) far exceeds their test value: the forests fit patient-specific structure that does not transfer. ET gains through recall of the severe class ({a('ET','rec')}) at the cost of precision ({a('ET','prec')}).")
p(f"**Federated models.** FedAvg-LR ({a('FedAvg-LR','auc')}) and FedProx ({a('FedProx-LR (mu=0.1)','auc')}; same ROC-AUC to three decimals) are close to ET and above centralized LR, whereas the one-shot federated forests sit at chance ({a('FedForest-RF','auc')}, {a('FedForest-ET','auc')}). Training loss falls monotonically with full participation (\\figref{{fig:fed}}a) and test ROC-AUC settles near {a('FedAvg-LR','auc',2)} (\\figref{{fig:fed}}b). Sampling 50% of clients halves traffic ({cost.loc['FedAvg-LR (50% clients)','comm_bytes']/1e3:.0f} versus {cost.loc['FedAvg-LR','comm_bytes']/1e3:.0f} kB) and lowers ROC-AUC to {a('FedAvg-LR (50% clients)','auc')} with noisier convergence; FedProx brings no gain. For forests, adding clients does not help (\\figref{{fig:fed}}c): a local forest trained on a single-class client mostly learns that patient’s class.")
p(f"**Quantum models.** The VQC (ROC-AUC {pm('VQC','auc')}, MCC {a('VQC','mcc')}) and QNN ({pm('QNN','auc')}, MCC {a('QNN','mcc')}) stay close to chance, although they were trained on the entire development set. Training loss decreases (QNN 0.65 → 0.58, VQC 0.66 → 0.64) while validation loss rises for the QNN (0.72 → 0.88) and stays flat for the VQC (0.71 → 0.72) (\\figref{{fig:cost}}c), indicating over-fitting by the QNN even with 41 parameters. Limits of the circuits cannot be separated from limits of the data and of the four-component projection: the result is *no evidence of benefit under this protocol*, not evidence against QML.")
p(f"**Overall.** The ordering by ROC-AUC is ET > FedAvg-LR ≈ FedProx > LR ≈ SVM > VQC > RF ≈ QNN ≈ Fed-ET ≈ Fed-RF. Federated logistic regression is thus competitive with its centralized counterpart. The ROC and precision–recall curves and confusion matrices for partition 0 (\\figref{{fig:roc}}) agree: Fed-ET predicts the milder class for most records, and the VQC and QNN curves follow the diagonal. Performance varies strongly between patients: ET accuracy per held-out patient ranges from 0 to 1 and exceeds 0.5 for {n_gt} of the {n_subj} patients that were ever in a test set ({100*n_gt/n_subj:.0f}%). The partition-0 patient-cluster bootstrap interval for ET ROC-AUC is [{boot.loc['ET','lo']:.2f}, {boot.loc['ET','hi']:.2f}], so on a single partition chance cannot be excluded even for the best model.")
B.append(dict(t='fig', w='wide', label='fig:acc', files=[('fig9_accuracy_auc_comparison', 0.7)], cap="Balanced accuracy and ROC-AUC (mean ± 95% interval over ten partitions); dashed line = chance."))
B.append(dict(t='fig', w='wide', label='fig:roc', files=[('figR_roc_pr_cm', 0.8)], cap="Partition-0 test patients: (a) ROC and (b) precision–recall curves of LR, ET, FedAvg-LR, Fed-ET, VQC and QNN; confusion matrices (rows: true class) of (c) ET, (d) FedAvg-LR and (e) VQC."))
B.append(dict(t='fig', w='wide', label='fig:fed', files=[('figF_federated', 0.78)], cap="Federated analysis (mean over ten partitions): (a) training log-loss and (b) test ROC-AUC against communication rounds for FedAvg, FedAvg with 50% participation and FedProx; (c) forest ensembles against the number of contributing clients; (d) total communication."))

h1("Statistical and Computational Analysis")
sr = []
order_m = ['ET', 'FedAvg-LR', 'FedProx-LR (mu=0.1)', 'FedAvg-LR (50% clients)', 'LR', 'SVM', 'VQC', 'RF', 'QNN', 'FedForest-ET', 'FedForest-RF']
nm_ = {'ET': 'Extra Trees', 'FedAvg-LR': 'FedAvg-LR', 'FedProx-LR (mu=0.1)': 'FedProx-LR', 'FedAvg-LR (50% clients)': 'FedAvg-LR, 50%', 'LR': 'LR', 'SVM': 'Linear SVM', 'VQC': 'VQC', 'RF': 'Random Forest', 'QNN': 'QNN', 'FedForest-ET': 'Fed-ET', 'FedForest-RF': 'Fed-RF'}
for m in order_m:
    if m == 'ET': sr.append([nm_[m], a(m, 'auc'), f"{pv(m,'AUC','p_holm'):.3f}", '–', '–'])
    else: sr.append([nm_[m], a(m, 'auc'), f"{pv(m,'AUC','p_holm'):.3f}", f"{pv(m,'Wilcoxon','mean_diff'):+.3f}", f"{pv(m,'Wilcoxon','p_holm'):.3f}"])
B.append(dict(t='table', w='col', label='tab:stats', cap="Tests over ten partitions (Holm-corrected *p*). *p*₁: ROC-AUC against 0.5 (one-sample *t*-test). Δ: ET minus model; *p*₂: Wilcoxon signed-rank test of ET against the model.", cols='lcccc', tabcolsep='2.5pt', head=['Model', 'AUC', '*p*₁', 'Δ AUC', '*p*₂'], colw=[1.5,.8,.8,.9,.8], rows=sr))
p(f"Against chance (ROC-AUC = 0.5), only Extra Trees remains significant after Holm correction [@holm1979] across the 11 models (*p* = {pv('ET','AUC','p'):.4f} uncorrected, {pv('ET','AUC','p_holm'):.3f} corrected); LR, SVM, FedAvg-LR and FedProx are nominally significant (*p* between {sig[sig.test.str.startswith('AUC')&sig.model.isin(['FedAvg-LR','LR','SVM','FedProx-LR (mu=0.1)'])].p.min():.3f} and {sig[sig.test.str.startswith('AUC')&sig.model.isin(['FedAvg-LR','LR','SVM','FedProx-LR (mu=0.1)'])].p.max():.3f}) but not after correction (\\tabref{{tab:stats}}). In paired Wilcoxon tests [@wilcoxon1945], ET exceeds the QNN (corrected *p* = {pv('QNN','Wilcoxon'):.3f}); its advantage over the VQC, RF and the federated forests is nominal (uncorrected *p* = {pv('VQC','Wilcoxon','p'):.3f}–{pv('RF','Wilcoxon','p'):.3f}) but not significant after correction, and ET is not distinguishable from FedAvg-LR, FedProx, LR or the SVM (*p* ≥ {min(pv(m,'Wilcoxon','p_holm') for m in ['FedAvg-LR','FedProx-LR (mu=0.1)','LR','SVM']):.2f}). The data support “ET is above chance” and “ET beats the QNN”, not “centralized beats federated logistic regression”.")
p("**These tests are optimistic.** The ten partitions resample the same 42 patients, so test sets overlap and the ten observations are positively correlated; the effective sample size is closer to the number of patients than to ten. The smallest attainable two-sided Wilcoxon *p* with ten pairs is 0.002. Intervals and *p*-values are indicative, not confirmatory.")
p(f"**Computational cost** (\\figref{{fig:cost}}a,b). LR and the SVM train in about 0.01 s, ET in {cost.loc['ET','fit_s']:.1f} s and RF in {cost.loc['RF','fit_s']:.1f} s; the simulated VQC and QNN need about {cost.loc['VQC','fit_s']:.0f}–{cost.loc['QNN','fit_s']:.0f} s on the full development set, i.e. 30 to 3,000 times more, with inference of about {cost.loc['VQC','infer_s']*1e3:.0f} ms for roughly 1,300 test records against about 1 ms for LR. The simulator is exponential in qubit count and hardware would add shot costs, so this comparison favours the quantum models if anything. FedAvg communicates {cost.loc['FedAvg-LR','comm_bytes']/1e3:.0f} kB over 40 rounds and the forest ensembles {cost.loc['FedForest-RF','comm_bytes']/1e3:.0f}–{cost.loc['FedForest-ET','comm_bytes']/1e3:.0f} kB once.")
B.append(dict(t='fig', w='wide', label='fig:cost', files=[('figQ_cost_training', 0.78)], cap="(a) Training time and (b) test inference time (log scale) of centralized and simulated quantum models; (c) QNN and VQC training and validation loss (mean over ten partitions)."))

h1("Limitations and Threats to Validity")
p(f"With 42 patients any partition has nine test patients, and the patient-cluster bootstrap interval spans most of the unit interval; the ten partitions are dependent. The median-threshold label is not a clinical severity class and results may differ for other cut-offs. Real deployments add device and site heterogeneity beyond single-patient clients. Selected hyper-parameters change across partitions, and validation sets of seven patients are noisy (mean validation ROC-AUC {v_lo:.2f}–{v_hi:.2f} against test values of 0.50–0.64), so they are unreliable for early stopping. The quantum models use one circuit depth, one optimiser setting and a four-component projection, with no tuning, noise model or hardware. Federated forests inherit centralized settings and a pooled scaler; no differential privacy, secure aggregation or attack evaluation is implemented.")

h1("Conclusion and Future Work")
p(f"Under a unified patient-wise protocol on the complete Parkinson’s Telemonitoring data, centralized Extra Trees reaches ROC-AUC {pm('ET','auc')}, federated logistic regression is statistically indistinguishable from it ({a('FedAvg-LR','auc')}) at a communication cost of {cost.loc['FedAvg-LR','comm_bytes']/1e3:.0f} kB, the one-shot federated forests are at chance, and the simulated four-qubit VQC and QNN stay close to chance, the QNN being significantly worse than ET. We found no evidence of quantum advantage and make no claim about hardware. Future work should use larger multi-site cohorts, predict continuous UPDRS, add federated boosting with secure aggregation and measured differential privacy, and test the quantum models with tuned depth, noise models and hardware.")

if __name__ == '__main__':
    order = PL.order_cites(B + [dict(t='p', text=meta['abstract'])])
    missing = [k for k in order if k not in REFS]; assert not missing, missing
    # abstract cites none; order scanned from blocks
    PL.render_tex(meta, B, REFS, order, OUT + 'main.tex')
    PL.render_docx(meta, B, REFS, order, OUT + 'paper.docx', FIGDIR, TMP)
    json.dump(order, open(OUT + 'cite_order.json', 'w'))
    print('refs cited:', len(order))

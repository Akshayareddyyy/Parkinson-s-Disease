"""
Experiment B: Quantum Machine Learning Architecture & Feature Representation Investigation
Compares:
  1. Feature representation: PCA-4 vs Supervised Top-4 (ANOVA F-statistic on training folds)
  2. Circuit Depth: L in {1, 2, 3, 4}
  3. Entanglement topology: Ring, Linear, and All-to-All
  4. Architectures: VQC vs Data Re-uploading QNN
  5. Matched classical baselines (LR, SVM, RF, ET) on the identical 4-feature representations
Strict subject-wise partitions (26/7/9).
Outputs saved to results/exp_b_*.csv.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, DATA, FEATS, load, partition, make_sets, metrics
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import pennylane as qml
from pennylane import numpy as pnp

R = os.path.join(BASE_DIR, "results") + os.sep
os.makedirs(R, exist_ok=True)
df = load()

NQ = 4
dev_q = qml.device("default.qubit", wires=NQ)

def apply_entanglement(wires, topology='ring'):
    if topology == 'ring':
        for i in range(len(wires)):
            qml.CNOT(wires=[wires[i], wires[(i+1)%len(wires)]])
    elif topology == 'linear':
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
    elif topology == 'all_to_all':
        for i in range(len(wires)):
            for j in range(i+1, len(wires)):
                qml.CNOT(wires=[wires[i], wires[j]])
    else:
        raise ValueError(f"Unknown topology {topology}")

def make_vqc_qnode(L=3, topology='ring'):
    @qml.qnode(dev_q, interface="autograd", diff_method="backprop")
    def circuit(x, th):
        # Feature map: Hadamard + RZ + ZZ ring
        for i in range(NQ):
            qml.Hadamard(i)
            qml.RZ(x[:, i], i)
        for i in range(NQ):
            j = (i + 1) % NQ
            qml.CNOT(wires=[i, j])
            qml.RZ(x[:, i] * x[:, j], wires=j)
            qml.CNOT(wires=[i, j])
        # Variational ansatz
        for l in range(L):
            for i in range(NQ):
                qml.RY(th[l, i, 0], wires=i)
                qml.RZ(th[l, i, 1], wires=i)
            apply_entanglement(list(range(NQ)), topology)
        return qml.expval(qml.PauliZ(0))
    return circuit

def make_qnn_qnode(L=3, topology='ring'):
    @qml.qnode(dev_q, interface="autograd", diff_method="backprop")
    def circuit(x, w, c, th):
        for l in range(L):
            for i in range(NQ):
                qml.RY(w[l, i] * x[:, i] + c[l, i], wires=i)
                qml.RZ(th[l, i], wires=i)
            apply_entanglement(list(range(NQ)), topology)
        return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]
    return circuit

sig = lambda z: 1.0 / (1.0 + pnp.exp(-z))

def bce(p, y):
    p = pnp.clip(p, 1e-6, 1.0 - 1e-6)
    return -pnp.mean(y * pnp.log(p) + (1.0 - y) * pnp.log(1.0 - p))

def extract_features(D, V, T, mode='pca', seed=0):
    """
    Fits scaler and dimensionality reduction on development set D ONLY.
    mode='pca': StandardScaler + PCA(4) + MinMaxScaler(0, pi)
    mode='supervised': StandardScaler + SelectKBest(f_classif, 4) + MinMaxScaler(0, pi)
    """
    sc = StandardScaler().fit(D[FEATS])
    XD_scaled = sc.transform(D[FEATS])
    XV_scaled = sc.transform(V[FEATS])
    XT_scaled = sc.transform(T[FEATS])
    
    if mode == 'pca':
        reducer = PCA(NQ, random_state=seed).fit(XD_scaled)
        sel_features = [f"PCA_{i}" for i in range(NQ)]
    elif mode == 'supervised':
        reducer = SelectKBest(f_classif, k=NQ).fit(XD_scaled, D.y.values)
        mask = reducer.get_support()
        sel_features = [f for f, m in zip(FEATS, mask) if m]
    else:
        raise ValueError(f"Unknown mode {mode}")
        
    XD_red = reducer.transform(XD_scaled)
    XV_red = reducer.transform(XV_scaled)
    XT_red = reducer.transform(XT_scaled)
    
    mm = MinMaxScaler((0.0, np.pi)).fit(XD_red)
    f_map = lambda arr: np.clip(mm.transform(arr), 0.0, np.pi)
    
    return f_map(XD_red), f_map(XV_red), f_map(XT_red), sel_features

def run_classical_4feat_baselines(XD, yD, XT, yT, seed=0):
    res = []
    # 1. LR
    m_lr = LogisticRegression(max_iter=2000, random_state=seed).fit(XD, yD)
    s_lr = m_lr.predict_proba(XT)[:, 1]
    res.append(('LR', metrics(yT, (s_lr > 0.5).astype(int), s_lr)))
    
    # 2. Linear SVM
    m_svm = LinearSVC(max_iter=5000, random_state=seed).fit(XD, yD)
    s_svm = m_svm.decision_function(XT)
    res.append(('Linear SVM', metrics(yT, (s_svm > 0.0).astype(int), s_svm)))
    
    # 3. Random Forest
    m_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1).fit(XD, yD)
    s_rf = m_rf.predict_proba(XT)[:, 1]
    res.append(('Random Forest', metrics(yT, (s_rf > 0.5).astype(int), s_rf)))
    
    # 4. Extra Trees
    m_et = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1).fit(XD, yD)
    s_et = m_et.predict_proba(XT)[:, 1]
    res.append(('Extra Trees', metrics(yT, (s_et > 0.5).astype(int), s_et)))
    
    return res

def run_experiment_b(n_seeds=5, n_train=800, epochs=30):
    print(f"=== Starting Experiment B: Quantum Machine Learning Investigation ===")
    print(f"Seeds: {n_seeds}, Dev Training Records: {n_train if n_train > 0 else 'Full Dev Set'}, Epochs: {epochs}")
    
    q_runs = []
    q_curves = []
    cl_runs = []
    
    # Test matrix:
    # 1. Compare representations: PCA vs Supervised (L=3, ring)
    # 2. Compare depths: L in {1, 2, 3, 4} (supervised, ring)
    # 3. Compare topologies: ring vs linear vs all_to_all (supervised, L=2)
    configs = [
        # (feat_mode, model_kind, L, topology)
        ('pca', 'VQC', 3, 'ring'),
        ('pca', 'QNN', 3, 'ring'),
        ('supervised', 'VQC', 3, 'ring'),
        ('supervised', 'QNN', 3, 'ring'),
        ('supervised', 'VQC', 1, 'ring'),
        ('supervised', 'VQC', 2, 'ring'),
        ('supervised', 'VQC', 4, 'ring'),
        ('supervised', 'QNN', 1, 'ring'),
        ('supervised', 'QNN', 2, 'ring'),
        ('supervised', 'QNN', 4, 'ring'),
        ('supervised', 'VQC', 2, 'linear'),
        ('supervised', 'VQC', 2, 'all_to_all'),
        ('supervised', 'QNN', 2, 'linear'),
        ('supervised', 'QNN', 2, 'all_to_all'),
    ]
    
    for seed in range(n_seeds):
        D, V, T, thr = make_sets(df, seed)
        rng = np.random.RandomState(seed + 10)
        
        # Subsampling balanced dev set if n_train > 0
        if n_train > 0 and n_train < len(D):
            idx0 = np.where(D.y == 0)[0]
            idx1 = np.where(D.y == 1)[0]
            pick = np.concatenate([
                rng.choice(idx0, min(n_train // 2, len(idx0)), replace=False),
                rng.choice(idx1, min(n_train // 2, len(idx1)), replace=False)
            ])
            Dq = D.iloc[pick].reset_index(drop=True)
        else:
            Dq = D
            
        yD = Dq.y.values.astype(float)
        yV = V.y.values.astype(float)
        yT = T.y.values.astype(int)
        
        # Precompute features for both PCA and Supervised
        feats_dict = {}
        for fmode in ['pca', 'supervised']:
            XD, XV, XT, sel_cols = extract_features(Dq, V, T, mode=fmode, seed=seed)
            feats_dict[fmode] = (XD, XV, XT, sel_cols)
            # Evaluate matched classical 4-feature baselines
            cl_res = run_classical_4feat_baselines(XD, yD.astype(int), XT, yT, seed=seed)
            for mname, mmets in cl_res:
                cl_runs.append({
                    'seed': seed,
                    'feat_mode': fmode,
                    'selected_features': str(sel_cols),
                    'model': mname,
                    **mmets
                })
                
        # Run quantum configurations
        for fmode, kind, L, topo in configs:
            XD, XV, XT, sel_cols = feats_dict[fmode]
            r = np.random.RandomState(seed + L*100 + (10 if kind=='QNN' else 0))
            
            if kind == 'VQC':
                circuit = make_vqc_qnode(L=L, topology=topo)
                P = [pnp.array(r.uniform(0.0, 2.0*np.pi, (L, NQ, 2)), requires_grad=True)]
                fwd = lambda P_, X_: (1.0 - circuit(X_, P_[0])) / 2.0
            else: # QNN
                circuit = make_qnn_qnode(L=L, topology=topo)
                P = [
                    pnp.array(np.ones((L, NQ)), requires_grad=True),
                    pnp.array(r.uniform(0.0, 2.0*np.pi, (L, NQ)), requires_grad=True),
                    pnp.array(r.uniform(0.0, 2.0*np.pi, (L, NQ)), requires_grad=True),
                    pnp.array(r.normal(0.0, 0.5, NQ), requires_grad=True),
                    pnp.array(0.0, requires_grad=True)
                ]
                def fwd(P_, X_):
                    ex = pnp.stack(circuit(X_, P_[0], P_[1], P_[2]), axis=1)
                    return sig(ex @ P_[3] + P_[4])
                    
            n_params = sum(int(np.prod(p.shape)) for p in P)
            opt = qml.AdamOptimizer(0.05)
            t0 = time.time()
            
            batch_sz = 128
            for ep in range(1, epochs + 1):
                perm = r.permutation(len(XD))
                for b in range(0, len(perm), batch_sz):
                    j = perm[b:b+batch_sz]
                    if kind == 'QNN':
                        P = opt.step(lambda *q: bce(fwd(list(q), XD[j]), yD[j]), *P)
                    else:
                        P = [opt.step(lambda q: bce(fwd([q], XD[j]), yD[j]), P[0])]
                        
                # Log curves every 5 epochs
                if ep % 5 == 0 or ep == 1:
                    pv = np.array(fwd(P, XV))
                    pd_mon = np.array(fwd(P, XD[:min(400, len(XD))]))
                    q_curves.append({
                        'seed': seed,
                        'feat_mode': fmode,
                        'model': kind,
                        'L_depth': L,
                        'topology': topo,
                        'epoch': ep,
                        'train_loss': float(bce(fwd(P, XD[:min(400, len(XD))]), yD[:min(400, len(XD))])),
                        'val_loss': float(bce(pv, yV)),
                        'val_auc': metrics(yV.astype(int), (pv > 0.5).astype(int), pv)['auc']
                    })
                    
            fit_time = time.time() - t0
            infer_t0 = time.time()
            sT = np.array(fwd(P, XT))
            infer_time = time.time() - infer_t0
            
            q_runs.append({
                'seed': seed,
                'feat_mode': fmode,
                'model': kind,
                'L_depth': L,
                'topology': topo,
                'n_params': n_params,
                'fit_s': fit_time,
                'infer_s': infer_time,
                **metrics(yT, (sT > 0.5).astype(int), sT)
            })
            
        print(f"Quantum seed {seed}/{n_seeds-1} completed.", flush=True)
        
    df_qruns = pd.DataFrame(q_runs)
    df_curves = pd.DataFrame(q_curves)
    df_cl = pd.DataFrame(cl_runs)
    
    out_qruns = os.path.join(R, "exp_b_quantum_runs.csv")
    out_curves = os.path.join(R, "exp_b_quantum_curves.csv")
    out_cl = os.path.join(R, "exp_b_classical_4feat_runs.csv")
    
    df_qruns.to_csv(out_qruns, index=False)
    df_curves.to_csv(out_curves, index=False)
    df_cl.to_csv(out_cl, index=False)
    
    # Summary
    metrics_cols = ['acc', 'bacc', 'prec', 'rec', 'f1m', 'f1w', 'auc', 'prauc', 'mcc']
    summary_rows = []
    for (fmode, kind, L, topo), grp in df_qruns.groupby(['feat_mode', 'model', 'L_depth', 'topology']):
        s_row = {
            'feat_mode': fmode,
            'model': kind,
            'L_depth': L,
            'topology': topo,
            'n_params': grp.n_params.iloc[0],
            'fit_s_avg': grp.fit_s.mean(),
            'infer_s_avg': grp.infer_s.mean()
        }
        for mc in metrics_cols:
            m = grp[mc].mean()
            h = stats.t.ppf(0.975, len(grp) - 1) * grp[mc].std(ddof=1) / np.sqrt(len(grp)) if len(grp) > 1 else 0.0
            s_row[mc + '_mean'] = m
            s_row[mc + '_ci'] = f"{m:.3f} +/- {h:.3f}"
        summary_rows.append(s_row)
        
    df_sum = pd.DataFrame(summary_rows)
    out_sum = os.path.join(R, "exp_b_quantum_summary.csv")
    df_sum.to_csv(out_sum, index=False)
    
    print("Experiment B completed successfully!")
    print(f"Outputs written to:\n  {out_qruns}\n  {out_curves}\n  {out_cl}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_experiment_b(ns)


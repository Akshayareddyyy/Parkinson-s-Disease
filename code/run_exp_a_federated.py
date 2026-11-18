"""
Experiment A: Federated Learning under Multi-Patient Clinical Centers
Evaluates multi-patient client aggregation for K in {3, 5, 6} under:
  1. Natural client assignment (regional clinic simulation)
  2. Label-balanced client assignment (controlled simulation benchmark)
Includes:
  - Decentralized standardization (aggregating client sums and sum-of-squares)
  - FedAvg-LR and FedProx-LR (mu in {0.01, 0.1, 1.0})
  - Federated Random Forest and Extra Trees (weighted soft-vote ensembles)
  - Preserves locked-test evaluation on identical 10 partitions
  - Saves all outputs to separate results/exp_a_*.csv files
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from common import BASE_DIR, DATA, FEATS, load, partition, make_sets, metrics

R = os.path.join(BASE_DIR, "results") + os.sep
os.makedirs(R, exist_ok=True)

df = load()
NSEEDS = 10
K_CONFIGS = [3, 5, 6]
ALLOCATIONS = ['natural', 'label_balanced']
MU_LIST = [0.0, 0.01, 0.1, 1.0]  # 0.0 corresponds to standard FedAvg
ROUNDS = 40
LOCAL_EPOCHS = 5
LR_RATE = 0.1

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

def partition_dev_subjects(dev_subjects, D, K, mode='natural', seed=0):
    """
    Partitions the 26 development subjects into K clinical centers.
    mode='natural': groups consecutive subjects in permuted order.
    mode='label_balanced': greedy allocation based on patient positive rate.
    """
    if mode == 'natural':
        rng = np.random.RandomState(seed + 1000)
        perm = rng.permutation(dev_subjects)
        splits = np.array_split(perm, K)
        client_subjs = [list(s) for s in splits]
    elif mode == 'label_balanced':
        # Compute positive rate for each development subject
        sub_rates = []
        for sid in dev_subjects:
            sub_df = D[D['subject#'] == sid]
            rate = sub_df.y.mean()
            sub_rates.append((sid, rate, len(sub_df)))
        # Sort subjects by positive rate descending
        sub_rates.sort(key=lambda x: x[1], reverse=True)
        # Greedy serpentine distribution into K clinics
        client_subjs = [[] for _ in range(K)]
        client_pos = [0.0 for _ in range(K)]
        client_total = [0 for _ in range(K)]
        
        for sid, rate, n in sub_rates:
            # Assign to clinic with lowest positive count so far
            idx = min(range(K), key=lambda i: (client_pos[i] / max(1, client_total[i]), client_total[i]))
            client_subjs[idx].append(sid)
            client_pos[idx] += rate * n
            client_total[idx] += n
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Map subject list to record index lists
    client_indices = []
    for sub_list in client_subjs:
        idx = D[D['subject#'].isin(sub_list)].index.values
        client_indices.append(idx)
    return client_subjs, client_indices

def distributed_standardization(XD_clients, XV, XT):
    """
    Decentralized standardization via federated sufficient statistics aggregation:
    Clients calculate local n_k, sum(x_k), sum(x_k^2).
    Server aggregates to compute global mean and std, then broadcasts.
    """
    total_n = sum(len(x) for x in XD_clients)
    global_sum = sum(np.sum(x, axis=0) for x in XD_clients)
    global_sq_sum = sum(np.sum(x**2, axis=0) for x in XD_clients)
    
    mean = global_sum / total_n
    var = (global_sq_sum / total_n) - (mean**2)
    std = np.sqrt(np.maximum(var, 1e-12))
    
    ZD_clients = [(x - mean) / std for x in XD_clients]
    ZV = (XV - mean) / std
    ZT = (XT - mean) / std
    return ZD_clients, ZV, ZT, mean, std

def run_federated_logistic(ZD_clients, yD_clients, ZV, yV, ZT, yT, K, mu=0.0, rounds=ROUNDS, E=LOCAL_EPOCHS, lr=LR_RATE, seed=0):
    """
    Runs FedAvg (mu=0) or FedProx (mu>0) with full client participation.
    """
    d = ZD_clients[0].shape[1]
    rng = np.random.RandomState(seed + 500)
    w = np.zeros(d)
    b = 0.0
    ns = [len(y) for y in yD_clients]
    total_n = sum(ns)
    weights = [n / total_n for n in ns]
    
    comm_bytes_per_round = 2 * K * (d + 1) * 8  # uplink w,b and downlink w,b (64-bit float)
    total_comm = 0
    hist = []
    
    t0 = time.time()
    for r in range(1, rounds + 1):
        ws, bs = [], []
        for k in range(K):
            Xk = ZD_clients[k]
            yk = yD_clients[k]
            lw = w.copy()
            lb = b
            nk = len(yk)
            # Local gradient steps
            for _ in range(E):
                pred = sigmoid(Xk @ lw + lb)
                err = pred - yk
                # Proximal gradient term: mu * (lw - w)
                grad_w = (Xk.T @ err) / nk + mu * (lw - w)
                grad_b = np.mean(err)
                lw -= lr * grad_w
                lb -= lr * grad_b
            ws.append(lw)
            bs.append(lb)
            
        # Server aggregation
        w = sum(a * wk for a, wk in zip(weights, ws))
        b = sum(a * bk for a, bk in zip(weights, bs))
        total_comm += comm_bytes_per_round
        
        # Round evaluation
        sV = sigmoid(ZV @ w + b)
        sT = sigmoid(ZT @ w + b)
        hist.append({
            'round': r,
            'val_auc': metrics(yV, (sV > 0.5).astype(int), sV)['auc'],
            'test_auc': metrics(yT, (sT > 0.5).astype(int), sT)['auc'],
            'comm_bytes': total_comm
        })
        
    fit_time = time.time() - t0
    infer_t0 = time.time()
    final_sT = sigmoid(ZT @ w + b)
    infer_time = time.time() - infer_t0
    
    test_mets = metrics(yT, (final_sT > 0.5).astype(int), final_sT)
    val_mets = metrics(yV, (sigmoid(ZV @ w + b) > 0.5).astype(int), sigmoid(ZV @ w + b))
    
    return {
        'fit_time': fit_time,
        'infer_time': infer_time,
        'comm_bytes': total_comm,
        'test_metrics': test_mets,
        'val_metrics': val_mets,
        'hist': hist
    }

def run_federated_forests(XD_clients, yD_clients, XT, yT, cls, n_estimators=20, max_depth=None, min_samples_leaf=1, seed=0):
    """
    Federated Tree Ensemble: Each clinic trains a local forest, server aggregates via weighted soft voting.
    """
    K = len(XD_clients)
    ns = [len(y) for y in yD_clients]
    total_n = sum(ns)
    weights = [n / total_n for n in ns]
    
    t0 = time.time()
    models = []
    active_weights = []
    comm_bytes = 0
    import pickle
    
    for k in range(K):
        Xk = XD_clients[k]
        yk = yD_clients[k]
        # Only fit if client has multiple classes, otherwise single-class constant model
        unique_classes = np.unique(yk)
        m = cls(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed + k*10, n_jobs=-1)
        m.fit(Xk, yk)
        models.append(m)
        active_weights.append(weights[k])
        comm_bytes += len(pickle.dumps(m))
        
    fit_time = time.time() - t0
    active_weights = np.array(active_weights)
    active_weights /= active_weights.sum()
    
    # Soft vote inference
    infer_t0 = time.time()
    scores = np.zeros(len(XT))
    for m, wk in zip(models, active_weights):
        if len(m.classes_) == 1:
            cls_val = m.classes_[0]
            scores += wk * (1.0 if cls_val == 1 else 0.0)
        else:
            cls_idx = list(m.classes_).index(1)
            scores += wk * m.predict_proba(XT)[:, cls_idx]
    infer_time = time.time() - infer_t0
    
    test_mets = metrics(yT, (scores > 0.5).astype(int), scores)
    return {
        'fit_time': fit_time,
        'infer_time': infer_time,
        'comm_bytes': comm_bytes,
        'test_metrics': test_mets
    }

def run_experiment_a(n_seeds=NSEEDS):
    print(f"=== Starting Experiment A: Federated Learning under Multi-Patient Clinics ===")
    print(f"Seeds: {n_seeds}, K configs: {K_CONFIGS}, Allocation modes: {ALLOCATIONS}")
    
    run_records = []
    hist_records = []
    
    for seed in range(n_seeds):
        D, V, T, thr = make_sets(df, seed)
        dev_subs = np.sort(D['subject#'].unique())
        XD = D[FEATS].values
        XV = V[FEATS].values
        XT = T[FEATS].values
        yD = D.y.values
        yV = V.y.values
        yT = T.y.values
        
        for K in K_CONFIGS:
            for alloc in ALLOCATIONS:
                client_subs, client_indices = partition_dev_subjects(dev_subs, D, K, mode=alloc, seed=seed)
                XD_clients = [XD[idx] for idx in client_indices]
                yD_clients = [yD[idx] for idx in client_indices]
                
                # Check client class representation
                mono_count = sum(1 for yk in yD_clients if len(np.unique(yk)) < 2)
                
                # 1. Decentralized Standardization
                ZD_clients, ZV, ZT, _, _ = distributed_standardization(XD_clients, XV, XT)
                
                # 2. FedAvg and FedProx
                for mu in MU_LIST:
                    model_tag = 'FedAvg-LR' if mu == 0.0 else f'FedProx-LR (mu={mu})'
                    res_lr = run_federated_logistic(ZD_clients, yD_clients, ZV, yV, ZT, yT, K, mu=mu, seed=seed)
                    
                    row = {
                        'seed': seed,
                        'K_clients': K,
                        'allocation': alloc,
                        'mono_clients': mono_count,
                        'model': model_tag,
                        'paradigm': 'Federated-Linear',
                        'fit_s': res_lr['fit_time'],
                        'infer_s': res_lr['infer_time'],
                        'comm_bytes': res_lr['comm_bytes'],
                        **res_lr['test_metrics']
                    }
                    run_records.append(row)
                    
                    for h in res_lr['hist']:
                        hist_records.append({
                            'seed': seed,
                            'K_clients': K,
                            'allocation': alloc,
                            'model': model_tag,
                            **h
                        })
                
                # 3. Federated Forests (RF and ET)
                # RF
                res_rf = run_federated_forests(XD_clients, yD_clients, XT, yT, RandomForestClassifier, n_estimators=20, max_depth=None, min_samples_leaf=1, seed=seed)
                run_records.append({
                    'seed': seed,
                    'K_clients': K,
                    'allocation': alloc,
                    'mono_clients': mono_count,
                    'model': 'FedForest-RF',
                    'paradigm': 'Federated-Ensemble',
                    'fit_s': res_rf['fit_time'],
                    'infer_s': res_rf['infer_time'],
                    'comm_bytes': res_rf['comm_bytes'],
                    **res_rf['test_metrics']
                })
                
                # ET
                res_et = run_federated_forests(XD_clients, yD_clients, XT, yT, ExtraTreesClassifier, n_estimators=20, max_depth=None, min_samples_leaf=1, seed=seed)
                run_records.append({
                    'seed': seed,
                    'K_clients': K,
                    'allocation': alloc,
                    'mono_clients': mono_count,
                    'model': 'FedForest-ET',
                    'paradigm': 'Federated-Ensemble',
                    'fit_s': res_et['fit_time'],
                    'infer_s': res_et['infer_time'],
                    'comm_bytes': res_et['comm_bytes'],
                    **res_et['test_metrics']
                })
                
        print(f"Seed {seed}/{n_seeds-1} completed.", flush=True)
        
    df_runs = pd.DataFrame(run_records)
    df_hist = pd.DataFrame(hist_records)
    
    out_runs_path = os.path.join(R, "exp_a_federated_runs.csv")
    out_hist_path = os.path.join(R, "exp_a_fl_hist.csv")
    df_runs.to_csv(out_runs_path, index=False)
    df_hist.to_csv(out_hist_path, index=False)
    
    # Generate summary statistics (mean and 95% t-interval CI)
    from scipy import stats
    metrics_cols = ['acc', 'bacc', 'prec', 'rec', 'f1m', 'f1w', 'auc', 'prauc', 'mcc']
    def ci_str(vals):
        v = np.asarray(vals, float)
        m = v.mean()
        if len(v) > 1:
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        else:
            h = 0.0
        return f"{m:.3f} +/- {h:.3f}"
        
    summary_rows = []
    for (k, alloc, model), grp in df_runs.groupby(['K_clients', 'allocation', 'model']):
        s_row = {'K_clients': k, 'allocation': alloc, 'model': model, 'mono_clients_avg': grp.mono_clients.mean(), 'comm_bytes_avg': grp.comm_bytes.mean()}
        for mc in metrics_cols:
            s_row[mc + '_mean'] = grp[mc].mean()
            s_row[mc + '_ci'] = ci_str(grp[mc])
        summary_rows.append(s_row)
        
    df_summary = pd.DataFrame(summary_rows)
    out_summary_path = os.path.join(R, "exp_a_summary.csv")
    df_summary.to_csv(out_summary_path, index=False)
    
    print(f"Experiment A completed successfully!")
    print(f"Outputs written to:\n  {out_runs_path}\n  {out_hist_path}\n  {out_summary_path}")
    return df_summary

if __name__ == '__main__':
    # When run directly with sys.argv[1], can specify number of seeds (default 10)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_a(n)


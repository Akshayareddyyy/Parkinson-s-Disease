"""
Phase 4: Federated Learning Execution Suite
Implements strictly matched centralized vs federated comparisons:
  1. Centralized Logistic Regression vs. FedAvg-LR (K=26, 100% clients) vs. FedAvg-LR (50% clients)
  2. Centralized L2-Regularized LR vs. FedProx-LR (mu in {0.01, 0.1, 1.0})
  3. FedAvg-LR Multi-Center (K=5 Natural vs. K=5 Label-Balanced clinics)
  4. Centralized Random Forest & Extra Trees vs. Client-Ensembled Distributed Forest (CED-RF, CED-ET)
  5. Centralized Linear/Ridge Regression vs. FedAvg-Linear and FedAvg-Ridge Regression
Strict parity: 10 seeds, 26 dev / 7 val / 9 test, 100% development records, zero data leakage.
"""

import os, sys, time, json, pickle, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import make_sets
from phase4_common import (
    load_data, PREDICTORS, decentralized_standardization,
    get_natural_26_clients, get_multicenter_clients,
    eval_classification_metrics, eval_regression_metrics,
    compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

ROUNDS = 40
LOCAL_EPOCHS = 5
LR_RATE = 0.1

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

# -----------------------------------------------------------------------------
# FEDERATED LOGISTIC REGRESSION (FedAvg and FedProx)
# -----------------------------------------------------------------------------
def run_federated_logistic(client_data, ZV, yV, ZT, yT, ZD, yD, frac=1.0, mu=0.0,
                           rounds=ROUNDS, E=LOCAL_EPOCHS, lr=LR_RATE, seed=0):
    K = len(client_data)
    d = ZD.shape[1]
    rng = np.random.RandomState(seed + 100)
    
    w = np.zeros(d, dtype=float)
    b = 0.0
    
    history = []
    
    for r in range(1, rounds + 1):
        m_sel = max(1, int(round(frac * K)))
        selected_clients = rng.choice(K, m_sel, replace=False) if frac < 1.0 else list(range(K))
        
        local_weights = []
        local_biases = []
        local_sample_counts = []
        
        for cid in selected_clients:
            c = client_data[cid]
            Xc = c['X']
            yc = c['y']
            nk = len(yc)
            
            # Local initial weights
            lw = w.copy()
            lb = b
            
            # Local training epochs (batch gradient descent with proximal term)
            for _ in range(E):
                pred_prob = sigmoid(np.dot(Xc, lw) + lb)
                err = pred_prob - yc
                grad_w = np.dot(Xc.T, err) / nk + mu * (lw - w)
                grad_b = np.mean(err)
                
                lw -= lr * grad_w
                lb -= lr * grad_b
                
            local_weights.append(lw)
            local_biases.append(lb)
            local_sample_counts.append(nk)
            
        # Server Aggregation (FedAvg weighted by local sample size)
        total_samples = sum(local_sample_counts)
        weights_agg = np.zeros(d, dtype=float)
        bias_agg = 0.0
        
        for lw, lb, nk in zip(local_weights, local_biases, local_sample_counts):
            alpha = nk / total_samples
            weights_agg += alpha * lw
            bias_agg += alpha * lb
            
        w = weights_agg
        b = bias_agg
        
        # Round evaluation
        prob_V = sigmoid(np.dot(ZV, w) + b)
        prob_T = sigmoid(np.dot(ZT, w) + b)
        prob_D = sigmoid(np.dot(ZD, w) + b)
        
        loss_D = -float(np.mean(yD * np.log(prob_D + 1e-9) + (1.0 - yD) * np.log(1.0 - prob_D + 1e-9)))
        val_auc = float(eval_classification_metrics(yV, (prob_V > 0.5).astype(int), prob_V)['auc'])
        val_bacc = float(eval_classification_metrics(yV, (prob_V > 0.5).astype(int), prob_V)['bacc'])
        test_auc = float(eval_classification_metrics(yT, (prob_T > 0.5).astype(int), prob_T)['auc'])
        
        # Communication overhead (2 * |S| * (d + 1) * 8 bytes per round)
        comm_bytes_round = 2 * len(selected_clients) * (d + 1) * 8
        cum_comm_bytes = comm_bytes_round * r
        
        history.append({
            'seed': seed,
            'round': r,
            'frac': frac,
            'mu': mu,
            'train_loss': loss_D,
            'val_auc': val_auc,
            'val_bacc': val_bacc,
            'test_auc': test_auc,
            'comm_bytes': cum_comm_bytes
        })
        
    return w, b, history

# -----------------------------------------------------------------------------
# FEDERATED LINEAR / RIDGE REGRESSION
# -----------------------------------------------------------------------------
def run_federated_ridge(client_data, ZV, yV, ZT, yT, ZD, yD, alpha_l2=10.0,
                        rounds=ROUNDS, E=LOCAL_EPOCHS, lr=0.01, seed=0):
    K = len(client_data)
    d = ZD.shape[1]
    
    w = np.zeros(d, dtype=float)
    b = float(np.mean(yD))
    
    history = []
    
    for r in range(1, rounds + 1):
        local_weights = []
        local_biases = []
        local_sample_counts = []
        
        for cid in range(K):
            c = client_data[cid]
            Xc = c['X']
            yc = c['y_reg']
            nk = len(yc)
            
            lw = w.copy()
            lb = b
            
            for _ in range(E):
                pred = np.dot(Xc, lw) + lb
                err = pred - yc
                grad_w = np.dot(Xc.T, err) / nk + (alpha_l2 / nk) * lw
                grad_b = np.mean(err)
                
                lw -= lr * grad_w
                lb -= lr * grad_b
                
            local_weights.append(lw)
            local_biases.append(lb)
            local_sample_counts.append(nk)
            
        total_samples = sum(local_sample_counts)
        w = sum((nk / total_samples) * lw for lw, nk in zip(local_weights, local_sample_counts))
        b = sum((nk / total_samples) * lb for lb, nk in zip(local_biases, local_sample_counts))
        
        pred_V = np.dot(ZV, w) + b
        pred_T = np.dot(ZT, w) + b
        
        val_mae = float(eval_regression_metrics(yV, pred_V)['mae'])
        test_mae = float(eval_regression_metrics(yT, pred_T)['mae'])
        
        comm_bytes_round = 2 * K * (d + 1) * 8
        history.append({
            'seed': seed,
            'round': r,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'comm_bytes': comm_bytes_round * r
        })
        
    return w, b, history

# -----------------------------------------------------------------------------
# CLIENT-ENSEMBLED DISTRIBUTED FOREST (CED-Forest)
# Transparently disclosed as server soft-voting of local forests (NOT FedAvg)
# -----------------------------------------------------------------------------
def run_ced_forest(client_data, XT, yT, model_type='RF', n_trees_per_client=15, seed=0):
    cls = RandomForestClassifier if model_type == 'RF' else ExtraTreesClassifier
    local_models = []
    local_counts = []
    
    t0 = time.time()
    for cid, c in enumerate(client_data):
        Xc = c['X_raw']
        yc = c['y']
        
        # Check if client has both classes or single class
        unique_classes = np.unique(yc)
        if len(unique_classes) < 2:
            # Client has only one class; store dummy predictor
            single_class = int(unique_classes[0])
            local_models.append(('single_class', single_class))
        else:
            m = cls(n_estimators=n_trees_per_client, max_depth=8, min_samples_leaf=2,
                    random_state=seed + cid, n_jobs=1)
            m.fit(Xc, yc)
            local_models.append(('model', m))
        local_counts.append(len(yc))
        
    fit_s = time.time() - t0
    total_samples = sum(local_counts)
    
    # Server-side soft-voting ensemble
    prob_T = np.zeros(len(yT), dtype=float)
    for (m_type, obj), nk in zip(local_models, local_counts):
        weight = nk / total_samples
        if m_type == 'single_class':
            p_pos = float(obj)
            prob_T += weight * p_pos
        else:
            probs = obj.predict_proba(XT)
            classes = list(obj.classes_)
            if 1 in classes:
                p_pos = probs[:, classes.index(1)]
            else:
                p_pos = np.zeros(len(XT))
            prob_T += weight * p_pos
            
    # Serialize to measure transmission bytes
    comm_bytes = sum(len(pickle.dumps(obj)) for _, obj in local_models)
    pred_T = (prob_T > 0.5).astype(int)
    mets = eval_classification_metrics(yT, pred_T, prob_T)
    return mets, fit_s, comm_bytes

# -----------------------------------------------------------------------------
# MASTER PHASE 4 RUNNER
# -----------------------------------------------------------------------------
def run_phase4(n_seeds=10):
    print("=" * 80)
    print("PHASE 4: FEDERATED LEARNING UNDER STRICT FAIR COMPARISON PROTOCOL")
    print(f"Seeds: {n_seeds}, Tasks: Binary Severity Classification & Continuous Progression")
    print("Zero Data Leakage: Decentralized Sufficient Statistics Standardization")
    print("=" * 80)
    
    all_runs = []
    all_history = []
    all_clients = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        XD = D[PREDICTORS].values
        XV = V[PREDICTORS].values
        XT = T[PREDICTORS].values
        yD = D.y.values
        yV = V.y.values
        yT = T.y.values
        yD_reg = D['motor_UPDRS'].values
        yV_reg = V['motor_UPDRS'].values
        yT_reg = T['motor_UPDRS'].values
        
        # ---------------------------------------------------------------------
        # TOPOLOGY 1: 26 NATURAL CLIENTS (1 PER PATIENT)
        # ---------------------------------------------------------------------
        clients_26 = get_natural_26_clients(D)
        if seed == 0:
            for c in clients_26:
                all_clients.append({
                    'seed': seed,
                    'topology': '26_natural_edge',
                    'client_id': c['client_id'],
                    'subject_id': c['subject_id'],
                    'n_records': c['n_records'],
                    'pos_rate': c['pos_rate'],
                    'mean_updrs': c['mean_updrs']
                })
                
        client_X_list = [D.iloc[c['indices']][PREDICTORS].values for c in clients_26]
        
        # Decentralized Sufficient Statistics Standardization
        ZD_clients, ZV, ZT, _, _ = decentralized_standardization(client_X_list, XV, XT)
        ZD = np.vstack(ZD_clients)
        
        client_data_26 = []
        for cid, c in enumerate(clients_26):
            client_data_26.append({
                'client_id': cid,
                'X': ZD_clients[cid],
                'X_raw': client_X_list[cid],
                'y': D.iloc[c['indices']].y.values,
                'y_reg': D.iloc[c['indices']]['motor_UPDRS'].values
            })
            
        # =====================================================================
        # TASK 1: BINARY MOTOR SEVERITY CLASSIFICATION
        # =====================================================================
        
        # Matched Pair 1A: Centralized Logistic Regression
        t0 = time.time()
        sc_cent = StandardScaler().fit(XD)
        m_cent_lr = LogisticRegression(max_iter=2000, random_state=seed).fit(sc_cent.transform(XD), yD)
        fit_s = time.time() - t0
        prob_T_cent = m_cent_lr.predict_proba(sc_cent.transform(XT))[:, 1]
        mets_cent_lr = eval_classification_metrics(yT, (prob_T_cent > 0.5).astype(int), prob_T_cent)
        all_runs.append({
            'seed': seed, 'task': 'Classification', 'model': 'Centralized Logistic Regression',
            'methodology': 'Centralized', 'topology': 'N/A', 'n_clients': 1, 'rounds': 0,
            'comm_bytes': 0, 'fit_s': fit_s, **mets_cent_lr
        })
        
        # Matched Pair 1B: FedAvg-LR (K=26, 100% participation)
        t0 = time.time()
        w, b, hist_fedavg = run_federated_logistic(client_data_26, ZV, yV, ZT, yT, ZD, yD,
                                                  frac=1.0, mu=0.0, seed=seed)
        fit_s = time.time() - t0
        prob_T_fed = sigmoid(np.dot(ZT, w) + b)
        mets_fedavg = eval_classification_metrics(yT, (prob_T_fed > 0.5).astype(int), prob_T_fed)
        all_runs.append({
            'seed': seed, 'task': 'Classification', 'model': 'FedAvg-LR (100% clients)',
            'methodology': 'Federated', 'topology': '26_edge_devices', 'n_clients': 26, 'rounds': ROUNDS,
            'comm_bytes': hist_fedavg[-1]['comm_bytes'], 'fit_s': fit_s, **mets_fedavg
        })
        for h in hist_fedavg:
            h['model'] = 'FedAvg-LR (100% clients)'
            all_history.append(h)
            
        # Matched Pair 1C: FedAvg-LR (K=26, 50% participation)
        t0 = time.time()
        w, b, hist_fedavg_50 = run_federated_logistic(client_data_26, ZV, yV, ZT, yT, ZD, yD,
                                                     frac=0.5, mu=0.0, seed=seed)
        fit_s = time.time() - t0
        prob_T_fed50 = sigmoid(np.dot(ZT, w) + b)
        mets_fedavg_50 = eval_classification_metrics(yT, (prob_T_fed50 > 0.5).astype(int), prob_T_fed50)
        all_runs.append({
            'seed': seed, 'task': 'Classification', 'model': 'FedAvg-LR (50% clients)',
            'methodology': 'Federated', 'topology': '26_edge_devices', 'n_clients': 13, 'rounds': ROUNDS,
            'comm_bytes': hist_fedavg_50[-1]['comm_bytes'], 'fit_s': fit_s, **mets_fedavg_50
        })
        for h in hist_fedavg_50:
            h['model'] = 'FedAvg-LR (50% clients)'
            all_history.append(h)
            
        # Matched Pair 2: FedProx-LR (mu in {0.01, 0.1, 1.0})
        for mu_val in [0.01, 0.1, 1.0]:
            t0 = time.time()
            w, b, hist_fedprox = run_federated_logistic(client_data_26, ZV, yV, ZT, yT, ZD, yD,
                                                       frac=1.0, mu=mu_val, seed=seed)
            fit_s = time.time() - t0
            prob_T_prox = sigmoid(np.dot(ZT, w) + b)
            mets_fedprox = eval_classification_metrics(yT, (prob_T_prox > 0.5).astype(int), prob_T_prox)
            all_runs.append({
                'seed': seed, 'task': 'Classification', 'model': f'FedProx-LR (mu={mu_val})',
                'methodology': 'Federated', 'topology': '26_edge_devices', 'n_clients': 26, 'rounds': ROUNDS,
                'comm_bytes': hist_fedprox[-1]['comm_bytes'], 'fit_s': fit_s, **mets_fedprox
            })
            for h in hist_fedprox:
                h['model'] = f'FedProx-LR (mu={mu_val})'
                all_history.append(h)
                
        # Matched Pair 3: Multi-Center Clinics (K=5 Natural vs. K=5 Balanced)
        for alloc_mode in ['natural', 'label_balanced']:
            clients_5 = get_multicenter_clients(D, K=5, mode=alloc_mode, seed=seed)
            c5_X_list = [D.iloc[c['indices']][PREDICTORS].values for c in clients_5]
            ZD5_clients, ZV5, ZT5, _, _ = decentralized_standardization(c5_X_list, XV, XT)
            ZD5 = np.vstack(ZD5_clients)
            
            c5_data = []
            for cid, c in enumerate(clients_5):
                c5_data.append({
                    'client_id': cid,
                    'X': ZD5_clients[cid],
                    'y': D.iloc[c['indices']].y.values,
                    'y_reg': D.iloc[c['indices']]['motor_UPDRS'].values
                })
                
            t0 = time.time()
            w, b, hist_c5 = run_federated_logistic(c5_data, ZV5, yV, ZT5, yT, ZD5, yD,
                                                  frac=1.0, mu=0.0, seed=seed)
            fit_s = time.time() - t0
            prob_T_c5 = sigmoid(np.dot(ZT5, w) + b)
            mets_c5 = eval_classification_metrics(yT, (prob_T_c5 > 0.5).astype(int), prob_T_c5)
            all_runs.append({
                'seed': seed, 'task': 'Classification', 'model': f'FedAvg-LR (Multi-Center K=5, {alloc_mode})',
                'methodology': 'Federated', 'topology': f'5_clinics_{alloc_mode}', 'n_clients': 5, 'rounds': ROUNDS,
                'comm_bytes': hist_c5[-1]['comm_bytes'], 'fit_s': fit_s, **mets_c5
            })
            
        # Matched Pair 4: Centralized Tree Ensembles vs. Client-Ensembled Distributed Forest (CED-Forest)
        # 4A: Random Forest
        t0 = time.time()
        m_cent_rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                           random_state=seed, n_jobs=1).fit(XD, yD)
        fit_s = time.time() - t0
        prob_T_rf = m_cent_rf.predict_proba(XT)[:, 1]
        mets_cent_rf = eval_classification_metrics(yT, (prob_T_rf > 0.5).astype(int), prob_T_rf)
        all_runs.append({
            'seed': seed, 'task': 'Classification', 'model': 'Centralized Random Forest',
            'methodology': 'Centralized', 'topology': 'N/A', 'n_clients': 1, 'rounds': 0,
            'comm_bytes': 0, 'fit_s': fit_s, **mets_cent_rf
        })
        
        mets_ced_rf, fit_s_ced_rf, bytes_ced_rf = run_ced_forest(client_data_26, XT, yT, model_type='RF', seed=seed)
        all_runs.append({
            'seed': seed, 'task': 'Classification', 'model': 'CED-Forest (Random Forest)',
            'methodology': 'Federated (Distributed Ensemble)', 'topology': '26_edge_devices', 'n_clients': 26, 'rounds': 1,
            'comm_bytes': bytes_ced_rf, 'fit_s': fit_s_ced_rf, **mets_ced_rf
        })
        
        # 4B: Extra Trees
        t0 = time.time()
        m_cent_et = ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                         random_state=seed, n_jobs=1).fit(XD, yD)
        fit_s = time.time() - t0
        prob_T_et = m_cent_et.predict_proba(XT)[:, 1]
        mets_cent_et = eval_classification_metrics(yT, (prob_T_et > 0.5).astype(int), prob_T_et)
        all_runs.append({
            'seed': seed, 'task': 'Classification', 'model': 'Centralized Extra Trees',
            'methodology': 'Centralized', 'topology': 'N/A', 'n_clients': 1, 'rounds': 0,
            'comm_bytes': 0, 'fit_s': fit_s, **mets_cent_et
        })
        
        mets_ced_et, fit_s_ced_et, bytes_ced_et = run_ced_forest(client_data_26, XT, yT, model_type='ET', seed=seed)
        all_runs.append({
            'seed': seed, 'task': 'Classification', 'model': 'CED-Forest (Extra Trees)',
            'methodology': 'Federated (Distributed Ensemble)', 'topology': '26_edge_devices', 'n_clients': 26, 'rounds': 1,
            'comm_bytes': bytes_ced_et, 'fit_s': fit_s_ced_et, **mets_ced_et
        })
        
        # =====================================================================
        # TASK 2: CONTINUOUS MOTOR PROGRESSION REGRESSION
        # =====================================================================
        
        # Matched Pair 5: Centralized Ridge vs. FedAvg-Ridge
        t0 = time.time()
        m_cent_ridge = Ridge(alpha=10.0, random_state=seed).fit(sc_cent.transform(XD), yD_reg)
        fit_s = time.time() - t0
        pred_T_ridge = m_cent_ridge.predict(sc_cent.transform(XT))
        mets_cent_ridge = eval_regression_metrics(yT_reg, pred_T_ridge)
        sub_ridge = compute_per_subject_regression_metrics(T, pred_T_ridge)
        all_runs.append({
            'seed': seed, 'task': 'Regression', 'model': 'Centralized Ridge Regression',
            'methodology': 'Centralized', 'topology': 'N/A', 'n_clients': 1, 'rounds': 0,
            'comm_bytes': 0, 'fit_s': fit_s, **mets_cent_ridge, **sub_ridge
        })
        
        t0 = time.time()
        w_reg, b_reg, hist_ridge = run_federated_ridge(client_data_26, ZV, yV_reg, ZT, yT_reg,
                                                      ZD, yD_reg, alpha_l2=10.0, seed=seed)
        fit_s = time.time() - t0
        pred_T_fed_ridge = np.dot(ZT, w_reg) + b_reg
        mets_fed_ridge = eval_regression_metrics(yT_reg, pred_T_fed_ridge)
        sub_fed_ridge = compute_per_subject_regression_metrics(T, pred_T_fed_ridge)
        all_runs.append({
            'seed': seed, 'task': 'Regression', 'model': 'FedAvg-Ridge (K=26)',
            'methodology': 'Federated', 'topology': '26_edge_devices', 'n_clients': 26, 'rounds': ROUNDS,
            'comm_bytes': hist_ridge[-1]['comm_bytes'], 'fit_s': fit_s, **mets_fed_ridge, **sub_fed_ridge
        })
        
        # Matched Pair 6: Centralized Linear Regression vs. FedAvg-Linear
        t0 = time.time()
        m_cent_lin = LinearRegression().fit(sc_cent.transform(XD), yD_reg)
        fit_s = time.time() - t0
        pred_T_lin = m_cent_lin.predict(sc_cent.transform(XT))
        mets_cent_lin = eval_regression_metrics(yT_reg, pred_T_lin)
        sub_lin = compute_per_subject_regression_metrics(T, pred_T_lin)
        all_runs.append({
            'seed': seed, 'task': 'Regression', 'model': 'Centralized Linear Regression',
            'methodology': 'Centralized', 'topology': 'N/A', 'n_clients': 1, 'rounds': 0,
            'comm_bytes': 0, 'fit_s': fit_s, **mets_cent_lin, **sub_lin
        })
        
        t0 = time.time()
        w_lin, b_lin, hist_lin = run_federated_ridge(client_data_26, ZV, yV_reg, ZT, yT_reg,
                                                    ZD, yD_reg, alpha_l2=0.0, seed=seed)
        fit_s = time.time() - t0
        pred_T_fed_lin = np.dot(ZT, w_lin) + b_lin
        mets_fed_lin = eval_regression_metrics(yT_reg, pred_T_fed_lin)
        sub_fed_lin = compute_per_subject_regression_metrics(T, pred_T_fed_lin)
        all_runs.append({
            'seed': seed, 'task': 'Regression', 'model': 'FedAvg-Linear (K=26)',
            'methodology': 'Federated', 'topology': '26_edge_devices', 'n_clients': 26, 'rounds': ROUNDS,
            'comm_bytes': hist_lin[-1]['comm_bytes'], 'fit_s': fit_s, **mets_fed_lin, **sub_fed_lin
        })
        
        print(f"Phase 4 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_runs = pd.DataFrame(all_runs)
    df_hist = pd.DataFrame(all_history)
    df_clients = pd.DataFrame(all_clients)
    
    out_runs = os.path.join(R, "phase4_federated_runs.csv")
    out_hist = os.path.join(R, "phase4_fl_convergence_history.csv")
    out_clients = os.path.join(R, "phase4_client_summary.csv")
    out_sum = os.path.join(R, "phase4_federated_summary.csv")
    
    df_runs.to_csv(out_runs, index=False)
    df_hist.to_csv(out_hist, index=False)
    df_clients.to_csv(out_clients, index=False)
    
    # Consolidated summary
    sum_rows = []
    for (task, model, meth, top), grp in df_runs.groupby(['task', 'model', 'methodology', 'topology']):
        r_entry = {
            'task': task,
            'model': model,
            'methodology': meth,
            'topology': top,
            'n_clients_avg': int(grp['n_clients'].mean()),
            'rounds': int(grp['rounds'].mean()),
            'comm_bytes_avg': int(grp['comm_bytes'].mean()),
            'fit_s_avg': grp['fit_s'].mean(),
        }
        if task == 'Classification':
            r_entry.update({
                'roc_auc_mean': grp['auc'].mean(),
                'roc_auc_std': grp['auc'].std(),
                'prauc_mean': grp['prauc'].mean(),
                'bacc_mean': grp['bacc'].mean(),
                'bacc_std': grp['bacc'].std(),
                'mcc_mean': grp['mcc'].mean(),
                'mcc_std': grp['mcc'].std(),
                'f1m_mean': grp['f1m'].mean(),
                'f1m_std': grp['f1m'].std(),
                'sens_mean': grp['sens'].mean(),
                'spec_mean': grp['spec'].mean()
            })
        else:
            r_entry.update({
                'mae_mean': grp['mae'].mean(),
                'mae_std': grp['mae'].std(),
                'rmse_mean': grp['rmse'].mean(),
                'rmse_std': grp['rmse'].std(),
                'r2_mean': grp['r2'].mean(),
                'r2_std': grp['r2'].std(),
                'pearson_r_mean': grp['pearson_r'].mean(),
                'spearman_rho_mean': grp['spearman_rho'].mean(),
                'mean_subj_mae_mean': grp['mean_subj_mae'].mean(),
                'mean_subj_mae_std': grp['mean_subj_mae'].std()
            })
        sum_rows.append(r_entry)
        
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  {out_runs}\n  {out_sum}\n  {out_hist}\n  {out_clients}")
    print("=" * 80)
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_phase4(ns)


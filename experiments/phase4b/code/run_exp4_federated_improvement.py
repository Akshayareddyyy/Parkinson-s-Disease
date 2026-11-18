"""
Phase 4B - Experiment 4: Federated Learning Improvement & Stabilization
Evaluates:
  1. Standard FedAvg-LR (Edge-26, constant lr=0.1)
  2. Cosine Annealing FedAvg-LR (Edge-26, lr in [0.01, 0.15])
  3. Globally Class-Weighted FedAvg-LR (Global inverse class prevalence, resolves single-class starvation)
  4. Safe Local Class-Weighted FedAvg-LR (Local class weighting with global fallback for single-class clients)
  5. Validation-Tuned FedProx-LR (mu in {0.0, 0.01, 0.1, 1.0, 5.0, 10.0} tuned on 7 server validation subjects)
  6. Clinic-5 Topology FedAvg-LR (5 Regional Clinics with multi-subject clustering)
  7. FedAvg-Ridge Regression (Continuous regression under Edge-26 and Clinic-5 topologies)
  8. Matched Centralized Controls (Logistic Regression and Ridge Regression)

All federated models use decentralized sufficient statistics standardization (zero test leakage).
Reports complete classification metrics (including Brier score and Log Loss) and regression metrics.
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from common import make_sets
from phase4b_common import (
    load_data, PREDICTORS, decentralized_standardization,
    get_natural_26_clients, get_multicenter_clients,
    eval_classification_metrics, eval_regression_metrics,
    compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def get_cosine_lr(round_idx, total_rounds, lr_max=0.15, lr_min=0.01):
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * round_idx / total_rounds))

# -----------------------------------------------------------------------------
# FEDERATED LOGISTIC REGRESSION ENGINE
# -----------------------------------------------------------------------------
def train_fed_lr(client_data, ZV, yV, ZT, yT, rounds=50, epochs=5,
                 lr_schedule='constant', base_lr=0.1, mu=0.0,
                 class_weighting='none', global_w0=1.0, global_w1=1.0):
    """
    Simulates federated training of logistic regression with:
      - FedAvg (mu=0) or FedProx (mu>0)
      - Constant or Cosine Annealing learning rate
      - Class weighting: 'none', 'global', or 'safe_local'
    """
    d = client_data[0]['X'].shape[1]
    w = np.zeros(d, dtype=float)
    b = 0.0
    K = len(client_data)
    
    # Calculate global weights if not provided
    if class_weighting in ['global', 'safe_local']:
        all_y = np.concatenate([c['y'] for c in client_data])
        n_pos = np.sum(all_y == 1)
        n_neg = np.sum(all_y == 0)
        total_n = len(all_y)
        w0_glob = total_n / (2.0 * max(1, n_neg))
        w1_glob = total_n / (2.0 * max(1, n_pos))
    else:
        w0_glob, w1_glob = 1.0, 1.0
        
    val_auc_history = []
    
    for r in range(rounds):
        lr = get_cosine_lr(r, rounds) if lr_schedule == 'cosine' else base_lr
        
        local_weights = []
        local_biases = []
        local_agg_weights = []
        
        for cid in range(K):
            c = client_data[cid]
            Xc = c['X']
            yc = c['y']
            nk = len(yc)
            
            # Determine sample weights for local optimization
            if class_weighting == 'none':
                sw = np.ones(nk, dtype=float)
                agg_w = float(nk)
            elif class_weighting == 'global':
                sw = np.where(yc == 1, w1_glob, w0_glob)
                agg_w = float(np.sum(sw))
            elif class_weighting == 'safe_local':
                n_c_pos = np.sum(yc == 1)
                n_c_neg = np.sum(yc == 0)
                if n_c_pos > 0 and n_c_neg > 0:
                    # Bi-class client: local inverse frequency
                    loc_w0 = nk / (2.0 * n_c_neg)
                    loc_w1 = nk / (2.0 * n_c_pos)
                    sw = np.where(yc == 1, loc_w1, loc_w0)
                else:
                    # Single-class client: fallback to global weights
                    sw = np.where(yc == 1, w1_glob, w0_glob)
                agg_w = float(np.sum(sw))
            else:
                raise ValueError(f"Unknown class_weighting: {class_weighting}")
                
            sum_sw = np.sum(sw)
            
            # Local initial parameters
            lw = w.copy()
            lb = b
            
            # Local epochs
            for _ in range(epochs):
                prob = sigmoid(np.dot(Xc, lw) + lb)
                err = prob - yc
                weighted_err = sw * err
                
                grad_w = np.dot(Xc.T, weighted_err) / sum_sw + mu * (lw - w)
                grad_b = np.sum(weighted_err) / sum_sw
                
                lw -= lr * grad_w
                lb -= lr * grad_b
                
            local_weights.append(lw)
            local_biases.append(lb)
            local_agg_weights.append(agg_w)
            
        # Server Aggregation
        total_agg_weight = sum(local_agg_weights)
        w_new = np.zeros(d, dtype=float)
        b_new = 0.0
        
        for lw, lb, aw in zip(local_weights, local_biases, local_agg_weights):
            alpha = aw / total_agg_weight
            w_new += alpha * lw
            b_new += alpha * lb
            
        w = w_new
        b = b_new
        
        # Validation AUC check
        prob_V = sigmoid(np.dot(ZV, w) + b)
        val_auc = eval_classification_metrics(yV, (prob_V > 0.5).astype(int), prob_V)['auc']
        val_auc_history.append((val_auc, w.copy(), b))
        
    # Evaluate final model on Test set
    prob_T = sigmoid(np.dot(ZT, w) + b)
    pred_T = (prob_T > 0.5).astype(int)
    test_mets = eval_classification_metrics(yT, pred_T, prob_T)
    
    # Best validation round metric
    best_val_auc = max(h[0] for h in val_auc_history)
    
    return w, b, test_mets, best_val_auc

# -----------------------------------------------------------------------------
# FEDERATED RIDGE REGRESSION ENGINE
# -----------------------------------------------------------------------------
def train_fed_ridge(client_data, ZV, yV, ZT, yT, rounds=50, epochs=5,
                    lr_schedule='cosine', base_lr=0.01, alpha_reg=10.0):
    """
    Simulates federated continuous regression (Ridge L2 regularization).
    """
    d = client_data[0]['X'].shape[1]
    w = np.zeros(d, dtype=float)
    b = float(np.mean([np.mean(c['y']) for c in client_data]))
    K = len(client_data)
    
    val_mae_history = []
    
    for r in range(rounds):
        lr = get_cosine_lr(r, rounds, lr_max=0.015, lr_min=0.001) if lr_schedule == 'cosine' else base_lr
        
        local_weights = []
        local_biases = []
        local_sample_counts = []
        
        for cid in range(K):
            c = client_data[cid]
            Xc = c['X']
            yc = c['y']
            nk = len(yc)
            
            lw = w.copy()
            lb = b
            
            for _ in range(epochs):
                pred = np.dot(Xc, lw) + lb
                err = pred - yc
                grad_w = np.dot(Xc.T, err) / nk + (alpha_reg / nk) * lw
                grad_b = np.mean(err)
                
                # Safe gradient step
                lw -= lr * np.clip(grad_w, -50.0, 50.0)
                lb -= lr * np.clip(grad_b, -50.0, 50.0)
                
            local_weights.append(lw)
            local_biases.append(lb)
            local_sample_counts.append(nk)
            
        total_samples = sum(local_sample_counts)
        w_new = np.zeros(d, dtype=float)
        b_new = 0.0
        for lw, lb, nk in zip(local_weights, local_biases, local_sample_counts):
            alpha = nk / total_samples
            w_new += alpha * lw
            b_new += alpha * lb
            
        w = w_new
        b = b_new
        
        pred_V = np.dot(ZV, w) + b
        val_mae = eval_regression_metrics(yV, pred_V)['mae']
        val_mae_history.append((val_mae, w.copy(), b))
        
    pred_T = np.dot(ZT, w) + b
    test_mets = eval_regression_metrics(yT, pred_T)
    best_val_mae = min(h[0] for h in val_mae_history)
    
    return w, b, test_mets, pred_T, best_val_mae

# -----------------------------------------------------------------------------
# MAIN EXPERIMENT EXECUTION
# -----------------------------------------------------------------------------
def run_experiment_4(n_seeds=10):
    print("=" * 80)
    print("PHASE 4B - EXPERIMENT 4: FEDERATED LEARNING IMPROVEMENT & STABILIZATION")
    print("Topologies: Edge-26 (1 patient/client) & Clinic-5 (Regional Centers)")
    print("Algorithms: Class-Weighted FedAvg, Cosine Schedule, Validation-Tuned FedProx, FedAvg-Ridge")
    print("Zero Test Leakage: Decentralized Sufficient Statistics Preprocessing")
    print("=" * 80)
    
    exp4_runs = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        yD_cls = D.y.values
        yV_cls = V.y.values
        yT_cls = T.y.values
        
        yD_reg = D['motor_UPDRS'].values
        yV_reg = V['motor_UPDRS'].values
        yT_reg = T['motor_UPDRS'].values
        
        # --- Topology 1: Edge-26 ---
        edge26_clients = get_natural_26_clients(D)
        client_X_list_26 = [D.loc[c['indices'], PREDICTORS].values for c in edge26_clients]
        XV_raw = V[PREDICTORS].values
        XT_raw = T[PREDICTORS].values
        
        ZD_list_26, ZV_26, ZT_26, _, _ = decentralized_standardization(client_X_list_26, XV_raw, XT_raw)
        
        client_data_cls_26 = []
        client_data_reg_26 = []
        for i, c in enumerate(edge26_clients):
            client_data_cls_26.append({'X': ZD_list_26[i], 'y': D.loc[c['indices'], 'y'].values})
            client_data_reg_26.append({'X': ZD_list_26[i], 'y': D.loc[c['indices'], 'motor_UPDRS'].values})
            
        # --- Topology 2: Clinic-5 (Regional) ---
        clinic5_clients = get_multicenter_clients(D, K=5, mode='natural', seed=seed)
        client_X_list_5 = [D.loc[c['indices'], PREDICTORS].values for c in clinic5_clients]
        ZD_list_5, ZV_5, ZT_5, _, _ = decentralized_standardization(client_X_list_5, XV_raw, XT_raw)
        
        client_data_cls_5 = []
        client_data_reg_5 = []
        for i, c in enumerate(clinic5_clients):
            client_data_cls_5.append({'X': ZD_list_5[i], 'y': D.loc[c['indices'], 'y'].values})
            client_data_reg_5.append({'X': ZD_list_5[i], 'y': D.loc[c['indices'], 'motor_UPDRS'].values})
            
        # ---------------------------------------------------------------------
        # 1. Centralized Classification Controls
        # ---------------------------------------------------------------------
        # A. Centralized LR Standard
        t0 = time.time()
        ZD_all = np.concatenate(ZD_list_26, axis=0)
        yD_all_cls = np.concatenate([c['y'] for c in client_data_cls_26], axis=0)
        lr_std = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        lr_std.fit(ZD_all, yD_all_cls)
        fit_s = time.time() - t0
        prob_T = lr_std.predict_proba(ZT_26)[:, 1]
        mets = eval_classification_metrics(yT_cls, (prob_T > 0.5).astype(int), prob_T)
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Centralized',
            'model': 'Centralized LR (Standard)', 'mu': 0.0,
            'lr_schedule': 'L-BFGS', 'class_weighting': 'none',
            'fit_s': fit_s, 'best_val_metric': np.nan, **mets
        })
        
        # B. Centralized LR Balanced
        t0 = time.time()
        lr_bal = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=seed)
        lr_bal.fit(ZD_all, yD_all_cls)
        fit_s = time.time() - t0
        prob_T = lr_bal.predict_proba(ZT_26)[:, 1]
        mets = eval_classification_metrics(yT_cls, (prob_T > 0.5).astype(int), prob_T)
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Centralized',
            'model': 'Centralized LR (Balanced)', 'mu': 0.0,
            'lr_schedule': 'L-BFGS', 'class_weighting': 'balanced',
            'fit_s': fit_s, 'best_val_metric': np.nan, **mets
        })
        
        # ---------------------------------------------------------------------
        # 2. Federated Classification Algorithms (Edge-26)
        # ---------------------------------------------------------------------
        # A. Standard FedAvg-LR (Baseline Reference)
        t0 = time.time()
        _, _, mets, bval = train_fed_lr(
            client_data_cls_26, ZV_26, yV_cls, ZT_26, yT_cls,
            rounds=50, epochs=5, lr_schedule='constant', base_lr=0.1,
            mu=0.0, class_weighting='none'
        )
        fit_s = time.time() - t0
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Edge-26',
            'model': 'FedAvg-LR (Standard Baseline)', 'mu': 0.0,
            'lr_schedule': 'constant', 'class_weighting': 'none',
            'fit_s': fit_s, 'best_val_metric': bval, **mets
        })
        
        # B. Cosine Annealing FedAvg-LR
        t0 = time.time()
        _, _, mets, bval = train_fed_lr(
            client_data_cls_26, ZV_26, yV_cls, ZT_26, yT_cls,
            rounds=50, epochs=5, lr_schedule='cosine',
            mu=0.0, class_weighting='none'
        )
        fit_s = time.time() - t0
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Edge-26',
            'model': 'FedAvg-LR (Cosine Schedule)', 'mu': 0.0,
            'lr_schedule': 'cosine', 'class_weighting': 'none',
            'fit_s': fit_s, 'best_val_metric': bval, **mets
        })
        
        # C. Globally Class-Weighted FedAvg-LR
        t0 = time.time()
        _, _, mets, bval = train_fed_lr(
            client_data_cls_26, ZV_26, yV_cls, ZT_26, yT_cls,
            rounds=50, epochs=5, lr_schedule='cosine',
            mu=0.0, class_weighting='global'
        )
        fit_s = time.time() - t0
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Edge-26',
            'model': 'FedAvg-LR (Global Class-Weighted)', 'mu': 0.0,
            'lr_schedule': 'cosine', 'class_weighting': 'global',
            'fit_s': fit_s, 'best_val_metric': bval, **mets
        })
        
        # D. Safe Local Class-Weighted FedAvg-LR
        t0 = time.time()
        _, _, mets, bval = train_fed_lr(
            client_data_cls_26, ZV_26, yV_cls, ZT_26, yT_cls,
            rounds=50, epochs=5, lr_schedule='cosine',
            mu=0.0, class_weighting='safe_local'
        )
        fit_s = time.time() - t0
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Edge-26',
            'model': 'FedAvg-LR (Safe Local Class-Weighted)', 'mu': 0.0,
            'lr_schedule': 'cosine', 'class_weighting': 'safe_local',
            'fit_s': fit_s, 'best_val_metric': bval, **mets
        })
        
        # E. Validation-Tuned FedProx-LR (mu in {0.0, 0.01, 0.1, 1.0, 5.0, 10.0})
        t0 = time.time()
        candidate_mus = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]
        best_mu = 0.0
        best_val_auc = -1.0
        best_test_mets = None
        
        for mu_cand in candidate_mus:
            w_cand, b_cand, mets_cand, val_auc_cand = train_fed_lr(
                client_data_cls_26, ZV_26, yV_cls, ZT_26, yT_cls,
                rounds=50, epochs=5, lr_schedule='cosine',
                mu=mu_cand, class_weighting='global'
            )
            if val_auc_cand > best_val_auc:
                best_val_auc = val_auc_cand
                best_mu = mu_cand
                best_test_mets = mets_cand
                
        fit_s = time.time() - t0
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Edge-26',
            'model': 'FedProx-LR (Validation-Tuned)', 'mu': best_mu,
            'lr_schedule': 'cosine', 'class_weighting': 'global',
            'fit_s': fit_s, 'best_val_metric': best_val_auc, **best_test_mets
        })
        
        # F. Clinic-5 Topology (Regional Centers) FedAvg-LR
        t0 = time.time()
        _, _, mets, bval = train_fed_lr(
            client_data_cls_5, ZV_5, yV_cls, ZT_5, yT_cls,
            rounds=50, epochs=5, lr_schedule='cosine',
            mu=0.0, class_weighting='global'
        )
        fit_s = time.time() - t0
        exp4_runs.append({
            'seed': seed, 'task': 'Classification', 'topology': 'Clinic-5',
            'model': 'FedAvg-LR (Clinic-5 Regional)', 'mu': 0.0,
            'lr_schedule': 'cosine', 'class_weighting': 'global',
            'fit_s': fit_s, 'best_val_metric': bval, **mets
        })
        
        # ---------------------------------------------------------------------
        # 3. Federated & Centralized Continuous Regression
        # ---------------------------------------------------------------------
        # A. Centralized Ridge Regression
        t0 = time.time()
        yD_all_reg = np.concatenate([c['y'] for c in client_data_reg_26], axis=0)
        ridge_cent = Ridge(alpha=1.0, random_state=seed)
        ridge_cent.fit(ZD_all, yD_all_reg)
        fit_s = time.time() - t0
        pred_T_ridge = ridge_cent.predict(ZT_26)
        r_mets = eval_regression_metrics(yT_reg, pred_T_ridge)
        sub_mets = compute_per_subject_regression_metrics(T, pred_T_ridge)
        exp4_runs.append({
            'seed': seed, 'task': 'Regression', 'topology': 'Centralized',
            'model': 'Centralized Ridge Regression', 'mu': 0.0,
            'lr_schedule': 'Closed-Form', 'class_weighting': 'none',
            'fit_s': fit_s, 'best_val_metric': np.nan,
            **r_mets, **sub_mets
        })
        
        # B. FedAvg-Ridge Regression (Edge-26)
        t0 = time.time()
        _, _, r_mets, pred_T_fedr_26, bval = train_fed_ridge(
            client_data_reg_26, ZV_26, yV_reg, ZT_26, yT_reg,
            rounds=50, epochs=5, lr_schedule='cosine', alpha_reg=10.0
        )
        fit_s = time.time() - t0
        sub_mets = compute_per_subject_regression_metrics(T, pred_T_fedr_26)
        exp4_runs.append({
            'seed': seed, 'task': 'Regression', 'topology': 'Edge-26',
            'model': 'FedAvg-Ridge (Edge-26)', 'mu': 0.0,
            'lr_schedule': 'cosine', 'class_weighting': 'none',
            'fit_s': fit_s, 'best_val_metric': bval,
            **r_mets, **sub_mets
        })
        
        # C. FedAvg-Ridge Regression (Clinic-5)
        t0 = time.time()
        _, _, r_mets, pred_T_fedr_5, bval = train_fed_ridge(
            client_data_reg_5, ZV_5, yV_reg, ZT_5, yT_reg,
            rounds=50, epochs=5, lr_schedule='cosine', alpha_reg=10.0
        )
        fit_s = time.time() - t0
        sub_mets = compute_per_subject_regression_metrics(T, pred_T_fedr_5)
        exp4_runs.append({
            'seed': seed, 'task': 'Regression', 'topology': 'Clinic-5',
            'model': 'FedAvg-Ridge (Clinic-5 Regional)', 'mu': 0.0,
            'lr_schedule': 'cosine', 'class_weighting': 'none',
            'fit_s': fit_s, 'best_val_metric': bval,
            **r_mets, **sub_mets
        })
        
        print(f"Exp 4 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_runs = pd.DataFrame(exp4_runs)
    out_runs = os.path.join(R, "exp4_federated_runs.csv")
    out_sum = os.path.join(R, "exp4_federated_summary.csv")
    df_runs.to_csv(out_runs, index=False)
    
    # Consolidated summary
    sum_rows = []
    for (task, topo, m_name), grp in df_runs.groupby(['task', 'topology', 'model']):
        row = {
            'task': task,
            'topology': topo,
            'model': m_name,
            'fit_s_avg': grp['fit_s'].mean()
        }
        if task == 'Classification':
            row.update({
                'roc_auc_mean': grp['auc'].mean(),
                'roc_auc_std': grp['auc'].std(),
                'prauc_mean': grp['prauc'].mean(),
                'bacc_mean': grp['bacc'].mean(),
                'bacc_std': grp['bacc'].std(),
                'sens_mean': grp['sens'].mean(),
                'spec_mean': grp['spec'].mean(),
                'f1m_mean': grp['f1m'].mean(),
                'f1m_std': grp['f1m'].std(),
                'mcc_mean': grp['mcc'].mean(),
                'mcc_std': grp['mcc'].std(),
                'brier_mean': grp['brier'].mean(),
                'brier_std': grp['brier'].std(),
                'log_loss_mean': grp['log_loss'].mean(),
                'log_loss_std': grp['log_loss'].std()
            })
        else:
            row.update({
                'mae_mean': grp['mae'].mean(),
                'mae_std': grp['mae'].std(),
                'rmse_mean': grp['rmse'].mean(),
                'rmse_std': grp['rmse'].std(),
                'r2_mean': grp['r2'].mean(),
                'r2_std': grp['r2'].std(),
                'pearson_r_mean': grp['pearson_r'].mean(),
                'pearson_r_std': grp['pearson_r'].std(),
                'mean_subj_mae_mean': grp['mean_subj_mae'].mean(),
                'mean_subj_mae_std': grp['mean_subj_mae'].std()
            })
        sum_rows.append(row)
        
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 4 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_4(ns)

"""
Phase 4: Common Utilities & Federated Learning Primitives
Provides:
  - Dataset loading and 19-feature definitions
  - Decentralized sufficient statistics aggregation (privacy-preserving standardization)
  - Subject-to-client partition generators (26-device edge, K-clinic multi-patient)
  - Full evaluation metric functions for classification and regression
"""

import os, sys, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, r2_score
)
from common import load, partition, make_sets

PREDICTORS = [
    'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)',
    'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]

def load_data():
    return load()

# -----------------------------------------------------------------------------
# DECENTRALIZED SUFFICIENT STATISTICS AGGREGATION
# -----------------------------------------------------------------------------
def decentralized_standardization(client_X_list, XV, XT):
    """
    Decentralized standardization without transferring raw patient data:
    1. Each client computes n_k, sum(x_k), sum(x_k^2)
    2. Server aggregates to compute global mean and std
    3. Server broadcasts global mean and std to standardize client, val, and test data
    Yields exact algebraic parity with centralized StandardScaler!
    """
    total_n = sum(len(x) for x in client_X_list)
    global_sum = sum(np.sum(x, axis=0) for x in client_X_list)
    global_sq_sum = sum(np.sum(x**2, axis=0) for x in client_X_list)
    
    mean = global_sum / total_n
    var = (global_sq_sum / total_n) - (mean**2)
    std = np.sqrt(np.maximum(var, 1e-12))
    
    ZD_clients = [(x - mean) / std for x in client_X_list]
    ZV = (XV - mean) / std
    ZT = (XT - mean) / std
    return ZD_clients, ZV, ZT, mean, std

# -----------------------------------------------------------------------------
# CLIENT PARTITIONING PROTOCOLS
# -----------------------------------------------------------------------------
def get_natural_26_clients(df_dev):
    """
    Topology 1: 26 Autonomous Edge Devices (One client per subject).
    Returns list of client dictionaries with client_id, subject_id, indices, and sizes.
    """
    clients = []
    for cid, (sid, grp) in enumerate(df_dev.groupby('subject#')):
        clients.append({
            'client_id': cid,
            'subject_id': int(sid),
            'indices': grp.index.values,
            'n_records': len(grp),
            'pos_rate': float(grp.y.mean()),
            'mean_updrs': float(grp['motor_UPDRS'].mean())
        })
    return clients

def get_multicenter_clients(df_dev, K=5, mode='natural', seed=0):
    """
    Topology 2: Multi-Patient Clinical Centers (K=5).
    mode='natural': regional hospital assignment (permuted subject grouping).
    mode='label_balanced': serpentine greedy assignment balancing positive rates.
    """
    dev_subjects = list(df_dev['subject#'].unique())
    if mode == 'natural':
        rng = np.random.RandomState(seed + 500)
        perm = rng.permutation(dev_subjects)
        splits = np.array_split(perm, K)
        clinic_subjs = [list(s) for s in splits]
    elif mode == 'label_balanced':
        # Sort subjects by positive rate descending
        sub_rates = []
        for sid in dev_subjects:
            sub_df = df_dev[df_dev['subject#'] == sid]
            sub_rates.append((sid, float(sub_df.y.mean()), len(sub_df)))
        sub_rates.sort(key=lambda x: x[1], reverse=True)
        
        clinic_subjs = [[] for _ in range(K)]
        clinic_pos = [0.0 for _ in range(K)]
        clinic_total = [0 for _ in range(K)]
        
        for sid, rate, n in sub_rates:
            idx = min(range(K), key=lambda i: (clinic_pos[i] / max(1, clinic_total[i]), clinic_total[i]))
            clinic_subjs[idx].append(sid)
            clinic_pos[idx] += rate * n
            clinic_total[idx] += n
    else:
        raise ValueError(f"Unknown mode {mode}")
        
    clients = []
    for cid, sub_list in enumerate(clinic_subjs):
        grp = df_dev[df_dev['subject#'].isin(sub_list)]
        clients.append({
            'client_id': cid,
            'subject_ids': [int(s) for s in sub_list],
            'indices': grp.index.values,
            'n_records': len(grp),
            'pos_rate': float(grp.y.mean()),
            'mean_updrs': float(grp['motor_UPDRS'].mean())
        })
    return clients

# -----------------------------------------------------------------------------
# EVALUATION METRICS
# -----------------------------------------------------------------------------
def eval_classification_metrics(y_true, y_pred, y_prob):
    """Full 8-metric classification evaluation suite."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    prauc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    
    p, r, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0
    
    return {
        'auc': auc, 'prauc': prauc, 'acc': acc, 'bacc': bacc,
        'sens': sens, 'spec': spec, 'prec': prec, 'rec': sens,
        'f1m': float(f1_macro), 'f1w': float(f1_weighted), 'mcc': mcc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def eval_regression_metrics(y_true, y_pred):
    """Full 6-metric regression evaluation suite."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    std_t = np.std(y_true)
    std_p = np.std(y_pred)
    if std_t > 1e-9 and std_p > 1e-9:
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    else:
        pearson_r, spearman_rho = 0.0, 0.0
        
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'pearson_r': float(pearson_r),
        'spearman_rho': float(spearman_rho)
    }

def compute_per_subject_regression_metrics(df_test, y_pred):
    """Computes subject-macro regression error and inter-subject variance."""
    df_test = df_test.reset_index(drop=True)
    y_true = df_test['motor_UPDRS'].values
    y_pred = np.asarray(y_pred).astype(float)
    
    sub_maes = []
    for sid, grp in df_test.groupby('subject#'):
        idx = grp.index.values
        sub_maes.append(float(np.mean(np.abs(y_true[idx] - y_pred[idx]))))
        
    return {
        'mean_subj_mae': float(np.mean(sub_maes)),
        'std_subj_mae': float(np.std(sub_maes, ddof=1)) if len(sub_maes) > 1 else 0.0,
        'n_eval_subjects': len(sub_maes)
    }


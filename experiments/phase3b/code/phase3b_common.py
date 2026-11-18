"""
Phase 3B: Common utilities, dataset loading, and subject-balanced weighting functions.
Strict scientific integrity:
- Excludes total_UPDRS and subject# from predictors
- Strictly 19 eligible voice dysphonia and demographic predictors
- Subject weights computed strictly on training folds
- Subject calibration protocol simulator for personalized prediction
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(BASE_DIR, 'code'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.tmp_mpl'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons_updrs.csv")

VOICE_FEATS = [
    'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]
PREDICTORS = ['age', 'sex', 'test_time'] + VOICE_FEATS

def load_data():
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 5875, f"Expected 5,875 records, found {len(df)}"
    assert df['subject#'].nunique() == 42, f"Expected 42 subjects, found {df['subject#'].nunique()}"
    return df

def get_subject_weights(df_train):
    """
    Computes subject-balanced sample weights for a training fold:
    w_{ij} = N_train / (K_train * n_j)
    such that:
    1. Every training subject receives equal total weight: N_train / K_train
    2. Sum of all sample weights equals N_train
    """
    subjects = df_train['subject#'].values
    counts = pd.Series(subjects).value_counts().to_dict()
    k_train = len(counts)
    n_train = len(df_train)
    
    weights = np.array([n_train / (k_train * counts[s]) for s in subjects], dtype=float)
    assert np.isclose(weights.sum(), n_train), "Weights sum must equal N_train"
    return weights

def eval_classification_metrics(y_true, y_pred, y_score):
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan
    prauc = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan
    
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    # Specificity
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'bacc': float(balanced_accuracy_score(y_true, y_pred)),
        'sens': sens,
        'spec': spec,
        'prec': float(precision_score(y_true, y_pred, zero_division=0)),
        'rec': sens,
        'f1m': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1w': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'auc': float(auc) if not np.isnan(auc) else np.nan,
        'prauc': float(prauc) if not np.isnan(prauc) else np.nan,
        'mcc': float(matthews_corrcoef(y_true, y_pred))
    }

def eval_regression_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    if np.std(y_pred) > 1e-9:
        pr, _ = stats.pearsonr(y_true, y_pred)
        sr, _ = stats.spearmanr(y_true, y_pred)
    else:
        pr, sr = 0.0, 0.0
        
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': float(pr),
        'spearman_rho': float(sr)
    }

def compute_per_subject_regression_metrics(df_test, y_pred):
    df_test = df_test.reset_index(drop=True)
    sub_maes = []
    sub_rmses = []
    for sid, grp in df_test.groupby('subject#'):
        idx = grp.index.values
        y_sub = df_test.loc[idx, 'motor_UPDRS'].values
        p_sub = y_pred[idx]
        sub_maes.append(mean_absolute_error(y_sub, p_sub))
        sub_rmses.append(np.sqrt(mean_squared_error(y_sub, p_sub)))
        
    return {
        'mean_subj_mae': float(np.mean(sub_maes)),
        'std_subj_mae': float(np.std(sub_maes, ddof=1)) if len(sub_maes) > 1 else 0.0,
        'mean_subj_rmse': float(np.mean(sub_rmses)),
        'n_eval_subjects': len(sub_maes)
    }

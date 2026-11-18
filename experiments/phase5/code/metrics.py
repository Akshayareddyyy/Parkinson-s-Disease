"""
Phase 5: Evaluation Metrics & Descriptive Statistical Primitives
Provides:
  - Full classification suite: ROC-AUC, Balanced Accuracy, MCC, Macro-F1, Brier Score, Log Loss
  - Full regression suite: MAE, RMSE, Pearson r, Spearman rho, R2, Subject-level MAE
  - Descriptive confidence intervals
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error, r2_score
)

def evaluate_classification(y_true, y_pred_prob, y_pred_hard=None):
    """
    Evaluates binary classification predictions.
    y_true: binary labels (0, 1)
    y_pred_prob: continuous probability in [0, 1]
    y_pred_hard: binary predictions (default: threshold 0.5)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    y_pred_prob_clipped = np.clip(y_pred_prob, 1e-12, 1.0 - 1e-12)
    
    if y_pred_hard is None:
        y_pred_hard = (y_pred_prob >= 0.5).astype(int)
    else:
        y_pred_hard = np.asarray(y_pred_hard, dtype=int)
        
    acc = accuracy_score(y_true, y_pred_hard)
    bacc = balanced_accuracy_score(y_true, y_pred_hard)
    mcc = matthews_corrcoef(y_true, y_pred_hard)
    
    cm = confusion_matrix(y_true, y_pred_hard, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_hard, average='macro', zero_division=0)
    
    auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else np.nan
    prauc = average_precision_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_pred_prob)
    loss = log_loss(y_true, y_pred_prob_clipped, labels=[0, 1])
    
    return {
        'roc_auc': auc,
        'prauc': prauc,
        'bacc': bacc,
        'acc': acc,
        'sens': sens,
        'spec': spec,
        'f1_macro': f1,
        'mcc': mcc,
        'brier': brier,
        'log_loss': loss,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

def evaluate_regression(y_true, y_pred, df_records=None):
    """
    Evaluates continuous regression predictions.
    df_records: optional dataframe containing 'subject#' for per-subject analysis.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Correlation metrics
    if np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        pr, _ = stats.pearsonr(y_true, y_pred)
        sr, _ = stats.spearmanr(y_true, y_pred)
    else:
        pr, sr = 0.0, 0.0
        
    out = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pr,
        'spearman_rho': sr
    }
    
    if df_records is not None and 'subject#' in df_records.columns:
        subj_maes = []
        df_eval = df_records.copy()
        df_eval['y_true'] = y_true
        df_eval['y_pred'] = y_pred
        for s, grp in df_eval.groupby('subject#'):
            subj_maes.append(mean_absolute_error(grp['y_true'], grp['y_pred']))
        out['mean_subj_mae'] = float(np.mean(subj_maes))
        out['std_subj_mae'] = float(np.std(subj_maes))
        
    return out

def compute_ci(values, confidence=0.95):
    """Computes descriptive mean, standard deviation, and Student's t 95% CI."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 2:
        return np.mean(vals), 0.0, (np.mean(vals), np.mean(vals))
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    se = std / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, std, (mean - h, mean + h)

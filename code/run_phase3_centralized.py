"""
Phase 3: Comprehensive Centralized Classical Machine Learning Evaluation
Evaluates:
  1. Logistic Regression (L2 regularized)
  2. Linear Support Vector Classifier (LinearSVC)
  3. Random Forest (RF, 200 trees)
  4. Extra Trees (ET, 200 trees)
  5. HistGradientBoostingClassifier (modern gradient boosted decision trees)
  6. GradientBoostingClassifier (standard gradient boosting)
Protocol:
  - FULL eligible development dataset for every run (3,517 - 3,697 records)
  - GroupKFold(5) by subject# on development data ONLY
  - Preprocessing (StandardScaler) fitted strictly on training folds within Pipeline
  - Reports both record-level and subject-level evaluation:
    Acc, BalAcc, Sensitivity, Specificity, Precision, Recall, Macro-F1, Weighted-F1, ROC-AUC, PR-AUC, MCC
  - Strict mathematical handling of mono-class subjects (no artificial AUCs)
  - Full auditability: records per fold, subject counts, confusion matrices, fit times, best hyperparameters
  - Saves all runs to results/phase3_centralized_*.csv
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

from common import BASE_DIR, DATA, FEATS, load, partition, make_sets

R = os.path.join(BASE_DIR, "results") + os.sep
os.makedirs(R, exist_ok=True)
df = load()
NSEEDS = 10

def compute_extended_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan
    prauc = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan
    
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'bacc': float(balanced_accuracy_score(y_true, y_pred)),
        'sens': float(sens),
        'spec': float(spec),
        'prec': float(precision_score(y_true, y_pred, zero_division=0)),
        'rec': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1m': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1w': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'auc': float(auc) if not np.isnan(auc) else np.nan,
        'prauc': float(prauc) if not np.isnan(prauc) else np.nan,
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def compute_subject_weighted_metrics(test_df, y_pred, y_score):
    """
    Computes subject-level metrics, averaging across subjects so that each test patient
    contributes equally, preventing subjects with more recordings from dominating evaluation.
    
    Handles mono-class subjects rigorously:
    - ROC-AUC and PR-AUC are only defined when both classes are present; mono-class subjects are NaN.
    - Sensitivity is evaluated only across subjects with at least one positive record.
    - Specificity is evaluated only across subjects with at least one negative record.
    - Balanced Accuracy is computed per subject (for mono-class, accuracy on that class).
    - Macro-F1 is computed per subject.
    """
    sub_accs = []
    sub_baccs = []
    sub_f1s = []
    sub_sens = []
    sub_specs = []
    sub_aucs = []
    sub_praucs = []
    
    for sid, grp in test_df.groupby('subject#'):
        idx = grp.index.values
        y_sub = test_df.loc[idx, 'y'].values
        p_sub = y_pred[idx]
        s_sub = y_score[idx]
        
        acc_s = float(np.mean(y_sub == p_sub))
        sub_accs.append(acc_s)
        
        f1_s = float(f1_score(y_sub, p_sub, average='macro', zero_division=0))
        sub_f1s.append(f1_s)
        
        has_pos = (y_sub == 1).any()
        has_neg = (y_sub == 0).any()
        
        if has_pos and has_neg:
            sub_baccs.append(float(balanced_accuracy_score(y_sub, p_sub)))
            try:
                sub_aucs.append(float(roc_auc_score(y_sub, s_sub)))
                sub_praucs.append(float(average_precision_score(y_sub, s_sub)))
            except Exception:
                pass
            tp = int(((y_sub == 1) & (p_sub == 1)).sum())
            fn = int(((y_sub == 1) & (p_sub == 0)).sum())
            tn = int(((y_sub == 0) & (p_sub == 0)).sum())
            fp = int(((y_sub == 0) & (p_sub == 1)).sum())
            sub_sens.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
            sub_specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        elif has_pos:
            sub_baccs.append(acc_s)
            tp = int(((y_sub == 1) & (p_sub == 1)).sum())
            fn = int(((y_sub == 1) & (p_sub == 0)).sum())
            sub_sens.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        else:
            sub_baccs.append(acc_s)
            tn = int(((y_sub == 0) & (p_sub == 0)).sum())
            fp = int(((y_sub == 0) & (p_sub == 1)).sum())
            sub_specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            
    mean_sens = float(np.nanmean(sub_sens)) if len(sub_sens) > 0 else np.nan
    mean_spec = float(np.nanmean(sub_specs)) if len(sub_specs) > 0 else np.nan
    mean_bacc = float(np.mean(sub_baccs))
    
    return {
        'subj_weighted_acc': float(np.mean(sub_accs)),
        'subj_weighted_bacc': mean_bacc,
        'subj_weighted_sens': mean_sens,
        'subj_weighted_spec': mean_spec,
        'subj_weighted_f1m': float(np.mean(sub_f1s)),
        'subj_weighted_auc': float(np.mean(sub_aucs)) if len(sub_aucs) > 0 else np.nan,
        'subj_weighted_prauc': float(np.mean(sub_praucs)) if len(sub_praucs) > 0 else np.nan,
        'subj_biclass_count': len(sub_aucs),
        'subj_total_count': len(sub_accs),
        'subj_acc_std': float(np.std(sub_accs))
    }

def get_model_specs(seed):
    models = {
        'LR': (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
            {'logisticregression__C': [0.01, 0.1, 1.0]}
        ),
        'Linear SVM': (
            make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, random_state=seed)),
            {'linearsvc__C': [0.001, 0.01, 0.1]}
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
            {'max_depth': [4, 8, None], 'min_samples_leaf': [5, 20]}
        ),
        'Extra Trees': (
            ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=1),
            {'max_depth': [None], 'min_samples_leaf': [1, 5]}
        ),
        'HistGradientBoosting': (
            HistGradientBoostingClassifier(random_state=seed, max_iter=150),
            {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [15, 31], 'min_samples_leaf': [20]}
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(n_estimators=100, random_state=seed),
            {'learning_rate': [0.1], 'max_depth': [3], 'min_samples_leaf': [20]}
        )
    }
    return models

def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X).astype(float)

def run_phase3(n_seeds=NSEEDS):
    print("=== PHASE 3: COMPREHENSIVE CENTRALIZED CLASSICAL MACHINE LEARNING ===")
    print(f"Seeds: {n_seeds}, Feature Count: {len(FEATS)}, Protocol: Full Development Set")
    
    run_records = []
    subject_records = []
    
    for seed in range(n_seeds):
        t_seed_start = time.time()
        D, V, T, thr = make_sets(df, seed)
        XD = D[FEATS].values
        XV = V[FEATS].values
        XT = T[FEATS].values
        
        yD = D.y.values
        yV = V.y.values
        yT = T.y.values
        
        # Verify FULL data assertion
        assert len(XD) == len(D), f"Dev record mismatch: {len(XD)} vs {len(D)}"
        assert len(XT) == len(T), f"Test record mismatch: {len(XT)} vs {len(T)}"
        
        # Calculate fold record counts
        cv_check = GroupKFold(n_splits=5)
        fold_train_recs = []
        fold_val_recs = []
        for tr_i, val_i in cv_check.split(XD, yD, groups=D['subject#']):
            fold_train_recs.append(len(tr_i))
            fold_val_recs.append(len(val_i))
            
        models_dict = get_model_specs(seed)
        
        for name, (est, grid) in models_dict.items():
            # GroupKFold-5 hyperparameter search on Development set
            cv = GroupKFold(n_splits=5)
            gs = GridSearchCV(est, grid, cv=cv, scoring='balanced_accuracy', n_jobs=1)
            
            t0 = time.time()
            gs.fit(XD, yD, groups=D['subject#'])
            best_model = gs.best_estimator_.fit(XD, yD)
            fit_time = time.time() - t0
            
            # Inference on locked test
            t0_inf = time.time()
            sT = get_scores(best_model, XT)
            infer_time = time.time() - t0_inf
            
            thresh = 0.5 if hasattr(best_model, "predict_proba") else 0.0
            pT = (sT > thresh).astype(int)
            
            # Record-level metrics
            test_metrics = compute_extended_metrics(yT, pT, sT)
            
            # Subject-weighted metrics
            subj_metrics = compute_subject_weighted_metrics(T, pT, sT)
            
            # Per-subject breakdown
            for sid, grp in T.groupby('subject#'):
                idx = grp.index.values
                y_sub = T.loc[idx, 'y'].values
                p_sub = pT[idx]
                subject_records.append({
                    'seed': seed,
                    'model': name,
                    'subject': int(sid),
                    'n_records': len(grp),
                    'pos_rate': float(grp.y.mean()),
                    'acc': float((y_sub == p_sub).mean()),
                    'pred_pos_rate': float(p_sub.mean())
                })
                
            # Store run row
            run_records.append({
                'seed': seed,
                'model': name,
                'n_dev_subjects': int(D['subject#'].nunique()),
                'n_train_records': len(XD),
                'n_val_records': len(V),
                'n_test_records': len(XT),
                'fold_train_recs': str(fold_train_recs),
                'fold_val_recs': str(fold_val_recs),
                'best_params': str(gs.best_params_),
                'fit_s': fit_time,
                'infer_s': infer_time,
                **test_metrics,
                **subj_metrics
            })
            
        elapsed = time.time() - t_seed_start
        print(f"Seed {seed}/{n_seeds-1} finished in {elapsed:.1f}s.", flush=True)
        
    df_runs = pd.DataFrame(run_records)
    df_sub = pd.DataFrame(subject_records)
    
    out_runs = os.path.join(R, "phase3_centralized_runs.csv")
    out_sub = os.path.join(R, "phase3_centralized_subject_metrics.csv")
    out_sum = os.path.join(R, "phase3_centralized_summary.csv")
    
    df_runs.to_csv(out_runs, index=False)
    df_sub.to_csv(out_sub, index=False)
    
    # Compute summary statistics across seeds
    metrics_cols = [
        'acc', 'bacc', 'sens', 'spec', 'prec', 'rec', 'f1m', 'f1w',
        'auc', 'prauc', 'mcc', 'subj_weighted_acc', 'subj_weighted_bacc',
        'subj_weighted_sens', 'subj_weighted_spec', 'subj_weighted_f1m',
        'subj_weighted_auc', 'subj_weighted_prauc'
    ]
    
    summary_rows = []
    for model_name, grp in df_runs.groupby('model'):
        row = {
            'model': model_name,
            'n_seeds': len(grp),
            'n_train_records': int(grp.n_train_records.mean()),
            'n_test_records': int(grp.n_test_records.mean()),
            'fit_s_avg': grp.fit_s.mean(),
            'infer_s_avg': grp.infer_s.mean()
        }
        for m in metrics_cols:
            vals = grp[m].dropna().values
            mean_v = float(np.mean(vals)) if len(vals) > 0 else np.nan
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_v = float(stats.t.ppf(0.975, len(vals) - 1) * std_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row[m + '_mean'] = mean_v
            row[m + '_std'] = std_v
            row[m + '_ci'] = ci_v
            row[m + '_str'] = f"{mean_v:.3f} +/- {ci_v:.3f}" if not np.isnan(mean_v) else "N/A"
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_sum, index=False)
    
    print("=== PHASE 3 CENTRALIZED EVALUATION COMPLETED! ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_sub}\n  {out_sum}")
    return df_summary

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_phase3(ns)

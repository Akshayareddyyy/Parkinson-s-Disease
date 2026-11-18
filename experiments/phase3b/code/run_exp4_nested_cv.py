"""
Phase 3B - Experiment 4: Nested GroupKFold Cross-Validation
Strict scientific protocol:
  - Outer GroupKFold(5) on development cohort (26 subjects)
  - Outer evaluation subjects completely excluded from outer training and all inner tuning
  - Inner GroupKFold(4) on outer training subjects (20-21 subjects)
  - Inner training and inner validation subjects disjoint
  - Preprocessing, scaling, and hyperparameter selection fitted strictly on inner training folds
  - Reports outer fold performance, parameter stability, and variance
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LogisticRegression

from common import partition, make_sets
from phase3b_common import (
    load_data, PREDICTORS, eval_classification_metrics, eval_regression_metrics,
    compute_per_subject_regression_metrics, get_subject_weights
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

def run_experiment_4(n_seeds=5):
    print("=" * 70)
    print("PHASE 3B - EXPERIMENT 4: NESTED GROUPKFOLD CROSS-VALIDATION")
    print(f"Outer Folds: 5, Inner Folds: 4, Disjoint Subject Sets Guaranteed")
    print("=" * 70)
    
    outer_fold_records = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        XD = D[PREDICTORS].values
        yD_cls = D.y.values
        yD_reg = D['motor_UPDRS'].values
        groups_D = D['subject#'].values
        
        outer_cv = GroupKFold(n_splits=5)
        
        outer_fold_idx = 0
        for outer_train_idx, outer_eval_idx in outer_cv.split(XD, yD_cls, groups=groups_D):
            outer_fold_idx += 1
            
            # Outer split data
            X_out_tr, y_out_tr_cls, y_out_tr_reg = XD[outer_train_idx], yD_cls[outer_train_idx], yD_reg[outer_train_idx]
            X_out_ev, y_out_ev_cls, y_out_ev_reg = XD[outer_eval_idx], yD_cls[outer_eval_idx], yD_reg[outer_eval_idx]
            groups_out_tr = groups_D[outer_train_idx]
            groups_out_ev = groups_D[outer_eval_idx]
            
            # Strict assertion: outer evaluation subjects completely excluded from outer training
            s_tr = set(groups_out_tr)
            s_ev = set(groups_out_ev)
            assert len(s_tr & s_ev) == 0, "Outer fold subject leakage violation!"
            
            # Inner GroupKFold for hyperparameter tuning
            inner_cv = GroupKFold(n_splits=4)
            
            # Verify inner fold separation
            for in_tr_idx, in_val_idx in inner_cv.split(X_out_tr, y_out_tr_cls, groups=groups_out_tr):
                s_in_tr = set(groups_out_tr[in_tr_idx])
                s_in_val = set(groups_out_tr[in_val_idx])
                assert len(s_in_tr & s_in_val) == 0, "Inner fold subject leakage violation!"
                assert len(s_in_tr & s_ev) == 0, "Inner training overlaps with outer evaluation!"
                
            # 1. Extra Trees Classification
            cls_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=1))
            ])
            cls_grid = {
                'model__max_depth': [8, None],
                'model__min_samples_leaf': [1, 5]
            }
            gs_cls = GridSearchCV(cls_pipe, cls_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=1)
            gs_cls.fit(X_out_tr, y_out_tr_cls, groups=groups_out_tr)
            
            # Fit winning model on entire outer training fold
            best_cls = gs_cls.best_estimator_.fit(X_out_tr, y_out_tr_cls)
            s_ev = best_cls.predict_proba(X_out_ev)[:, 1]
            p_ev = (s_ev > 0.5).astype(int)
            mets_cls = eval_classification_metrics(y_out_ev_cls, p_ev, s_ev)
            
            # 2. Extra Trees Regression
            reg_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=1))
            ])
            reg_grid = {
                'model__max_depth': [8, None],
                'model__min_samples_leaf': [2, 5]
            }
            gs_reg = GridSearchCV(reg_pipe, reg_grid, cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=1)
            gs_reg.fit(X_out_tr, y_out_tr_reg, groups=groups_out_tr)
            
            best_reg = gs_reg.best_estimator_.fit(X_out_tr, y_out_tr_reg)
            pred_reg_ev = best_reg.predict(X_out_ev)
            mets_reg = eval_regression_metrics(y_out_ev_reg, pred_reg_ev)
            
            # Per-subject outer evaluation metrics
            df_ev = D.iloc[outer_eval_idx]
            sub_reg_mets = compute_per_subject_regression_metrics(df_ev, pred_reg_ev)
            
            outer_fold_records.append({
                'seed': seed,
                'outer_fold': outer_fold_idx,
                'n_outer_train_subjects': len(s_tr),
                'n_outer_eval_subjects': len(s_ev),
                'n_outer_train_records': len(X_out_tr),
                'n_outer_eval_records': len(X_out_ev),
                'best_cls_params': str(gs_cls.best_params_),
                'best_reg_params': str(gs_reg.best_params_),
                'cls_roc_auc': mets_cls['auc'],
                'cls_bacc': mets_cls['bacc'],
                'cls_mcc': mets_cls['mcc'],
                'cls_f1m': mets_cls['f1m'],
                'reg_mae': mets_reg['mae'],
                'reg_rmse': mets_reg['rmse'],
                'reg_r2': mets_reg['r2'],
                'reg_pearson_r': mets_reg['pearson_r'],
                'mean_subj_mae': sub_reg_mets['mean_subj_mae']
            })
            
        print(f"Exp 4 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_nested = pd.DataFrame(outer_fold_records)
    out_runs = os.path.join(R, "exp4_nested_cv_runs.csv")
    out_sum = os.path.join(R, "exp4_nested_cv_summary.csv")
    df_nested.to_csv(out_runs, index=False)
    
    summary_data = [{
        'protocol': 'Nested GroupKFold(5, 4)',
        'total_outer_folds': len(df_nested),
        'cls_roc_auc_mean': df_nested['cls_roc_auc'].mean(),
        'cls_roc_auc_std': df_nested['cls_roc_auc'].std(),
        'cls_bacc_mean': df_nested['cls_bacc'].mean(),
        'cls_bacc_std': df_nested['cls_bacc'].std(),
        'reg_mae_mean': df_nested['reg_mae'].mean(),
        'reg_mae_std': df_nested['reg_mae'].std(),
        'reg_rmse_mean': df_nested['reg_rmse'].mean(),
        'reg_rmse_std': df_nested['reg_rmse'].std(),
        'reg_r2_mean': df_nested['reg_r2'].mean(),
        'reg_r2_std': df_nested['reg_r2'].std(),
        'mean_subj_mae_mean': df_nested['mean_subj_mae'].mean(),
        'mean_subj_mae_std': df_nested['mean_subj_mae'].std()
    }]
    df_sum = pd.DataFrame(summary_data)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 4 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_experiment_4(ns)

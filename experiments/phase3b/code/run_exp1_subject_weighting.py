"""
Phase 3B - Experiment 1: Subject-Balanced Learning (Sample Weighting)
Compares:
  - Ordinary Training (uniform weights: w_ij = 1)
  - Subject-Balanced Training (w_ij = N_train / (K_train * n_j))
Strict compliance:
  - Weights calculated strictly from the training fold
  - Zero leakage from validation or test subjects
  - Every training subject receives identical total weight
  - Sum of sample weights equals N_train
  - Validation and test sets are strictly unweighted
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from common import partition, make_sets
from phase3b_common import (
    load_data, PREDICTORS, get_subject_weights,
    eval_classification_metrics, eval_regression_metrics,
    compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

def run_experiment_1(n_seeds=10):
    print("=" * 70)
    print("PHASE 3B - EXPERIMENT 1: SUBJECT-BALANCED LEARNING (SAMPLE WEIGHTING)")
    print(f"Seeds: {n_seeds}, Features: {len(PREDICTORS)}, Weights: w_ij = N_train / (K_train * n_j)")
    print("=" * 70)
    
    classification_records = []
    regression_records = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        XD = D[PREDICTORS].values
        XT = T[PREDICTORS].values
        yD_cls = D.y.values
        yT_cls = T.y.values
        yD_reg = D['motor_UPDRS'].values
        yT_reg = T['motor_UPDRS'].values
        
        # Calculate training fold subject weights
        wD = get_subject_weights(D)
        
        # Verify weight invariants
        k_dev = D['subject#'].nunique()
        n_dev = len(D)
        for s in D['subject#'].unique():
            s_mask = (D['subject#'] == s).values
            assert np.isclose(wD[s_mask].sum(), n_dev / k_dev), f"Subject {s} total weight invariant violated"
        assert np.isclose(wD.sum(), n_dev), "Total sample weight invariant violated"
        
        # -------------------------------------------------------------
        # CLASSIFICATION MODELS
        # -------------------------------------------------------------
        cls_models = {
            'Extra Trees': (
                lambda: ExtraTreesClassifier(n_estimators=150, max_depth=None, min_samples_leaf=1, random_state=seed, n_jobs=1),
                True
            ),
            'Random Forest': (
                lambda: RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=1),
                True
            ),
            'HistGradientBoosting': (
                lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, random_state=seed),
                True
            ),
            'XGBoost': (
                lambda: XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8, eval_metric='logloss', random_state=seed, n_jobs=1),
                True
            ),
            'CatBoost': (
                lambda: CatBoostClassifier(iterations=150, learning_rate=0.05, depth=4, verbose=0, random_seed=seed, thread_count=1),
                True
            ),
            'Logistic Regression': (
                lambda: LogisticRegression(max_iter=2000, C=0.1, random_state=seed),
                True
            )
        }
        
        # Standard scaler for LR
        sc_cls = StandardScaler().fit(XD)
        XD_scaled = sc_cls.transform(XD)
        XT_scaled = sc_cls.transform(XT)
        
        for name, (m_factory, supports_w) in cls_models.items():
            for mode in ['Ordinary', 'Subject-Balanced']:
                model = m_factory()
                X_fit = XD_scaled if name == 'Logistic Regression' else XD
                X_eval = XT_scaled if name == 'Logistic Regression' else XT
                
                t_fit = time.time()
                if mode == 'Subject-Balanced' and supports_w:
                    model.fit(X_fit, yD_cls, sample_weight=wD)
                else:
                    model.fit(X_fit, yD_cls)
                fit_s = time.time() - t_fit
                
                sT = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_eval)
                pT = (sT > 0.5).astype(int) if hasattr(model, "predict_proba") else (sT > 0.0).astype(int)
                
                mets = eval_classification_metrics(yT_cls, pT, sT)
                classification_records.append({
                    'seed': seed,
                    'model': name,
                    'weighting_mode': mode,
                    'fit_s': fit_s,
                    **mets
                })
                
        # -------------------------------------------------------------
        # REGRESSION MODELS
        # -------------------------------------------------------------
        reg_models = {
            'Extra Trees': (
                lambda: ExtraTreesRegressor(n_estimators=150, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=1),
                True
            ),
            'Random Forest': (
                lambda: RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=1),
                True
            ),
            'HistGradientBoosting': (
                lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, random_state=seed),
                True
            ),
            'XGBoost': (
                lambda: XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=seed, n_jobs=1),
                True
            ),
            'CatBoost': (
                lambda: CatBoostRegressor(iterations=150, learning_rate=0.05, depth=4, verbose=0, random_seed=seed, thread_count=1),
                True
            ),
            'Ridge': (
                lambda: Ridge(alpha=10.0, random_state=seed),
                True
            )
        }
        
        sc_reg = StandardScaler().fit(XD)
        XD_reg_scaled = sc_reg.transform(XD)
        XT_reg_scaled = sc_reg.transform(XT)
        
        for name, (m_factory, supports_w) in reg_models.items():
            for mode in ['Ordinary', 'Subject-Balanced']:
                model = m_factory()
                X_fit = XD_reg_scaled if name == 'Ridge' else XD
                X_eval = XT_reg_scaled if name == 'Ridge' else XT
                
                t_fit = time.time()
                if mode == 'Subject-Balanced' and supports_w:
                    model.fit(X_fit, yD_reg, sample_weight=wD)
                else:
                    model.fit(X_fit, yD_reg)
                fit_s = time.time() - t_fit
                
                pred_test = model.predict(X_eval)
                reg_mets = eval_regression_metrics(yT_reg, pred_test)
                sub_mets = compute_per_subject_regression_metrics(T, pred_test)
                
                regression_records.append({
                    'seed': seed,
                    'model': name,
                    'weighting_mode': mode,
                    'fit_s': fit_s,
                    **reg_mets,
                    **sub_mets
                })
                
        print(f"Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_cls = pd.DataFrame(classification_records)
    df_reg = pd.DataFrame(regression_records)
    
    out_cls = os.path.join(R, "exp1_subject_weighting_classification_runs.csv")
    out_reg = os.path.join(R, "exp1_subject_weighting_regression_runs.csv")
    df_cls.to_csv(out_cls, index=False)
    df_reg.to_csv(out_reg, index=False)
    
    # Generate comparative summary
    sum_rows = []
    for (m, mode), grp in df_cls.groupby(['model', 'weighting_mode']):
        sum_rows.append({
            'task': 'Classification',
            'model': m,
            'weighting_mode': mode,
            'roc_auc_mean': grp['auc'].mean(),
            'roc_auc_std': grp['auc'].std(),
            'bacc_mean': grp['bacc'].mean(),
            'bacc_std': grp['bacc'].std(),
            'mcc_mean': grp['mcc'].mean(),
            'mcc_std': grp['mcc'].std()
        })
    for (m, mode), grp in df_reg.groupby(['model', 'weighting_mode']):
        sum_rows.append({
            'task': 'Regression',
            'model': m,
            'weighting_mode': mode,
            'mae_mean': grp['mae'].mean(),
            'mae_std': grp['mae'].std(),
            'rmse_mean': grp['rmse'].mean(),
            'rmse_std': grp['rmse'].std(),
            'r2_mean': grp['r2'].mean(),
            'r2_std': grp['r2'].std(),
            'mean_subj_mae_mean': grp['mean_subj_mae'].mean(),
            'mean_subj_mae_std': grp['mean_subj_mae'].std()
        })
    df_sum = pd.DataFrame(sum_rows)
    out_sum = os.path.join(R, "exp1_subject_weighting_summary.csv")
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 1 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_cls}\n  {out_reg}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_1(ns)

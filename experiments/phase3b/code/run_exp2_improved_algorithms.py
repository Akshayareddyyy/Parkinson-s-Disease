"""
Phase 3B - Experiment 2: Comprehensive Classical Algorithm Evaluation
Evaluates:
  1. Mean Prediction Baseline
  2. Test_Time Only Baseline
  3. Ridge Regression
  4. ElasticNet
  5. PLS Regression
  6. RBF SVR
  7. Random Forest Regressor
  8. Extra Trees Regressor
  9. HistGradientBoosting Regressor
  10. XGBoost Regressor
  11. CatBoost Regressor
Also benchmarks classification counterparts against Phase 3 Extra Trees (0.629 ROC-AUC).
Protocol: GroupKFold(5) on development data, 10 seeds, full dataset compliance.
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from common import partition, make_sets
from phase3b_common import (
    load_data, PREDICTORS, eval_classification_metrics, eval_regression_metrics,
    compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

def run_experiment_2(n_seeds=10):
    print("=" * 70)
    print("PHASE 3B - EXPERIMENT 2: COMPREHENSIVE CLASSICAL ALGORITHMS")
    print(f"Seeds: {n_seeds}, Full Development Data Protocol")
    print("=" * 70)
    
    reg_runs = []
    cls_runs = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        XD = D[PREDICTORS].values
        XT = T[PREDICTORS].values
        groups = D['subject#'].values
        
        yD_reg = D['motor_UPDRS'].values
        yT_reg = T['motor_UPDRS'].values
        yD_cls = D.y.values
        yT_cls = T.y.values
        
        cv = GroupKFold(n_splits=5)
        
        # -------------------------------------------------------------
        # 1. BASELINES
        # -------------------------------------------------------------
        # Mean baseline
        y_mean_pred = np.full_like(yT_reg, fill_value=float(np.mean(yD_reg)))
        reg_runs.append({
            'seed': seed, 'model': 'Mean Baseline', 'paradigm': 'Baseline', 'fit_s': 0.0,
            **eval_regression_metrics(yT_reg, y_mean_pred),
            **compute_per_subject_regression_metrics(T, y_mean_pred)
        })
        
        # Test_time only baseline
        m_time = LinearRegression().fit(D[['test_time']].values, yD_reg)
        y_time_pred = m_time.predict(T[['test_time']].values)
        reg_runs.append({
            'seed': seed, 'model': 'Test_Time Baseline', 'paradigm': 'Baseline', 'fit_s': 0.001,
            **eval_regression_metrics(yT_reg, y_time_pred),
            **compute_per_subject_regression_metrics(T, y_time_pred)
        })
        
        # -------------------------------------------------------------
        # 2. CONTINUOUS REGRESSION CANDIDATES
        # -------------------------------------------------------------
        reg_specs = {
            'Ridge': (
                make_pipeline(StandardScaler(), Ridge(random_state=seed)),
                {'ridge__alpha': [1.0, 10.0, 100.0]}
            ),
            'ElasticNet': (
                make_pipeline(StandardScaler(), ElasticNet(random_state=seed, max_iter=2000)),
                {'elasticnet__alpha': [0.5, 2.0], 'elasticnet__l1_ratio': [0.2, 0.7]}
            ),
            'PLS Regression': (
                make_pipeline(StandardScaler(), PLSRegression()),
                {'plsregression__n_components': [2, 4]}
            ),
            'RBF SVR': (
                make_pipeline(StandardScaler(), SVR()),
                {'svr__C': [1.0, 10.0], 'svr__epsilon': [0.2, 0.5]}
            ),
            'Random Forest': (
                RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                {'max_depth': [8, None], 'min_samples_leaf': [5]}
            ),
            'Extra Trees': (
                ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                {'max_depth': [None], 'min_samples_leaf': [1, 5]}
            ),
            'HistGradientBoosting': (
                HistGradientBoostingRegressor(max_iter=100, random_state=seed),
                {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31], 'min_samples_leaf': [20]}
            ),
            'XGBoost': (
                XGBRegressor(n_estimators=100, learning_rate=0.08, random_state=seed, n_jobs=1),
                {'max_depth': [4, 6], 'subsample': [0.8]}
            ),
            'CatBoost': (
                CatBoostRegressor(iterations=100, learning_rate=0.08, verbose=0, random_seed=seed, thread_count=1),
                {'depth': [4, 6]}
            )
        }
        
        for name, (est, grid) in reg_specs.items():
            gs = GridSearchCV(est, grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=1)
            t0 = time.time()
            gs.fit(XD, yD_reg, groups=groups)
            best_m = gs.best_estimator_.fit(XD, yD_reg)
            fit_time = time.time() - t0
            
            pT = best_m.predict(XT)
            if pT.ndim > 1: pT = pT.ravel()
            
            reg_runs.append({
                'seed': seed,
                'model': name,
                'paradigm': 'Machine Learning',
                'best_params': str(gs.best_params_),
                'fit_s': fit_time,
                **eval_regression_metrics(yT_reg, pT),
                **compute_per_subject_regression_metrics(T, pT)
            })
            
        # -------------------------------------------------------------
        # 3. CLASSIFICATION COUNTERPARTS (BENCHMARKED AGAINST EXTRA TREES)
        # -------------------------------------------------------------
        cls_specs = {
            'CatBoost': (
                CatBoostClassifier(iterations=100, learning_rate=0.08, verbose=0, random_seed=seed, thread_count=1),
                {'depth': [4, 6]}
            ),
            'XGBoost': (
                XGBClassifier(n_estimators=100, learning_rate=0.08, eval_metric='logloss', random_state=seed, n_jobs=1),
                {'max_depth': [4, 6], 'subsample': [0.8]}
            ),
            'HistGradientBoosting': (
                HistGradientBoostingClassifier(max_iter=100, random_state=seed),
                {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31], 'min_samples_leaf': [20]}
            ),
            'Extra Trees': (
                ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=1),
                {'max_depth': [None], 'min_samples_leaf': [1, 5]}
            ),
            'Random Forest': (
                RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
                {'max_depth': [8, None], 'min_samples_leaf': [5]}
            )
        }
        
        for name, (est, grid) in cls_specs.items():
            gs = GridSearchCV(est, grid, cv=cv, scoring='balanced_accuracy', n_jobs=1)
            t0 = time.time()
            gs.fit(XD, yD_cls, groups=groups)
            best_m = gs.best_estimator_.fit(XD, yD_cls)
            fit_time = time.time() - t0
            
            sT = best_m.predict_proba(XT)[:, 1] if hasattr(best_m, "predict_proba") else best_m.decision_function(XT)
            pT = (sT > 0.5).astype(int) if hasattr(best_m, "predict_proba") else (sT > 0.0).astype(int)
            
            cls_runs.append({
                'seed': seed,
                'model': name,
                'best_params': str(gs.best_params_),
                'fit_s': fit_time,
                **eval_classification_metrics(yT_cls, pT, sT)
            })
            
        print(f"Exp 2 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_reg = pd.DataFrame(reg_runs)
    df_cls = pd.DataFrame(cls_runs)
    
    out_reg = os.path.join(R, "exp2_improved_regression_runs.csv")
    out_cls = os.path.join(R, "exp2_improved_classification_runs.csv")
    out_sum = os.path.join(R, "exp2_improved_algorithms_summary.csv")
    
    df_reg.to_csv(out_reg, index=False)
    df_cls.to_csv(out_cls, index=False)
    
    # Regression summary
    sum_rows = []
    for model, grp in df_reg.groupby('model'):
        sum_rows.append({
            'task': 'Regression',
            'model': model,
            'fit_s_avg': grp['fit_s'].mean(),
            'mae_mean': grp['mae'].mean(),
            'mae_std': grp['mae'].std(),
            'rmse_mean': grp['rmse'].mean(),
            'rmse_std': grp['rmse'].std(),
            'r2_mean': grp['r2'].mean(),
            'r2_std': grp['r2'].std(),
            'pearson_r_mean': grp['pearson_r'].mean(),
            'spearman_rho_mean': grp['spearman_rho'].mean(),
            'mean_subj_mae_mean': grp['mean_subj_mae'].mean(),
            'mean_subj_mae_std': grp['mean_subj_mae'].std(),
            'std_subj_mae_mean': grp['std_subj_mae'].mean()
        })
    # Classification summary
    for model, grp in df_cls.groupby('model'):
        sum_rows.append({
            'task': 'Classification',
            'model': model,
            'fit_s_avg': grp['fit_s'].mean(),
            'roc_auc_mean': grp['auc'].mean(),
            'roc_auc_std': grp['auc'].std(),
            'bacc_mean': grp['bacc'].mean(),
            'bacc_std': grp['bacc'].std(),
            'mcc_mean': grp['mcc'].mean(),
            'mcc_std': grp['mcc'].std(),
            'f1m_mean': grp['f1m'].mean(),
            'f1m_std': grp['f1m'].std()
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 2 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_reg}\n  {out_cls}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_2(ns)

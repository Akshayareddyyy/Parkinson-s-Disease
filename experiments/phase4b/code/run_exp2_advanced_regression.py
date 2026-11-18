"""
Phase 4B - Experiment 2: Advanced Continuous Regression
Evaluates:
  1. Mean Dummy Baseline (hat{y} = bar{y}_train)
  2. Test_Time Linear Baseline (hat{y} = beta_0 + beta_1 * test_time)
  3. ElasticNet Regressor (Tuned)
  4. Ridge Regressor (Tuned)
  5. Huber Regressor (Outlier-Robust Linear Model)
  6. Extra Trees Regressor (Tuned)
  7. Random Forest Regressor (Tuned)
  8. CatBoost Regressor (Tuned)
  9. XGBoost Regressor (Tuned)
  10. RBF Support Vector Regressor (Tuned)
Protocol: GroupKFold(5) on development data strictly. Full record accounting.
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from common import make_sets
from phase4b_common import (
    load_data, PREDICTORS, eval_regression_metrics,
    compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

def run_experiment_2(n_seeds=10):
    print("=" * 80)
    print("PHASE 4B - EXPERIMENT 2: ADVANCED CONTINUOUS REGRESSION")
    print("Features: 19 Permitted Predictors, Protocol: GroupKFold(5) on Dev Cohort")
    print("=" * 80)
    
    exp2_runs = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        XD = D[PREDICTORS].values
        XT = T[PREDICTORS].values
        yD = D['motor_UPDRS'].values
        yT = T['motor_UPDRS'].values
        groups_D = D['subject#'].values
        
        sc = StandardScaler().fit(XD)
        ZD = sc.transform(XD)
        ZT = sc.transform(XT)
        
        # 1. Mean Dummy Baseline
        pred_mean = np.full(len(yT), float(np.mean(yD)))
        mets_mean = eval_regression_metrics(yT, pred_mean)
        sub_mean = compute_per_subject_regression_metrics(T, pred_mean)
        exp2_runs.append({
            'seed': seed, 'task': 'Regression', 'model': 'Mean Dummy Baseline',
            'n_train_records': len(D), 'n_calib_records': 0, 'n_val_records': len(V), 'n_test_records': len(T),
            'fit_s': 0.0, **mets_mean, **sub_mean
        })
        
        # 2. Test_Time Linear Baseline
        m_time = LinearRegression().fit(D[['test_time']].values, yD)
        pred_time = m_time.predict(T[['test_time']].values)
        mets_time = eval_regression_metrics(yT, pred_time)
        sub_time = compute_per_subject_regression_metrics(T, pred_time)
        exp2_runs.append({
            'seed': seed, 'task': 'Regression', 'model': 'Test_Time Baseline',
            'n_train_records': len(D), 'n_calib_records': 0, 'n_val_records': len(V), 'n_test_records': len(T),
            'fit_s': 0.0, **mets_time, **sub_time
        })
        
        # Machine learning regressors
        reg_configs = {
            'ElasticNet': (
                ElasticNet(random_state=seed, max_iter=3000),
                {'alpha': [0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}
            ),
            'Ridge': (
                Ridge(random_state=seed),
                {'alpha': [0.1, 1.0, 10.0, 50.0]}
            ),
            'Huber Regressor': (
                HuberRegressor(max_iter=1000),
                {'epsilon': [1.2, 1.35, 1.75], 'alpha': [0.01, 0.1]}
            ),
            'Extra Trees Regressor': (
                ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                {'max_depth': [8, None], 'min_samples_leaf': [2, 5]}
            ),
            'Random Forest Regressor': (
                RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                {'max_depth': [6, 8], 'min_samples_leaf': [2, 5]}
            ),
            'CatBoost Regressor': (
                CatBoostRegressor(iterations=150, random_state=seed, verbose=0, thread_count=1),
                {'depth': [4, 6], 'l2_leaf_reg': [1.0, 5.0]}
            ),
            'XGBoost Regressor': (
                XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=seed, n_jobs=1),
                {'max_depth': [3, 5], 'learning_rate': [0.03, 0.08]}
            ),
            'RBF SVR': (
                SVR(kernel='rbf'),
                {'C': [1.0, 5.0], 'epsilon': [0.2, 0.5]}
            )
        }
        
        cv_inner = GroupKFold(n_splits=5)
        
        for m_name, (base_est, grid) in reg_configs.items():
            t0 = time.time()
            gs = GridSearchCV(base_est, grid, cv=cv_inner, scoring='neg_mean_absolute_error', n_jobs=-1)
            gs.fit(ZD, yD, groups=groups_D)
            best_model = gs.best_estimator_
            fit_s = time.time() - t0
            
            pred_T = best_model.predict(ZT)
            mets = eval_regression_metrics(yT, pred_T)
            sub_mets = compute_per_subject_regression_metrics(T, pred_T)
            
            exp2_runs.append({
                'seed': seed,
                'task': 'Regression',
                'model': m_name,
                'n_train_records': len(ZD),
                'n_calib_records': 0,
                'n_val_records': len(V),
                'n_test_records': len(ZT),
                'best_params': str(gs.best_params_),
                'fit_s': fit_s,
                **mets,
                **sub_mets
            })
            
        print(f"Exp 2 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_runs = pd.DataFrame(exp2_runs)
    out_runs = os.path.join(R, "exp2_advanced_regression_runs.csv")
    out_sum = os.path.join(R, "exp2_advanced_regression_summary.csv")
    df_runs.to_csv(out_runs, index=False)
    
    # Consolidated summary
    sum_rows = []
    for m_name, grp in df_runs.groupby('model'):
        sum_rows.append({
            'model': m_name,
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
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 2 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_2(ns)


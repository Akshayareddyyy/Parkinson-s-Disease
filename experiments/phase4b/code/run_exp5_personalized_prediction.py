"""
Phase 4B - Experiment 5: Personalized Longitudinal Delta Prediction
Evaluates:
  Calibration windows: k in {3, 5, 10, 15}
  Normalization schemes:
    1. Patient Z-score: (x - mu_{s,k}) / sigma_{s,k}
    2. Patient Robust Median/IQR: (x - median_{s,k}) / IQR_{s,k}
    3. Patient Min-Max: (x - min_{s,k}) / (max_{s,k} - min_{s,k})
  Models:
    1. Zero Delta Dummy (Baseline): Delta y = 0 (Predicts patient baseline UPDRS)
    2. Mean Drift Dummy (Baseline): Delta y = bar{Delta y}_train
    3. Ridge Regressor (alpha=1.0)
    4. ElasticNet Regressor (alpha=0.1, l1_ratio=0.5)
    5. Random Forest Regressor (n_estimators=100, max_depth=10)
    6. Extra Trees Regressor (n_estimators=100, max_depth=10)

Protocol:
  Strictly causal calibration on initial k recordings sorted by test_time.
  All future recordings [k, N_s) evaluated out-of-sample.
  Zero target leakage. Full record accounting (N_calib vs N_test_eval).
  Per-subject MAE and SD computed across locked test subjects.
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from common import make_sets
from phase4b_common import load_data, PREDICTORS, eval_regression_metrics

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

# -----------------------------------------------------------------------------
# PATIENT-SPECIFIC NORMALIZATION PRIMITIVES
# -----------------------------------------------------------------------------
def extract_patient_calibration(df_patient, k, norm_type='z_score'):
    """
    Given a single patient's recordings sorted by test_time:
    - Calibration window: recordings [0, k)
    - Future evaluation: recordings [k, N)
    Computes baseline motor_UPDRS and patient feature scaling statistics strictly on [0, k).
    """
    df_p = df_patient.sort_values('test_time').reset_index(drop=True)
    N = len(df_p)
    if N <= k:
        raise ValueError(f"Patient has {N} records <= k={k}")
        
    calib_df = df_p.iloc[:k]
    future_df = df_p.iloc[k:].copy()
    
    # Baseline motor-UPDRS
    y_base = float(calib_df['motor_UPDRS'].mean())
    
    # Feature scaling statistics strictly from calibration records
    X_calib = calib_df[PREDICTORS].values
    
    if norm_type == 'z_score':
        center = np.mean(X_calib, axis=0)
        scale = np.std(X_calib, axis=0)
        scale = np.maximum(scale, 1e-6)
    elif norm_type == 'robust_iqr':
        center = np.median(X_calib, axis=0)
        q75 = np.percentile(X_calib, 75, axis=0)
        q25 = np.percentile(X_calib, 25, axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
    elif norm_type == 'min_max':
        center = np.min(X_calib, axis=0)
        scale = np.maximum(np.max(X_calib, axis=0) - center, 1e-6)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
        
    # Transform future features and targets
    X_future_raw = future_df[PREDICTORS].values
    X_future_norm = (X_future_raw - center) / scale
    
    # Replace any potential NaNs or infs with 0.0
    X_future_norm = np.nan_to_num(X_future_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_future_abs = future_df['motor_UPDRS'].values
    delta_future = y_future_abs - y_base
    
    return {
        'k': k,
        'N_total': N,
        'N_calib': k,
        'N_future': N - k,
        'y_base': y_base,
        'X_future': X_future_norm,
        'delta_future': delta_future,
        'y_future_abs': y_future_abs,
        'subject_id': int(df_p['subject#'].iloc[0])
    }

# -----------------------------------------------------------------------------
# MAIN EXPERIMENT EXECUTION
# -----------------------------------------------------------------------------
def run_experiment_5(n_seeds=10):
    print("=" * 80)
    print("PHASE 4B - EXPERIMENT 5: PERSONALIZED LONGITUDINAL DELTA PREDICTION")
    print("Calibration Windows: k in {3, 5, 10, 15}")
    print("Normalization Schemes: Z-score, Robust Median/IQR, Min-Max")
    print("Models: Zero Baseline, Mean Drift Baseline, Ridge, ElasticNet, RF, Extra Trees")
    print("Strictly Out-of-Sample Future Evaluation (Zero Test Calibration Leakage)")
    print("=" * 80)
    
    k_list = [3, 5, 10, 15]
    norm_types = ['z_score', 'robust_iqr', 'min_max']
    models_to_test = [
        'Zero Delta Dummy',
        'Mean Drift Dummy',
        'Ridge Regressor',
        'ElasticNet Regressor',
        'Random Forest Regressor',
        'Extra Trees Regressor'
    ]
    
    exp5_runs = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        dev_subjects = list(D['subject#'].unique())
        test_subjects = list(T['subject#'].unique())
        
        for k in k_list:
            for norm in norm_types:
                # 1. Process Development Cohort
                X_train_list = []
                delta_train_list = []
                n_calib_train = 0
                
                for sid in dev_subjects:
                    sub_df = D[D['subject#'] == sid]
                    res = extract_patient_calibration(sub_df, k, norm_type=norm)
                    X_train_list.append(res['X_future'])
                    delta_train_list.append(res['delta_future'])
                    n_calib_train += res['N_calib']
                    
                X_train = np.concatenate(X_train_list, axis=0)
                delta_train = np.concatenate(delta_train_list, axis=0)
                mean_delta_train = float(np.mean(delta_train))
                
                # 2. Process Locked-Test Cohort
                test_patient_data = []
                X_test_list = []
                delta_test_list = []
                y_test_abs_list = []
                y_base_test_list = []
                sub_ids_test_list = []
                n_calib_test = 0
                
                for sid in test_subjects:
                    sub_df = T[T['subject#'] == sid]
                    res = extract_patient_calibration(sub_df, k, norm_type=norm)
                    test_patient_data.append(res)
                    X_test_list.append(res['X_future'])
                    delta_test_list.append(res['delta_future'])
                    y_test_abs_list.append(res['y_future_abs'])
                    y_base_test_list.extend([res['y_base']] * res['N_future'])
                    sub_ids_test_list.extend([res['subject_id']] * res['N_future'])
                    n_calib_test += res['N_calib']
                    
                X_test = np.concatenate(X_test_list, axis=0)
                delta_test = np.concatenate(delta_test_list, axis=0)
                y_test_abs = np.concatenate(y_test_abs_list, axis=0)
                y_base_test = np.array(y_base_test_list)
                sub_ids_test = np.array(sub_ids_test_list)
                
                # 3. Evaluate each model
                for m_name in models_to_test:
                    t0 = time.time()
                    
                    if m_name == 'Zero Delta Dummy':
                        pred_delta = np.zeros(len(delta_test), dtype=float)
                    elif m_name == 'Mean Drift Dummy':
                        pred_delta = np.full(len(delta_test), mean_delta_train, dtype=float)
                    elif m_name == 'Ridge Regressor':
                        m = Ridge(alpha=1.0, random_state=seed)
                        m.fit(X_train, delta_train)
                        pred_delta = m.predict(X_test)
                    elif m_name == 'ElasticNet Regressor':
                        m = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed)
                        m.fit(X_train, delta_train)
                        pred_delta = m.predict(X_test)
                    elif m_name == 'Random Forest Regressor':
                        m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
                        m.fit(X_train, delta_train)
                        pred_delta = m.predict(X_test)
                    elif m_name == 'Extra Trees Regressor':
                        m = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
                        m.fit(X_train, delta_train)
                        pred_delta = m.predict(X_test)
                    else:
                        raise ValueError(f"Unknown model: {m_name}")
                        
                    fit_s = time.time() - t0
                    
                    # Absolute reconstructed motor-UPDRS
                    pred_y_abs = y_base_test + pred_delta
                    
                    # Delta prediction metrics
                    delta_mets = eval_regression_metrics(delta_test, pred_delta)
                    # Absolute UPDRS metrics
                    abs_mets = eval_regression_metrics(y_test_abs, pred_y_abs)
                    
                    # Per-subject MAE on absolute UPDRS
                    sub_maes = []
                    for sid in test_subjects:
                        idx = np.where(sub_ids_test == sid)[0]
                        sub_mae = float(np.mean(np.abs(y_test_abs[idx] - pred_y_abs[idx])))
                        sub_maes.append(sub_mae)
                    mean_subj_mae = float(np.mean(sub_maes))
                    std_subj_mae = float(np.std(sub_maes, ddof=1)) if len(sub_maes) > 1 else 0.0
                    
                    exp5_runs.append({
                        'seed': seed,
                        'k_window': k,
                        'norm_type': norm,
                        'model': m_name,
                        'fit_s': fit_s,
                        'n_train_subjects': len(dev_subjects),
                        'n_test_subjects': len(test_subjects),
                        'n_calib_train_records': n_calib_train,
                        'n_train_records': len(X_train),
                        'n_calib_test_records': n_calib_test,
                        'n_test_eval_records': len(X_test),
                        'delta_mae': delta_mets['mae'],
                        'delta_rmse': delta_mets['rmse'],
                        'delta_pearson_r': delta_mets['pearson_r'],
                        'abs_mae': abs_mets['mae'],
                        'abs_rmse': abs_mets['rmse'],
                        'abs_r2': abs_mets['r2'],
                        'abs_pearson_r': abs_mets['pearson_r'],
                        'abs_spearman_rho': abs_mets['spearman_rho'],
                        'mean_subj_mae': mean_subj_mae,
                        'std_subj_mae': std_subj_mae
                    })
                    
        print(f"Exp 5 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_runs = pd.DataFrame(exp5_runs)
    out_runs = os.path.join(R, "exp5_personalized_runs.csv")
    out_sum = os.path.join(R, "exp5_personalized_summary.csv")
    df_runs.to_csv(out_runs, index=False)
    
    # Consolidated summary
    sum_rows = []
    for (k, norm, m_name), grp in df_runs.groupby(['k_window', 'norm_type', 'model']):
        sum_rows.append({
            'k_window': k,
            'norm_type': norm,
            'model': m_name,
            'fit_s_avg': grp['fit_s'].mean(),
            'delta_mae_mean': grp['delta_mae'].mean(),
            'delta_mae_std': grp['delta_mae'].std(),
            'delta_pearson_r_mean': grp['delta_pearson_r'].mean(),
            'delta_pearson_r_std': grp['delta_pearson_r'].std(),
            'abs_mae_mean': grp['abs_mae'].mean(),
            'abs_mae_std': grp['abs_mae'].std(),
            'abs_rmse_mean': grp['abs_rmse'].mean(),
            'abs_rmse_std': grp['abs_rmse'].std(),
            'abs_r2_mean': grp['abs_r2'].mean(),
            'abs_r2_std': grp['abs_r2'].std(),
            'abs_pearson_r_mean': grp['abs_pearson_r'].mean(),
            'abs_pearson_r_std': grp['abs_pearson_r'].std(),
            'mean_subj_mae_mean': grp['mean_subj_mae'].mean(),
            'mean_subj_mae_std': grp['mean_subj_mae'].std()
        })
        
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 5 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_5(ns)


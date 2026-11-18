"""
Phase 3B - Experiment 5: Subject-Aware Normalization & Personalized Prediction Study
Clearly separates two clinical paradigms:
  Scenario A: Global Zero-Shot Prediction (unseen screening patient, global scaler)
  Scenario B: Subject-Calibrated Personalized Prediction (remote patient monitoring with baseline onboarding)
Calibration protocol:
  - First k recordings (k=5 and k=10) sorted chronologically by test_time used strictly for baseline estimation
  - Subsequent recordings (t > t_calib) normalized relative to subject baseline: z = (x - mu_calib) / sigma_calib
  - Zero/near-zero standard deviations handled safely: max(sigma, 1e-6)
  - Evaluates two distinct targets:
    1. Future absolute motor_UPDRS
    2. Progression change relative to calibration baseline: Delta y = y_t - y_baseline
  - Strict compliance: For model training, the identical calibration protocol is simulated independently for every training subject!
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from common import partition, make_sets
from phase3b_common import (
    load_data, PREDICTORS, eval_regression_metrics, compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

def process_subject_calibration(df_cohort, k_calib=5, target_mode='absolute'):
    """
    Simulates clinical baseline onboarding for a cohort:
    - Sorts each patient's records chronologically by test_time
    - Uses first k_calib records for baseline estimation
    - Normalizes subsequent records (t > k_calib)
    - Returns processed feature matrix X, target array y, and filtered dataframe
    """
    X_subsequent = []
    y_subsequent = []
    meta_rows = []
    
    n_excluded_subjects = 0
    
    for sid, grp in df_cohort.groupby('subject#'):
        grp_sorted = grp.sort_values('test_time').reset_index(drop=True)
        if len(grp_sorted) <= k_calib + 5:
            n_excluded_subjects += 1
            continue
            
        calib_recs = grp_sorted.iloc[:k_calib]
        subseq_recs = grp_sorted.iloc[k_calib:].copy()
        
        # Calibration baseline statistics
        mu_calib = calib_recs[PREDICTORS].mean().values
        sigma_calib = calib_recs[PREDICTORS].std(ddof=1).values
        # Safe standard deviation clipping
        sigma_calib = np.where(np.isnan(sigma_calib) | (sigma_calib < 1e-6), 1.0, sigma_calib)
        
        baseline_updrs = float(calib_recs['motor_UPDRS'].mean())
        
        # Normalize subsequent recordings relative to subject baseline
        X_norm = (subseq_recs[PREDICTORS].values - mu_calib) / sigma_calib
        
        if target_mode == 'absolute':
            y_vals = subseq_recs['motor_UPDRS'].values
        elif target_mode == 'delta':
            y_vals = subseq_recs['motor_UPDRS'].values - baseline_updrs
        else:
            raise ValueError(f"Unknown target_mode {target_mode}")
            
        X_subsequent.append(X_norm)
        y_subsequent.append(y_vals)
        meta_rows.append(subseq_recs)
        
    X_out = np.vstack(X_subsequent)
    y_out = np.concatenate(y_subsequent)
    df_out = pd.concat(meta_rows, ignore_index=True)
    
    return X_out, y_out, df_out, n_excluded_subjects

def run_experiment_5(n_seeds=10):
    print("=" * 70)
    print("PHASE 3B - EXPERIMENT 5: SUBJECT-AWARE NORMALIZATION & PERSONALIZED PREDICTION")
    print(f"Scenarios: Global Zero-Shot vs Subject-Calibrated (k=5 and k=10 recordings)")
    print(f"Targets: Future Absolute motor_UPDRS vs Delta motor_UPDRS (Progression)")
    print("=" * 70)
    
    exp5_records = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        # -------------------------------------------------------------
        # SCENARIO A: GLOBAL ZERO-SHOT (BASELINE UNSEEN GENERALIZATION)
        # -------------------------------------------------------------
        sc_global = StandardScaler().fit(D[PREDICTORS].values)
        XD_global = sc_global.transform(D[PREDICTORS].values)
        XT_global = sc_global.transform(T[PREDICTORS].values)
        yD_abs = D['motor_UPDRS'].values
        yT_abs = T['motor_UPDRS'].values
        
        m_et_global = ExtraTreesRegressor(n_estimators=150, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=1)
        m_et_global.fit(XD_global, yD_abs)
        pred_global = m_et_global.predict(XT_global)
        
        exp5_records.append({
            'seed': seed,
            'scenario': 'Global Zero-Shot',
            'calibration_k': 0,
            'target_type': 'Absolute motor_UPDRS',
            'model': 'Extra Trees',
            'n_train_records': len(XD_global),
            'n_test_records': len(XT_global),
            'excluded_subjects': 0,
            **eval_regression_metrics(yT_abs, pred_global),
            **compute_per_subject_regression_metrics(T, pred_global)
        })
        
        # -------------------------------------------------------------
        # SCENARIO B: SUBJECT-CALIBRATED PERSONALIZED PREDICTION
        # Evaluated for k in {5, 10} and target in {'absolute', 'delta'}
        # -------------------------------------------------------------
        for k in [5, 10]:
            for target_mode in ['absolute', 'delta']:
                target_label = 'Absolute motor_UPDRS' if target_mode == 'absolute' else 'Delta motor_UPDRS (Relative Change)'
                
                # Preprocess training cohort under calibration simulation
                XD_cal, yD_cal, dfD_cal, excl_D = process_subject_calibration(D, k_calib=k, target_mode=target_mode)
                
                # Preprocess locked-test cohort under identical calibration simulation
                XT_cal, yT_cal, dfT_cal, excl_T = process_subject_calibration(T, k_calib=k, target_mode=target_mode)
                
                assert excl_D == 0 and excl_T == 0, f"Unexpected subject exclusion (min records in dataset is 101)"
                
                # Fit models on subject-calibrated training data
                models = {
                    'Extra Trees': ExtraTreesRegressor(n_estimators=150, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=1),
                    'Ridge': Ridge(alpha=10.0, random_state=seed),
                    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=1)
                }
                
                for m_name, model in models.items():
                    model.fit(XD_cal, yD_cal)
                    pred_cal = model.predict(XT_cal)
                    
                    reg_mets = eval_regression_metrics(yT_cal, pred_cal)
                    
                    # For delta mode, per-subject MAE reflects accuracy in tracking symptom fluctuation
                    sub_maes = []
                    for sid, grp in dfT_cal.groupby('subject#'):
                        idx = grp.index.values
                        sub_maes.append(float(np.mean(np.abs(yT_cal[idx] - pred_cal[idx]))))
                        
                    exp5_records.append({
                        'seed': seed,
                        'scenario': 'Subject-Calibrated',
                        'calibration_k': k,
                        'target_type': target_label,
                        'model': m_name,
                        'n_train_records': len(XD_cal),
                        'n_test_records': len(XT_cal),
                        'excluded_subjects': excl_D + excl_T,
                        **reg_mets,
                        'mean_subj_mae': float(np.mean(sub_maes)),
                        'std_subj_mae': float(np.std(sub_maes, ddof=1)) if len(sub_maes) > 1 else 0.0,
                        'n_eval_subjects': len(sub_maes)
                    })
                    
        print(f"Exp 5 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_exp5 = pd.DataFrame(exp5_records)
    out_runs = os.path.join(R, "exp5_subject_normalization_runs.csv")
    out_sum = os.path.join(R, "exp5_subject_normalization_summary.csv")
    df_exp5.to_csv(out_runs, index=False)
    
    # Consolidated summary
    sum_rows = []
    for (scen, k, target_t, model), grp in df_exp5.groupby(['scenario', 'calibration_k', 'target_type', 'model']):
        sum_rows.append({
            'scenario': scen,
            'calibration_k': k,
            'target_type': target_t,
            'model': model,
            'n_train_records_avg': int(grp['n_train_records'].mean()),
            'n_test_records_avg': int(grp['n_test_records'].mean()),
            'mae_mean': grp['mae'].mean(),
            'mae_std': grp['mae'].std(),
            'rmse_mean': grp['rmse'].mean(),
            'rmse_std': grp['rmse'].std(),
            'r2_mean': grp['r2'].mean(),
            'r2_std': grp['r2'].std(),
            'pearson_r_mean': grp['pearson_r'].mean(),
            'spearman_rho_mean': grp['spearman_rho'].mean(),
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

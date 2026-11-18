"""
Phase 4B Final Independent Verification Script
Performs:
  1. Audit and extraction of selected features per seed for Yeo-Johnson + CorrelationFilter.
  2. Construction of paired per-seed differences across all comparison models.
  3. Mathematical calculation of baseline anchoring variance contribution in personalized prediction.
  4. Generation of all 4 required CSV deliverables:
     - best_model_configuration.csv
     - selected_features_per_seed.csv
     - paired_seed_comparison.csv
     - phase5_classical_control_matrix.csv
"""

import os, sys, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_DIR = os.path.join(ROOT_DIR, 'experiments', 'phase4b', 'code')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from common import load, make_sets
from phase4b_common import PREDICTORS, eval_regression_metrics

VERIF_DIR = os.path.join(ROOT_DIR, 'experiments', 'phase4b', 'final_verification')
RESULTS_DIR = os.path.join(ROOT_DIR, 'experiments', 'phase4b', 'results')
P3B_RESULTS = os.path.join(ROOT_DIR, 'experiments', 'phase3b', 'results')

df_all = load()

# -----------------------------------------------------------------------------
# 1. EXTRACT SELECTED FEATURES PER SEED (CORRELATION FILTER < 0.85)
# -----------------------------------------------------------------------------
def get_selected_features_per_seed():
    records = []
    
    for seed in range(10):
        D, V, T, thr = make_sets(df_all, seed)
        XD = D[PREDICTORS].values
        
        # Yeo-Johnson fit strictly on D
        pt = PowerTransformer(method='yeo-johnson')
        Z_D = pt.fit_transform(XD)
        
        # Correlation matrix on training fold
        corr_matrix = pd.DataFrame(Z_D, columns=PREDICTORS).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
        kept_features = [col for col in PREDICTORS if col not in to_drop]
        
        records.append({
            'seed': seed,
            'n_dev_records': len(D),
            'n_total_features': len(PREDICTORS),
            'n_features_selected': len(kept_features),
            'selected_features': "; ".join(kept_features),
            'dropped_features': "; ".join(to_drop)
        })
        
    df_sel = pd.DataFrame(records)
    out_path = os.path.join(VERIF_DIR, 'selected_features_per_seed.csv')
    df_sel.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df_sel

# -----------------------------------------------------------------------------
# 2. PAIRED PER-SEED COMPARISONS
# -----------------------------------------------------------------------------
def compute_paired_seed_comparisons():
    # Load Phase 4B Exp 1, 2, 3, 4, 5 runs
    df_exp1 = pd.read_csv(os.path.join(RESULTS_DIR, 'exp1_advanced_classification_runs.csv'))
    df_exp2 = pd.read_csv(os.path.join(RESULTS_DIR, 'exp2_advanced_regression_runs.csv'))
    df_exp3 = pd.read_csv(os.path.join(RESULTS_DIR, 'exp3_feature_engineering_runs.csv'))
    df_exp4 = pd.read_csv(os.path.join(RESULTS_DIR, 'exp4_federated_runs.csv'))
    df_exp5 = pd.read_csv(os.path.join(RESULTS_DIR, 'exp5_personalized_runs.csv'))
    
    # Load Phase 3B runs
    df_p3b_cls = pd.read_csv(os.path.join(P3B_RESULTS, 'exp2_improved_classification_runs.csv'))
    df_p3b_reg = pd.read_csv(os.path.join(P3B_RESULTS, 'exp2_improved_regression_runs.csv'))
    
    paired_rows = []
    
    for seed in range(10):
        # 1. Classification Comparisons
        # Exp 3: Yeo-Johnson + CorrFilter (<0.85)
        auc_p4b_opt = df_exp3[(df_exp3['seed'] == seed) & 
                              (df_exp3['pipeline'] == 'Yeo-Johnson + Correlation Filter (<0.85)') & 
                              (df_exp3['task'] == 'Classification')]['auc'].values[0]
                              
        # Exp 3: Baseline 19 Feats (Extra Trees)
        auc_p4b_base19 = df_exp3[(df_exp3['seed'] == seed) & 
                                (df_exp3['pipeline'] == 'Baseline (19 Feats, StandardScaler)') & 
                                (df_exp3['task'] == 'Classification')]['auc'].values[0]
                                
        # Phase 3B: Extra Trees Classification
        p3b_et = df_p3b_cls[(df_p3b_cls['seed'] == seed) & (df_p3b_cls['model'] == 'Extra Trees')]
        auc_p3b_et = p3b_et['auc'].values[0] if len(p3b_et) > 0 else np.nan
        
        # Exp 4: Safe Local Class-Weighted FedAvg-LR
        auc_fed_safe = df_exp4[(df_exp4['seed'] == seed) & 
                               (df_exp4['model'] == 'FedAvg-LR (Safe Local Class-Weighted)')]['auc'].values[0]
                               
        # Exp 4: Centralized LR (Balanced)
        auc_cent_lr = df_exp4[(df_exp4['seed'] == seed) & 
                              (df_exp4['model'] == 'Centralized LR (Balanced)')]['auc'].values[0]
                              
        # 2. Continuous Regression Comparisons
        # Exp 3: Yeo-Johnson + CorrFilter (<0.85) ElasticNet
        mae_p4b_reg_opt = df_exp3[(df_exp3['seed'] == seed) & 
                                  (df_exp3['pipeline'] == 'Yeo-Johnson + Correlation Filter (<0.85)') & 
                                  (df_exp3['task'] == 'Regression')]['mae'].values[0]
                                  
        # Exp 3: Baseline 19 Feats ElasticNet
        mae_p4b_reg_base19 = df_exp3[(df_exp3['seed'] == seed) & 
                                     (df_exp3['pipeline'] == 'Baseline (19 Feats, StandardScaler)') & 
                                     (df_exp3['task'] == 'Regression')]['mae'].values[0]
                                     
        # Exp 2: Mean Dummy Baseline
        mae_dummy = df_exp2[(df_exp2['seed'] == seed) & (df_exp2['model'] == 'Mean Dummy Baseline')]['mae'].values[0]
        
        # Phase 3B: ElasticNet Regression
        p3b_enet = df_p3b_reg[(df_p3b_reg['seed'] == seed) & (df_p3b_reg['model'] == 'ElasticNet')]
        mae_p3b_enet = p3b_enet['mae'].values[0] if len(p3b_enet) > 0 else np.nan
        
        # Exp 4: FedAvg-Ridge (Edge-26)
        mae_fed_ridge = df_exp4[(df_exp4['seed'] == seed) & (df_exp4['model'] == 'FedAvg-Ridge (Edge-26)')]['mae'].values[0]
        
        # 3. Personalized Progression Comparisons (k=15, Robust IQR)
        p5_enet = df_exp5[(df_exp5['seed'] == seed) & (df_exp5['k_window'] == 15) & 
                          (df_exp5['norm_type'] == 'robust_iqr') & (df_exp5['model'] == 'ElasticNet Regressor')]
        mae_p5_enet_k15 = p5_enet['abs_mae'].values[0]
        
        p5_drift = df_exp5[(df_exp5['seed'] == seed) & (df_exp5['k_window'] == 15) & 
                           (df_exp5['norm_type'] == 'robust_iqr') & (df_exp5['model'] == 'Mean Drift Dummy')]
        mae_p5_drift_k15 = p5_drift['abs_mae'].values[0]
        
        p5_zero = df_exp5[(df_exp5['seed'] == seed) & (df_exp5['k_window'] == 15) & 
                          (df_exp5['norm_type'] == 'robust_iqr') & (df_exp5['model'] == 'Zero Delta Dummy')]
        mae_p5_zero_k15 = p5_zero['abs_mae'].values[0]
        
        paired_rows.append({
            'seed': seed,
            # Classification
            'auc_pruned_9feat': auc_p4b_opt,
            'auc_p4b_base_19feat': auc_p4b_base19,
            'auc_p3b_extra_trees': auc_p3b_et,
            'delta_auc_pruned_vs_base': auc_p4b_opt - auc_p4b_base19,
            'delta_auc_pruned_vs_p3b': auc_p4b_opt - auc_p3b_et,
            'auc_fedavg_safe_local': auc_fed_safe,
            'auc_cent_lr_balanced': auc_cent_lr,
            'delta_auc_fed_vs_cent_lr': auc_fed_safe - auc_cent_lr,
            # Regression
            'mae_pruned_9feat': mae_p4b_reg_opt,
            'mae_p4b_base_19feat': mae_p4b_reg_base19,
            'mae_mean_dummy': mae_dummy,
            'mae_p3b_elasticnet': mae_p3b_enet,
            'delta_mae_pruned_vs_base': mae_p4b_reg_opt - mae_p4b_reg_base19,
            'delta_mae_pruned_vs_dummy': mae_p4b_reg_opt - mae_dummy,
            'delta_mae_pruned_vs_p3b': mae_p4b_reg_opt - mae_p3b_enet,
            'mae_fedavg_ridge': mae_fed_ridge,
            # Personalized Progression (k=15)
            'mae_p5_enet_k15': mae_p5_enet_k15,
            'mae_p5_drift_k15': mae_p5_drift_k15,
            'mae_p5_zero_k15': mae_p5_zero_k15,
            'delta_mae_p5_enet_vs_drift': mae_p5_enet_k15 - mae_p5_drift_k15,
            'delta_mae_p5_enet_vs_zero': mae_p5_enet_k15 - mae_p5_zero_k15
        })
        
    df_paired = pd.DataFrame(paired_rows)
    out_path = os.path.join(VERIF_DIR, 'paired_seed_comparison.csv')
    df_paired.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df_paired

# -----------------------------------------------------------------------------
# 3. MATHEMATICAL VARIANCE DECOMPOSITION (PERSONALIZED PROGRESSION)
# -----------------------------------------------------------------------------
def compute_variance_decomposition():
    k = 15
    decomp_rows = []
    
    for seed in range(10):
        D, V, T, thr = make_sets(df_all, seed)
        test_subjects = list(T['subject#'].unique())
        
        y_test_abs_all = []
        y_base_all = []
        
        for sid in test_subjects:
            df_p = T[T['subject#'] == sid].sort_values('test_time').reset_index(drop=True)
            y_base = float(df_p.iloc[:k]['motor_UPDRS'].mean())
            future_y = df_p.iloc[k:]['motor_UPDRS'].values
            
            y_test_abs_all.extend(future_y)
            y_base_all.extend([y_base] * len(future_y))
            
        y_true = np.array(y_test_abs_all)
        y_base = np.array(y_base_all)
        
        # Calculate development mean delta for Mean Drift Dummy
        dev_deltas = []
        for sid in D['subject#'].unique():
            df_p = D[D['subject#'] == sid].sort_values('test_time').reset_index(drop=True)
            base = float(df_p.iloc[:k]['motor_UPDRS'].mean())
            dev_deltas.extend(df_p.iloc[k:]['motor_UPDRS'].values - base)
        mean_drift = float(np.mean(dev_deltas))
        y_drift = y_base + mean_drift
        
        # Total Variance of test motor-UPDRS
        tot_var = float(np.var(y_true, ddof=0))
        
        # Mean Squared Errors
        mse_zero_base = float(np.mean((y_true - y_base)**2))
        mse_drift = float(np.mean((y_true - y_drift)**2))
        
        # R-squared equivalent (Proportion of variance explained)
        r2_base = 1.0 - (mse_zero_base / tot_var)
        r2_drift = 1.0 - (mse_drift / tot_var)
        
        decomp_rows.append({
            'seed': seed,
            'n_eval_records': len(y_true),
            'total_variance': tot_var,
            'mse_zero_baseline': mse_zero_base,
            'mse_drift_baseline': mse_drift,
            'var_explained_zero_base_pct': r2_base * 100.0,
            'var_explained_drift_base_pct': r2_drift * 100.0,
            'residual_variance_pct': (1.0 - r2_drift) * 100.0
        })
        
    df_decomp = pd.DataFrame(decomp_rows)
    return df_decomp

# -----------------------------------------------------------------------------
# 4. BEST MODEL CONFIGURATION MATRIX
# -----------------------------------------------------------------------------
def build_best_model_configuration(df_sel):
    configs = [
        {
            'task': 'Centralized Severity Classification',
            'model_name': 'ExtraTreesClassifier (Yeo-Johnson + Correlation Filter)',
            'hyperparameters': 'n_estimators=150, max_depth=8, min_samples_leaf=2, class_weight=balanced, random_state=seed',
            'preprocessing_pipeline': 'Yeo-Johnson PowerTransformer -> CorrelationFilter(|r| < 0.85)',
            'feature_space': 'Pruned Acoustic Subset (Mean 9.3 features: age, sex, test_time, Jitter(%), Shimmer, NHR, HNR, RPDE, DFA)',
            'key_metric_name': 'ROC-AUC',
            'mean_metric': 0.6618,
            'std_metric': 0.1077,
            'ci_95_low': 0.6618 - 2.262 * (0.1077 / np.sqrt(10)),
            'ci_95_high': 0.6618 + 2.262 * (0.1077 / np.sqrt(10)),
            'benchmark_comparison': 'Beats Phase 4B 19-feat Extra Trees (0.6354) and Phase 3B Extra Trees (0.6384)'
        },
        {
            'task': 'Centralized Continuous Regression',
            'model_name': 'ElasticNet Regressor (Yeo-Johnson + Correlation Filter)',
            'hyperparameters': 'alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=seed',
            'preprocessing_pipeline': 'Yeo-Johnson PowerTransformer -> CorrelationFilter(|r| < 0.85)',
            'feature_space': 'Pruned Acoustic Subset (Mean 9.3 features)',
            'key_metric_name': 'MAE',
            'mean_metric': 6.8824,
            'std_metric': 0.9988,
            'ci_95_low': 6.8824 - 2.262 * (0.9988 / np.sqrt(10)),
            'ci_95_high': 6.8824 + 2.262 * (0.9988 / np.sqrt(10)),
            'benchmark_comparison': 'Beats Phase 4B 19-feat ElasticNet (6.9317), Phase 3B ElasticNet (6.9635), and Mean Dummy (6.9700)'
        },
        {
            'task': 'Federated Severity Classification',
            'model_name': 'Safe Local Class-Weighted FedAvg-LR',
            'hyperparameters': 'rounds=50, epochs=5, cosine_lr in [0.01, 0.15], mu=0.0, safe_local class-weighting',
            'preprocessing_pipeline': 'Decentralized Sufficient Statistics Aggregation (Mean & Variance)',
            'feature_space': 'Standard 19 Predictors (Full original set)',
            'key_metric_name': 'ROC-AUC',
            'mean_metric': 0.6414,
            'std_metric': 0.1289,
            'ci_95_low': 0.6414 - 2.262 * (0.1289 / np.sqrt(10)),
            'ci_95_high': 0.6414 + 2.262 * (0.1289 / np.sqrt(10)),
            'benchmark_comparison': 'Beats Centralized LR Balanced (0.5775) and standard FedAvg-LR (0.6284)'
        },
        {
            'task': 'Federated Continuous Regression',
            'model_name': 'FedAvg-Ridge Regression (Edge-26)',
            'hyperparameters': 'rounds=50, epochs=5, cosine_lr in [0.001, 0.015], alpha_reg=10.0',
            'preprocessing_pipeline': 'Decentralized Sufficient Statistics Aggregation',
            'feature_space': 'Standard 19 Predictors (Full original set)',
            'key_metric_name': 'MAE',
            'mean_metric': 6.8993,
            'std_metric': 0.8729,
            'ci_95_low': 6.8993 - 2.262 * (0.8729 / np.sqrt(10)),
            'ci_95_high': 6.8993 + 2.262 * (0.8729 / np.sqrt(10)),
            'benchmark_comparison': 'Beats Centralized Ridge (7.0653) and Mean Dummy (6.9700)'
        },
        {
            'task': 'Personalized Longitudinal Progression',
            'model_name': 'ElasticNet Regressor (Patient Robust IQR Scaling, k=15)',
            'hyperparameters': 'alpha=0.1, l1_ratio=0.5, random_state=seed, k=15 calibration window',
            'preprocessing_pipeline': 'Patient-Specific Median Centering & IQR Normalization strictly on initial k=15 records',
            'feature_space': 'Standard 19 Predictors normalized to patient baseline',
            'key_metric_name': 'MAE',
            'mean_metric': 2.6965,
            'std_metric': 0.5771,
            'ci_95_low': 2.6965 - 2.262 * (0.5771 / np.sqrt(10)),
            'ci_95_high': 2.6965 + 2.262 * (0.5771 / np.sqrt(10)),
            'benchmark_comparison': 'Closely matched by Mean Drift Dummy (2.7077 MAE, diff -0.011) and Zero Baseline (2.8582 MAE)'
        }
    ]
    df_best = pd.DataFrame(configs)
    out_path = os.path.join(VERIF_DIR, 'best_model_configuration.csv')
    df_best.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df_best

# -----------------------------------------------------------------------------
# 5. PHASE 5 QUANTUM ML CLASSICAL CONTROL MATRIX
# -----------------------------------------------------------------------------
def build_phase5_control_matrix():
    controls = [
        {
            'qml_target_task': 'Quantum Binary Classification (VQC / QSVM)',
            'input_feature_space': '9-feature Pruned Acoustic Manifold (Yeo-Johnson + Correlation Filter <0.85)',
            'encoding_strategy': 'Angle Encoding (Rx, Rz rotations) on 9 Qubits',
            'primary_classical_control_1': 'ExtraTreesClassifier (Pruned 9 Feats): ROC-AUC 0.6618, BalAcc 0.6193, MCC 0.2286',
            'primary_classical_control_2': 'Safe Local Class-Weighted FedAvg-LR: ROC-AUC 0.6414, BalAcc 0.6103',
            'primary_classical_control_3': 'Centralized Logistic Regression: ROC-AUC 0.5775',
            'evaluation_protocol': '10 Audited Subject-Wise Splits (26 Dev / 7 Val / 9 Test)',
            'success_criterion': 'Quantum model must achieve mean ROC-AUC > 0.6618 under identical 10-seed subject splits without data leakage'
        },
        {
            'qml_target_task': 'Quantum Continuous Progression Regression (PQC Regressor)',
            'input_feature_space': '9-feature Pruned Acoustic Manifold',
            'encoding_strategy': 'Angle Encoding on 9 Qubits or Amplitude Encoding',
            'primary_classical_control_1': 'Centralized ElasticNet Regressor (Pruned 9 Feats): MAE 6.8824, r 0.2773',
            'primary_classical_control_2': 'FedAvg-Ridge Regressor (Edge-26): MAE 6.8993, r 0.2837',
            'primary_classical_control_3': 'Mean Dummy Baseline: MAE 6.9700',
            'evaluation_protocol': '10 Audited Subject-Wise Splits (26 Dev / 7 Val / 9 Test)',
            'success_criterion': 'Quantum regressor must achieve mean MAE < 6.8824 and Pearson r > 0.2773'
        },
        {
            'qml_target_task': 'Personalized Quantum Delta Prediction (PQC-Delta)',
            'input_feature_space': 'Patient-Calibrated Features (k=10 or k=15, Robust IQR)',
            'encoding_strategy': 'Angle Encoding on 9 Qubits representing Delta X',
            'primary_classical_control_1': 'ElasticNet Regressor (k=15): MAE 2.6965, r 0.8891',
            'primary_classical_control_2': 'Mean Drift Dummy Baseline (k=15): MAE 2.7077, r 0.8958',
            'primary_classical_control_3': 'Zero Delta Dummy Baseline (k=15): MAE 2.8582, r 0.8958',
            'evaluation_protocol': 'Chronological Calibration [0, k), Out-of-Sample Future Evaluation [k, N)',
            'success_criterion': 'Quantum progression model must achieve MAE < 2.6965 and beat the Mean Drift Dummy baseline'
        }
    ]
    df_qml = pd.DataFrame(controls)
    out_path = os.path.join(VERIF_DIR, 'phase5_classical_control_matrix.csv')
    df_qml.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df_qml

# -----------------------------------------------------------------------------
# MAIN VERIFICATION RUNNER
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("RUNNING PHASE 4B FINAL INDEPENDENT VERIFICATION")
    print("=" * 80)
    
    df_sel = get_selected_features_per_seed()
    df_paired = compute_paired_seed_comparisons()
    df_decomp = compute_variance_decomposition()
    df_best = build_best_model_configuration(df_sel)
    df_qml = build_phase5_control_matrix()
    
    print("\n" + "=" * 80)
    print("MATHEMATICAL VARIANCE DECOMPOSITION (k=15):")
    print(df_decomp[['seed', 'total_variance', 'var_explained_zero_base_pct', 'var_explained_drift_base_pct', 'residual_variance_pct']])
    print("-" * 80)
    print(f"Mean Variance Explained by Zero Baseline: {df_decomp['var_explained_zero_base_pct'].mean():.2f}% +/- {df_decomp['var_explained_zero_base_pct'].std():.2f}%")
    print(f"Mean Variance Explained by Drift Baseline: {df_decomp['var_explained_drift_base_pct'].mean():.2f}% +/- {df_decomp['var_explained_drift_base_pct'].std():.2f}%")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("PAIRED DIFFERENCES (Pruned 9-feat vs Baselines):")
    print(f"Classification Delta ROC-AUC (Pruned vs 19-feat Base): {df_paired['delta_auc_pruned_vs_base'].mean():+.4f} +/- {df_paired['delta_auc_pruned_vs_base'].std():.4f}")
    print(f"Classification Delta ROC-AUC (Pruned vs P3B Extra Trees): {df_paired['delta_auc_pruned_vs_p3b'].mean():+.4f} +/- {df_paired['delta_auc_pruned_vs_p3b'].std():.4f}")
    print(f"Regression Delta MAE (Pruned vs 19-feat Base): {df_paired['delta_mae_pruned_vs_base'].mean():+.4f} +/- {df_paired['delta_mae_pruned_vs_base'].std():.4f}")
    print(f"Regression Delta MAE (Pruned vs Mean Dummy): {df_paired['delta_mae_pruned_vs_dummy'].mean():+.4f} +/- {df_paired['delta_mae_pruned_vs_dummy'].std():.4f}")
    print(f"Personalized Delta MAE (ElasticNet vs Mean Drift Dummy): {df_paired['delta_mae_p5_enet_vs_drift'].mean():+.4f} +/- {df_paired['delta_mae_p5_enet_vs_drift'].std():.4f}")
    print("=" * 80)

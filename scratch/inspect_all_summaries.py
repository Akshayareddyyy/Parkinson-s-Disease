import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 35)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)

print("=== 1. PHASE 3 CENTRALIZED ===")
p3 = pd.read_csv('results/phase3_centralized_summary.csv')
for idx, r in p3.iterrows():
    print(f"Model: {r['model']:22s} | AUC: {r['auc_mean']:.4f} +/- {r['auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f} | MCC: {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f} | F1m: {r['f1m_mean']:.4f} +/- {r['f1m_std']:.4f} | Fit(s): {r['fit_s_avg']:.2f}")

print("\n=== 2. PHASE 3B EXP1 (SUBJECT WEIGHTING) ===")
p3b_e1 = pd.read_csv('experiments/phase3b/results/exp1_subject_weighting_summary.csv')
for idx, r in p3b_e1.iterrows():
    if r['task'] == 'Classification':
        print(f"Cls | Model: {r['model']:20s} | Mode: {r['weighting_mode']:16s} | AUC: {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f}")
    else:
        print(f"Reg | Model: {r['model']:20s} | Mode: {r['weighting_mode']:16s} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | RMSE: {r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f} | R2: {r['r2_mean']:.4f} +/- {r['r2_std']:.4f}")

print("\n=== 3. PHASE 3B EXP2 (IMPROVED ALGORITHMS) ===")
p3b_e2 = pd.read_csv('experiments/phase3b/results/exp2_improved_algorithms_summary.csv')
for idx, r in p3b_e2.iterrows():
    if r['task'] == 'Classification':
        print(f"Cls | Model: {r['model']:20s} | AUC: {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f} | MCC: {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f}")
    else:
        print(f"Reg | Model: {r['model']:20s} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | RMSE: {r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f} | Pearson r: {r['pearson_r_mean']:.4f} | Spearman rho: {r['spearman_rho_mean']:.4f} | R2: {r['r2_mean']:.4f}")

print("\n=== 4. PHASE 3B EXP5 (PERSONALIZATION) ===")
p3b_e5 = pd.read_csv('experiments/phase3b/results/exp5_subject_normalization_summary.csv')
for idx, r in p3b_e5.iterrows():
    print(f"k={r['calibration_k']:2d} | Target: {r['target_type'][:20]:20s} | Model: {r['model']:15s} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | RMSE: {r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f} | R2: {r['r2_mean']:.4f} | r: {r['pearson_r_mean']:.4f}")

print("\n=== 5. PHASE 4 FEDERATED LEARNING ===")
p4 = pd.read_csv('experiments/phase4/results/phase4_federated_summary.csv')
for idx, r in p4.iterrows():
    top = str(r['topology']) if pd.notna(r['topology']) else "Centralized"
    if r['task'] == 'Classification':
        print(f"Cls | Model: {r['model']:18s} | Topo: {top:24s} | AUC: {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f} | MCC: {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f}")
    else:
        print(f"Reg | Model: {r['model']:18s} | Topo: {top:24s} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | RMSE: {r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f} | r: {r['pearson_r_mean']:.4f}")

print("\n=== 6. PHASE 4B EXP1 (ADVANCED CLASSIFICATION) ===")
p4b_e1 = pd.read_csv('experiments/phase4b/results/exp1_advanced_classification_summary.csv')
for idx, r in p4b_e1.iterrows():
    print(f"Model: {r['model']:24s} | AUC: {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f} | MCC: {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f} | Brier: {r['brier_mean']:.4f}")

print("\n=== 7. PHASE 4B EXP2 (ADVANCED REGRESSION) ===")
p4b_e2 = pd.read_csv('experiments/phase4b/results/exp2_advanced_regression_summary.csv')
for idx, r in p4b_e2.iterrows():
    print(f"Model: {r['model']:24s} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | RMSE: {r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f} | r: {r['pearson_r_mean']:.4f} | rho: {r['spearman_rho_mean']:.4f} | R2: {r['r2_mean']:.4f}")

print("\n=== 8. PHASE 4B EXP3 (FEATURE ENGINEERING PIPELINES) ===")
p4b_e3 = pd.read_csv('experiments/phase4b/results/exp3_feature_engineering_summary.csv')
for idx, r in p4b_e3.iterrows():
    if r['task'] == 'Classification':
        print(f"Cls | Pipe: {r['pipeline']:40s} | Feats: {r['n_features_avg']:.1f} | AUC: {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f}")
    else:
        print(f"Reg | Pipe: {r['pipeline']:40s} | Feats: {r['n_features_avg']:.1f} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | r: {r['pearson_r_mean']:.4f}")

print("\n=== 9. PHASE 4B EXP4 (FEDERATED IMPROVEMENTS) ===")
p4b_e4 = pd.read_csv('experiments/phase4b/results/exp4_federated_summary.csv')
for idx, r in p4b_e4.iterrows():
    top = str(r['topology'])
    if r['task'] == 'Classification':
        print(f"Cls | Model: {r['model']:28s} | Topo: {top:20s} | AUC: {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f} | MCC: {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f}")
    else:
        print(f"Reg | Model: {r['model']:28s} | Topo: {top:20s} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | RMSE: {r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f} | r: {r['pearson_r_mean']:.4f}")

print("\n=== 10. PHASE 4B EXP5 (PERSONALIZED CALIBRATION WINDOWS) ===")
p4b_e5 = pd.read_csv('experiments/phase4b/results/exp5_personalized_summary.csv')
# Look at standard scaling and key models across k in [3, 5, 10, 15]
for k in [3, 5, 10, 15]:
    sub = p4b_e5[(p4b_e5['k_window'] == k) & (p4b_e5['norm_type'] == 'standard')]
    for idx, r in sub.iterrows():
        print(f"k={r['k_window']:2d} | Model: {r['model']:22s} | Delta MAE: {r['delta_mae_mean']:.4f} | Abs MAE: {r['abs_mae_mean']:.4f} +/- {r['abs_mae_std']:.4f} | Abs RMSE: {r['abs_rmse_mean']:.4f} | Abs r: {r['abs_pearson_r_mean']:.4f} | Abs R2: {r['abs_r2_mean']:.4f}")

print("\n=== 11. PHASE 5 QUANTUM VS CLASSICAL CONTROLS ===")
p5_c = pd.read_csv('experiments/phase5/results/exp1_vqc_classification_summary.csv')
print("--- Classification (9 Feats) ---")
for idx, r in p5_c.iterrows():
    print(f"Model: {r['model']:30s} | Seeds: {r['n_seeds']} | AUC: {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | BalAcc: {r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f} | MCC: {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f} | F1m: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f} | Fit(s): {r['fit_time_s_mean']:.2f}")

p5_r = pd.read_csv('experiments/phase5/results/exp2_pqcr_regression_summary.csv')
print("--- Regression (9 Feats) ---")
for idx, r in p5_r.iterrows():
    print(f"Model: {r['model']:30s} | Seeds: {r['n_seeds']} | MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | RMSE: {r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f} | r: {r['pearson_r_mean']:.4f} +/- {r['pearson_r_std']:.4f} | rho: {r['spearman_rho_mean']:.4f} | R2: {r['r2_mean']:.4f} | Fit(s): {r['fit_time_s_mean']:.2f}")

print("\n=== 12. PHASE 5 STRATIFIED NOISE SUMMARY ===")
p5_n = pd.read_csv('experiments/phase5/results/exp5_stratified_noise_summary.csv')
for idx, r in p5_n.iterrows():
    print(f"Model: {r['model']:15s} | Cond: {r['condition']:30s} | Metric: {r['primary_metric']:10s} | Mean: {r['metric_mean']:.4f} +/- {r['metric_std']:.4f} | CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")


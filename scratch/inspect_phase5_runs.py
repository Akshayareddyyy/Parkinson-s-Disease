import pandas as pd
df1 = pd.read_csv('experiments/phase5/results/exp1_vqc_classification_runs.csv')
print("=== VQC Classification Runs ===")
print(df1[['seed', 'model', 'depth_L', 'roc_auc', 'bacc', 'mcc', 'f1_macro', 'brier', 'fit_time_s']])

df2 = pd.read_csv('experiments/phase5/results/exp2_pqcr_regression_runs.csv')
print("\n=== PQC-R Regression Runs ===")
print(df2[['seed', 'model', 'depth_L', 'mae', 'rmse', 'pearson_r', 'spearman_rho', 'r2', 'fit_time_s']])

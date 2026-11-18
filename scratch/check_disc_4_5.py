import pandas as pd
df = pd.read_csv('experiments/phase4b/results/exp5_personalized_summary.csv')

print("=== ElasticNet Regressor in exp5_personalized_summary.csv ===")
enet = df[df['model'] == 'ElasticNet Regressor']
print(enet[['k_window', 'norm_type', 'abs_mae_mean', 'abs_mae_std', 'abs_rmse_mean', 'abs_pearson_r_mean', 'abs_r2_mean']])

print("\n=== Mean Drift Dummy in exp5_personalized_summary.csv ===")
drift = df[df['model'] == 'Mean Drift Dummy']
print(drift[['k_window', 'norm_type', 'abs_mae_mean', 'abs_mae_std', 'abs_rmse_mean', 'abs_pearson_r_mean', 'abs_r2_mean']])

print("\n=== Zero Delta Dummy in exp5_personalized_summary.csv ===")
zero = df[df['model'] == 'Zero Delta Dummy']
print(zero[['k_window', 'norm_type', 'abs_mae_mean', 'abs_mae_std', 'abs_rmse_mean', 'abs_pearson_r_mean', 'abs_r2_mean']])


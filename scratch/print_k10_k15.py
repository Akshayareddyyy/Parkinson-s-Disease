import pandas as pd
df = pd.read_csv('experiments/phase4b/results/exp5_personalized_summary.csv')

for k in [10, 15]:
    print(f"=== k = {k} ===")
    for m in ['ElasticNet Regressor', 'Mean Drift Dummy', 'Zero Delta Dummy']:
        sub = df[(df['k_window'] == k) & (df['model'] == m)]
        for idx, r in sub.iterrows():
            print(f"Model: {r['model']:22s} | Norm: {r['norm_type']:10s} | MAE: {r['abs_mae_mean']:.6f} +/- {r['abs_mae_std']:.6f} | RMSE: {r['abs_rmse_mean']:.6f} | r: {r['abs_pearson_r_mean']:.6f} | R2: {r['abs_r2_mean']:.6f}")


import pandas as pd
df = pd.read_csv('experiments/phase4b/results/exp5_personalized_summary.csv')
sub = df[df['norm_type'] == 'z_score']
for idx, r in sub.iterrows():
    print(f"k={r['k_window']:2d} | Model: {r['model']:24s} | Delta MAE: {r['delta_mae_mean']:.4f} +/- {r['delta_mae_std']:.4f} | Abs MAE: {r['abs_mae_mean']:.4f} +/- {r['abs_mae_std']:.4f} | Abs r: {r['abs_pearson_r_mean']:.4f} | Abs R2: {r['abs_r2_mean']:.4f}")


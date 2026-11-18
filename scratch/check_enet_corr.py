import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("=== Phase 5 Classical Control: ElasticNet on 9 Feats ===")
p5_regr_runs = pd.read_csv('experiments/phase5/results/exp2_pqcr_regression_runs.csv')
p5_enet = p5_regr_runs[p5_regr_runs['model'] == 'ElasticNet (9 Feats)']
print(p5_enet[['seed', 'mae', 'rmse', 'pearson_r', 'spearman_rho', 'r2']])
print(f"Per-seed Pearson r values: {p5_enet['pearson_r'].values}")
print(f"Mean per-seed Pearson r: {p5_enet['pearson_r'].mean():.6f} +/- {p5_enet['pearson_r'].std():.6f}")

# To compute pooled Pearson r, let's check if predictions are saved or if we can run inference using the exact model on the test sets
# Let's check if predictions are saved in results or if code/preprocessing.py + classical_controls.py can compute pooled predictions


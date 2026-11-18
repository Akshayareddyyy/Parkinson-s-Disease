import sys, os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'experiments', 'phase5', 'code'))

from common import load_data, get_partition_sets, FEATS_9
from preprocessing import fit_scaler, transform_features
from classical_controls import train_elastic_net

y_true_all = []
y_pred_all = []

per_seed_r = []
df = load_data()

for seed in range(10):
    D, V, T, thr = get_partition_sets(df, seed=seed)
    scaler = fit_scaler(D[FEATS_9].values)
    
    X_dev = transform_features(D[FEATS_9].values, scaler)
    y_dev = D['motor_UPDRS'].values
    
    X_test = transform_features(T[FEATS_9].values, scaler)
    y_test = T['motor_UPDRS'].values
    
    model = train_elastic_net(X_dev, y_dev, seed=seed)
    preds = model.predict(X_test)
    
    r_val, _ = pearsonr(y_test, preds)
    per_seed_r.append(r_val)
    
    y_true_all.extend(y_test)
    y_pred_all.extend(preds)

pooled_r, _ = pearsonr(y_true_all, y_pred_all)
print(f"Computed Per-Seed Pearson r: {np.mean(per_seed_r):.6f} +/- {np.std(per_seed_r, ddof=1):.6f}")
print(f"Per-seed Pearson r array: {per_seed_r}")
print(f"Computed Pooled Pearson r: {pooled_r:.6f}")

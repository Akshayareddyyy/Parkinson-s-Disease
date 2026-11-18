import sys, os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'experiments', 'phase5', 'code'))

from common import load_data, get_partition_sets
from preprocessing import FoldContainedQuantumPreprocessor

df = load_data()

y_true_all = []
y_pred_all = []
per_seed_r = []

for seed in range(10):
    D, V, T, thr = get_partition_sets(df, seed=seed)
    prep = FoldContainedQuantumPreprocessor()
    prep.fit(D)
    
    X_dev = prep.transform_classical(D)
    y_dev = D['motor_UPDRS'].values.astype(float)
    
    X_te = prep.transform_classical(T)
    y_te = T['motor_UPDRS'].values.astype(float)
    
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=seed)
    enet.fit(X_dev, y_dev)
    preds = enet.predict(X_te)
    
    r_val, _ = pearsonr(y_te, preds)
    per_seed_r.append(r_val)
    
    y_true_all.extend(y_te)
    y_pred_all.extend(preds)

pooled_r, _ = pearsonr(y_true_all, y_pred_all)
print(f"Per-seed Pearson r: {per_seed_r}")
print(f"Mean per-seed Pearson r: {np.mean(per_seed_r):.6f} +/- {np.std(per_seed_r, ddof=1):.6f}")
print(f"Pooled Pearson r (across all {len(y_true_all)} test samples): {pooled_r:.6f}")


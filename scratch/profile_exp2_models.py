import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import sys
sys.path.insert(0, 'code')
import common
df = common.load()
D, V, T, thr = common.make_sets(df, 0)
X = D[common.FEATS].values
y = D['motor_UPDRS'].values

models = {
    'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    'ElasticNet': make_pipeline(StandardScaler(), ElasticNet(alpha=1.0, l1_ratio=0.5)),
    'PLS': make_pipeline(StandardScaler(), PLSRegression(n_components=4)),
    'HistGB': HistGradientBoostingRegressor(max_iter=100, random_state=0),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=None, min_samples_leaf=2, random_state=0, n_jobs=1),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=0, n_jobs=1),
    'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.08, depth=4, verbose=0, random_seed=0, thread_count=1),
    'RBF SVR': make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.5))
}

print(f"Timing models on N={len(X)} records:")
for name, m in models.items():
    t0 = time.time()
    m.fit(X, y)
    dt = time.time() - t0
    print(f"  {name:15s}: {dt:.3f}s")

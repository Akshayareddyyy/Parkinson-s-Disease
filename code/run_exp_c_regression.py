"""
Experiment C: Continuous motor_UPDRS Progression Regression
Predicts continuous motor_UPDRS from 19 voice features without binarization.
Evaluates:
  - Ridge Regression (Centralized)
  - Support Vector Regressor (SVR, Centralized)
  - Random Forest Regressor (Centralized)
  - Extra Trees Regressor (Centralized)
  - Federated Linear Regression (FedAvg-MSE, K=26 and K=5 clinics)
Strict subject-wise 26/7/9 partitions across 10 seeds.
Reports: MAE, RMSE, R-squared, Pearson r, and Spearman rho.
Outputs written to results/exp_c_regression_runs.csv and exp_c_regression_summary.csv.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import BASE_DIR, DATA, FEATS, load, partition

R = os.path.join(BASE_DIR, "results") + os.sep
os.makedirs(R, exist_ok=True)
df = load()
NSEEDS = 10

def eval_reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr, _ = stats.pearsonr(y_true, y_pred) if np.std(y_pred) > 1e-9 else (0.0, 1.0)
    sr, _ = stats.spearmanr(y_true, y_pred) if np.std(y_pred) > 1e-9 else (0.0, 1.0)
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pr,
        'spearman_rho': sr
    }

def run_federated_ridge(ZD_clients, yD_clients, ZT, yT, rounds=40, E=5, lr=0.01, l2=0.1):
    """
    Federated linear regression optimizing MSE loss + L2 regularization.
    """
    d = ZD_clients[0].shape[1]
    w = np.zeros(d)
    b = float(np.mean(np.concatenate(yD_clients)))
    
    ns = [len(y) for y in yD_clients]
    total_n = sum(ns)
    weights = [n / total_n for n in ns]
    
    t0 = time.time()
    for _ in range(rounds):
        ws, bs = [], []
        for k in range(len(ZD_clients)):
            Xk = ZD_clients[k]
            yk = yD_clients[k]
            lw = w.copy()
            lb = b
            nk = len(yk)
            for _ in range(E):
                pred = Xk @ lw + lb
                err = pred - yk
                grad_w = (Xk.T @ err) / nk + l2 * lw
                grad_b = np.mean(err)
                lw -= lr * grad_w
                lb -= lr * grad_b
            ws.append(lw)
            bs.append(lb)
        w = sum(a * wk for a, wk in zip(weights, ws))
        b = sum(a * bk for a, bk in zip(weights, bs))
    fit_time = time.time() - t0
    
    pred_test = ZT @ w + b
    mets = eval_reg_metrics(yT, pred_test)
    return fit_time, mets

def run_experiment_c(n_seeds=NSEEDS):
    print(f"=== Starting Experiment C: Continuous motor_UPDRS Regression ===")
    print(f"Seeds: {n_seeds}, Input Features: {len(FEATS)} (target UPDRS excluded)")
    
    records = []
    
    for seed in range(n_seeds):
        dv, va, te = partition(df, seed)
        D = df[df['subject#'].isin(dv)].reset_index(drop=True)
        V = df[df['subject#'].isin(va)].reset_index(drop=True)
        T = df[df['subject#'].isin(te)].reset_index(drop=True)
        
        XD = D[FEATS].values
        XV = V[FEATS].values
        XT = T[FEATS].values
        
        yD = D['motor_UPDRS'].values
        yV = V['motor_UPDRS'].values
        yT = T['motor_UPDRS'].values
        
        # 1. Centralized Ridge
        ridge_pipe = make_pipeline(StandardScaler(), Ridge())
        ridge_grid = {'ridge__alpha': [0.1, 1.0, 10.0, 100.0]}
        gs_ridge = GridSearchCV(ridge_pipe, ridge_grid, cv=GroupKFold(5), scoring='neg_mean_absolute_error', n_jobs=1)
        gs_ridge.fit(XD, yD, groups=D['subject#'])
        t0 = time.time()
        m_ridge = gs_ridge.best_estimator_.fit(XD, yD)
        t_ridge = time.time() - t0
        pred_ridge = m_ridge.predict(XT)
        records.append({
            'seed': seed,
            'model': 'Ridge',
            'paradigm': 'Centralized',
            'fit_s': t_ridge,
            **eval_reg_metrics(yT, pred_ridge)
        })
        
        # 2. Centralized SVR
        svr_pipe = make_pipeline(StandardScaler(), SVR())
        svr_grid = {'svr__C': [0.1, 1.0, 10.0], 'svr__epsilon': [0.1, 0.5, 1.0]}
        gs_svr = GridSearchCV(svr_pipe, svr_grid, cv=GroupKFold(5), scoring='neg_mean_absolute_error', n_jobs=1)
        gs_svr.fit(XD, yD, groups=D['subject#'])
        t0 = time.time()
        m_svr = gs_svr.best_estimator_.fit(XD, yD)
        t_svr = time.time() - t0
        pred_svr = m_svr.predict(XT)
        records.append({
            'seed': seed,
            'model': 'SVR',
            'paradigm': 'Centralized',
            'fit_s': t_svr,
            **eval_reg_metrics(yT, pred_svr)
        })
        
        # 3. Centralized Random Forest
        rf_grid = {'max_depth': [6, 12, None], 'min_samples_leaf': [2, 10]}
        gs_rf = GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                             rf_grid, cv=GroupKFold(5), scoring='neg_mean_absolute_error', n_jobs=1)
        gs_rf.fit(XD, yD, groups=D['subject#'])
        t0 = time.time()
        m_rf = gs_rf.best_estimator_.fit(XD, yD)
        t_rf = time.time() - t0
        pred_rf = m_rf.predict(XT)
        records.append({
            'seed': seed,
            'model': 'Random Forest',
            'paradigm': 'Centralized',
            'fit_s': t_rf,
            **eval_reg_metrics(yT, pred_rf)
        })
        
        # 4. Centralized Extra Trees
        et_grid = {'max_depth': [6, 12, None], 'min_samples_leaf': [2, 10]}
        gs_et = GridSearchCV(ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                             et_grid, cv=GroupKFold(5), scoring='neg_mean_absolute_error', n_jobs=1)
        gs_et.fit(XD, yD, groups=D['subject#'])
        t0 = time.time()
        m_et = gs_et.best_estimator_.fit(XD, yD)
        t_et = time.time() - t0
        pred_et = m_et.predict(XT)
        records.append({
            'seed': seed,
            'model': 'Extra Trees',
            'paradigm': 'Centralized',
            'fit_s': t_et,
            **eval_reg_metrics(yT, pred_et)
        })
        
        # 5. Federated Linear Regression (K=26 single-subject clients)
        sc = StandardScaler().fit(XD)
        ZD = sc.transform(XD)
        ZT = sc.transform(XT)
        clients_26 = [D[D['subject#'] == s].index.values for s in dv]
        ZD_c26 = [ZD[idx] for idx in clients_26]
        yD_c26 = [yD[idx] for idx in clients_26]
        t_fed26, mets_fed26 = run_federated_ridge(ZD_c26, yD_c26, ZT, yT)
        records.append({
            'seed': seed,
            'model': 'FedAvg-Linear (K=26)',
            'paradigm': 'Federated',
            'fit_s': t_fed26,
            **mets_fed26
        })
        
        # 6. Federated Linear Regression (K=5 multi-patient clinics)
        perm = np.random.RandomState(seed + 1000).permutation(dv)
        splits = np.array_split(perm, 5)
        clients_5 = [D[D['subject#'].isin(s)].index.values for s in splits]
        ZD_c5 = [ZD[idx] for idx in clients_5]
        yD_c5 = [yD[idx] for idx in clients_5]
        t_fed5, mets_fed5 = run_federated_ridge(ZD_c5, yD_c5, ZT, yT)
        records.append({
            'seed': seed,
            'model': 'FedAvg-Linear (K=5 clinics)',
            'paradigm': 'Federated',
            'fit_s': t_fed5,
            **mets_fed5
        })
        
        print(f"Seed {seed}/{n_seeds-1} completed.", flush=True)
        
    df_runs = pd.DataFrame(records)
    out_runs_path = os.path.join(R, "exp_c_regression_runs.csv")
    df_runs.to_csv(out_runs_path, index=False)
    
    # Summary
    metrics_cols = ['mae', 'rmse', 'r2', 'pearson_r', 'spearman_rho']
    def ci_str(vals):
        v = np.asarray(vals, float)
        m = v.mean()
        if len(v) > 1:
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        else:
            h = 0.0
        return f"{m:.3f} +/- {h:.3f}"
        
    summary_rows = []
    for model, grp in df_runs.groupby('model'):
        s_row = {'model': model, 'paradigm': grp.paradigm.iloc[0], 'fit_s_avg': grp.fit_s.mean()}
        for mc in metrics_cols:
            s_row[mc + '_mean'] = grp[mc].mean()
            s_row[mc + '_ci'] = ci_str(grp[mc])
        summary_rows.append(s_row)
        
    df_summary = pd.DataFrame(summary_rows)
    out_summary_path = os.path.join(R, "exp_c_regression_summary.csv")
    df_summary.to_csv(out_summary_path, index=False)
    
    print(f"Experiment C completed successfully!")
    print(f"Outputs written to:\n  {out_runs_path}\n  {out_summary_path}")
    return df_summary

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_c(n)


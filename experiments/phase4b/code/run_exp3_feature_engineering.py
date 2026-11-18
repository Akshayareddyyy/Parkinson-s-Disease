"""
Phase 4B - Experiment 3: Acoustic Feature Engineering & Preprocessing
Evaluates:
  1. Baseline (19 Features, StandardScaler)
  2. Acoustic Domain Interactions (23 Features: Jitter/Shimmer, NHR/HNR, PPE*RPDE, DFA*PPE)
  3. QuantileTransformer (Non-linear rank normalization)
  4. Yeo-Johnson PowerTransformer
  5. Yeo-Johnson + Correlation Filter (|r| < 0.85 on training fold)
  6. Yeo-Johnson + Mutual Information Selection
  7. Yeo-Johnson + L1 Sparsity Selection
Evaluates both Extra Trees (Classification) and ElasticNet (Regression).
Strict compliance: All transformers and selectors fitted strictly within training folds.
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, mutual_info_regression, SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline

from common import make_sets
from phase4b_common import (
    load_data, PREDICTORS, ENGINEERED_PREDICTORS, add_acoustic_interactions,
    eval_classification_metrics, eval_regression_metrics,
    compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.keep_indices_ = None
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.keep_indices_ = [i for i in range(X.shape[1]) if i not in to_drop]
        if len(self.keep_indices_) == 0:
            self.keep_indices_ = list(range(X.shape[1]))
        return self
        
    def transform(self, X):
        return X[:, self.keep_indices_]

def run_experiment_3(n_seeds=10):
    print("=" * 80)
    print("PHASE 4B - EXPERIMENT 3: ACOUSTIC FEATURE ENGINEERING & SELECTION")
    print("Domain Ratios: Jitter/Shimmer, NHR/HNR, Entropy & DFA interactions")
    print("Transformers: Quantile, Yeo-Johnson, Robust; Selectors: CorrFilter, MI, L1")
    print("=" * 80)
    
    exp3_runs = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        # Base 19 features
        XD_19 = D[PREDICTORS].values
        XT_19 = T[PREDICTORS].values
        
        # Engineered 23 features
        D_eng = add_acoustic_interactions(D)
        T_eng = add_acoustic_interactions(T)
        XD_23 = D_eng[ENGINEERED_PREDICTORS].values
        XT_23 = T_eng[ENGINEERED_PREDICTORS].values
        
        yD_cls, yT_cls = D.y.values, T.y.values
        yD_reg, yT_reg = D['motor_UPDRS'].values, T['motor_UPDRS'].values
        
        pipelines_cls = {
            'Baseline (19 Feats, StandardScaler)': (
                Pipeline([('scaler', StandardScaler())]), XD_19, XT_19
            ),
            'Acoustic Interactions (23 Feats)': (
                Pipeline([('scaler', StandardScaler())]), XD_23, XT_23
            ),
            'Quantile Normalization (19 Feats)': (
                Pipeline([('scaler', QuantileTransformer(output_distribution='normal', n_quantiles=500, random_state=seed))]), XD_19, XT_19
            ),
            'Yeo-Johnson Transform (19 Feats)': (
                Pipeline([('scaler', PowerTransformer(method='yeo-johnson'))]), XD_19, XT_19
            ),
            'Yeo-Johnson + Correlation Filter (<0.85)': (
                Pipeline([
                    ('scaler', PowerTransformer(method='yeo-johnson')),
                    ('select', CorrelationFilter(threshold=0.85))
                ]), XD_19, XT_19
            ),
            'Yeo-Johnson + Mutual Information': (
                Pipeline([
                    ('scaler', PowerTransformer(method='yeo-johnson')),
                    ('select', SelectPercentile(mutual_info_classif, percentile=60))
                ]), XD_19, XT_19
            ),
            'Yeo-Johnson + L1 Sparsity': (
                Pipeline([
                    ('scaler', PowerTransformer(method='yeo-johnson')),
                    ('select', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=seed)))
                ]), XD_19, XT_19
            )
        }
        
        # Classification evaluation (Extra Trees)
        for p_name, (pipe, X_tr, X_te) in pipelines_cls.items():
            t0 = time.time()
            pipe.fit(X_tr, yD_cls)
            Z_tr = pipe.transform(X_tr)
            Z_te = pipe.transform(X_te)
            n_feats = Z_tr.shape[1]
            
            m_cls = ExtraTreesClassifier(n_estimators=150, max_depth=8, min_samples_leaf=2,
                                         class_weight='balanced', random_state=seed, n_jobs=-1)
            m_cls.fit(Z_tr, yD_cls)
            fit_s = time.time() - t0
            
            prob_T = m_cls.predict_proba(Z_te)[:, 1]
            pred_T = (prob_T > 0.5).astype(int)
            mets = eval_classification_metrics(yT_cls, pred_T, prob_T)
            
            exp3_runs.append({
                'seed': seed, 'task': 'Classification', 'pipeline': p_name,
                'n_features': n_feats, 'fit_s': fit_s,
                'n_train_records': len(X_tr), 'n_calib_records': 0, 'n_val_records': len(V), 'n_test_records': len(X_te),
                **mets
            })
            
        # Regression evaluation (ElasticNet)
        pipelines_reg = {
            'Baseline (19 Feats, StandardScaler)': (
                Pipeline([('scaler', StandardScaler())]), XD_19, XT_19
            ),
            'Acoustic Interactions (23 Feats)': (
                Pipeline([('scaler', StandardScaler())]), XD_23, XT_23
            ),
            'RobustScaler (19 Feats)': (
                Pipeline([('scaler', RobustScaler())]), XD_19, XT_19
            ),
            'Yeo-Johnson Transform (19 Feats)': (
                Pipeline([('scaler', PowerTransformer(method='yeo-johnson'))]), XD_19, XT_19
            ),
            'Yeo-Johnson + Correlation Filter (<0.85)': (
                Pipeline([
                    ('scaler', PowerTransformer(method='yeo-johnson')),
                    ('select', CorrelationFilter(threshold=0.85))
                ]), XD_19, XT_19
            ),
            'Yeo-Johnson + L1 Sparsity': (
                Pipeline([
                    ('scaler', PowerTransformer(method='yeo-johnson')),
                    ('select', SelectFromModel(Lasso(alpha=0.05, max_iter=2000, random_state=seed), threshold=1e-4))
                ]), XD_19, XT_19
            )
        }
        
        for p_name, (pipe, X_tr, X_te) in pipelines_reg.items():
            t0 = time.time()
            pipe.fit(X_tr, yD_reg)
            Z_tr = pipe.transform(X_tr)
            Z_te = pipe.transform(X_te)
            n_feats = Z_tr.shape[1]
            
            m_reg = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed, max_iter=2000)
            m_reg.fit(Z_tr, yD_reg)
            fit_s = time.time() - t0
            
            pred_T = m_reg.predict(Z_te)
            mets = eval_regression_metrics(yT_reg, pred_T)
            sub_mets = compute_per_subject_regression_metrics(T, pred_T)
            
            exp3_runs.append({
                'seed': seed, 'task': 'Regression', 'pipeline': p_name,
                'n_features': n_feats, 'fit_s': fit_s,
                'n_train_records': len(X_tr), 'n_calib_records': 0, 'n_val_records': len(V), 'n_test_records': len(X_te),
                **mets,
                **sub_mets
            })
            
        print(f"Exp 3 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_runs = pd.DataFrame(exp3_runs)
    out_runs = os.path.join(R, "exp3_feature_engineering_runs.csv")
    out_sum = os.path.join(R, "exp3_feature_engineering_summary.csv")
    df_runs.to_csv(out_runs, index=False)
    
    # Summary
    sum_rows = []
    for (task, pipe), grp in df_runs.groupby(['task', 'pipeline']):
        r = {
            'task': task,
            'pipeline': pipe,
            'n_features_avg': grp['n_features'].mean(),
            'fit_s_avg': grp['fit_s'].mean(),
        }
        if task == 'Classification':
            r.update({
                'roc_auc_mean': grp['auc'].mean(),
                'roc_auc_std': grp['auc'].std(),
                'prauc_mean': grp['prauc'].mean(),
                'bacc_mean': grp['bacc'].mean(),
                'bacc_std': grp['bacc'].std(),
                'mcc_mean': grp['mcc'].mean(),
                'mcc_std': grp['mcc'].std(),
                'brier_mean': grp['brier'].mean(),
                'brier_std': grp['brier'].std(),
                'log_loss_mean': grp['log_loss'].mean()
            })
        else:
            r.update({
                'mae_mean': grp['mae'].mean(),
                'mae_std': grp['mae'].std(),
                'rmse_mean': grp['rmse'].mean(),
                'rmse_std': grp['rmse'].std(),
                'r2_mean': grp['r2'].mean(),
                'r2_std': grp['r2'].std(),
                'pearson_r_mean': grp['pearson_r'].mean(),
                'spearman_rho_mean': grp['spearman_rho'].mean(),
                'mean_subj_mae_mean': grp['mean_subj_mae'].mean()
            })
        sum_rows.append(r)
        
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 3 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_3(ns)


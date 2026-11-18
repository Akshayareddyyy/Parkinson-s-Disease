"""
Phase 3B - Experiment 3: Feature Engineering, Transformations, & Fold-Contained Selection
Evaluates:
  Preprocessors:
    - StandardScaler
    - RobustScaler
    - PowerTransformer (Yeo-Johnson)
  Feature Selectors (Fitted strictly inside training folds):
    - No selection (19 features)
    - ANOVA F-selection (SelectKBest, k=10)
    - Mutual Information (SelectPercentile, percentile=60)
    - L1-based selection (SelectFromModel LassoCV / L1-Logistic)
    - Recursive Feature Elimination (RFE with Extra Trees)
    - Training-fold Correlation Filter (|r| > 0.90)
Logs selected feature subsets for every fold and seed.
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif,
    SelectPercentile, mutual_info_regression, mutual_info_classif,
    SelectFromModel, RFE
)
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from common import partition, make_sets
from phase3b_common import (
    load_data, PREDICTORS, eval_classification_metrics, eval_regression_metrics,
    compute_per_subject_regression_metrics
)

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Drops features with absolute Pearson correlation > threshold.
    Correlation matrix is computed STRICTLY on the training fold X.
    """
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.keep_indices_ = None
        self.dropped_indices_ = None
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.dropped_indices_ = to_drop
        self.keep_indices_ = [i for i in range(X.shape[1]) if i not in to_drop]
        return self
        
    def transform(self, X):
        return X[:, self.keep_indices_]

def run_experiment_3(n_seeds=10):
    print("=" * 70)
    print("PHASE 3B - EXPERIMENT 3: FEATURE ENGINEERING & FOLD-CONTAINED SELECTION")
    print(f"Seeds: {n_seeds}, Invariant: All transformations fitted inside training folds")
    print("=" * 70)
    
    exp3_records = []
    feature_selection_log = []
    
    # Preprocessor configurations
    preproc_configs = {
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'Yeo-Johnson': PowerTransformer(method='yeo-johnson')
    }
    
    # Feature selector configurations (Regression & Classification)
    def get_selectors(task, seed):
        if task == 'Regression':
            return {
                'None (All 19)': None,
                'ANOVA F-test (Top 10)': SelectKBest(score_func=f_regression, k=10),
                'Mutual Information': SelectPercentile(score_func=mutual_info_regression, percentile=60),
                'L1 Sparsity': SelectFromModel(Lasso(alpha=0.05, max_iter=2000, random_state=seed), threshold=1e-4),
                'RFE (Extra Trees)': RFE(estimator=ExtraTreesRegressor(n_estimators=50, random_state=seed, n_jobs=1), n_features_to_select=10, step=2),
                'Correlation Filter (>0.90)': CorrelationFilter(threshold=0.90)
            }
        else:
            return {
                'None (All 19)': None,
                'ANOVA F-test (Top 10)': SelectKBest(score_func=f_classif, k=10),
                'Mutual Information': SelectPercentile(score_func=mutual_info_classif, percentile=60),
                'L1 Sparsity': SelectFromModel(LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=2000, random_state=seed), threshold=1e-4),
                'RFE (Extra Trees)': RFE(estimator=ExtraTreesClassifier(n_estimators=50, random_state=seed, n_jobs=1), n_features_to_select=10, step=2),
                'Correlation Filter (>0.90)': CorrelationFilter(threshold=0.90)
            }
            
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        XD = D[PREDICTORS].values
        XT = T[PREDICTORS].values
        
        yD_reg = D['motor_UPDRS'].values
        yT_reg = T['motor_UPDRS'].values
        yD_cls = D.y.values
        yT_cls = T.y.values
        
        # Test across tasks
        for task in ['Regression', 'Classification']:
            selectors = get_selectors(task, seed)
            
            for p_name, preproc in preproc_configs.items():
                for s_name, selector in selectors.items():
                    steps = [('scaler', preproc)]
                    if selector is not None:
                        steps.append(('selector', selector))
                        
                    if task == 'Regression':
                        steps.append(('estimator', ExtraTreesRegressor(n_estimators=150, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=1)))
                    else:
                        steps.append(('estimator', ExtraTreesClassifier(n_estimators=150, max_depth=None, min_samples_leaf=1, random_state=seed, n_jobs=1)))
                        
                    pipe = Pipeline(steps)
                    
                    t0 = time.time()
                    if task == 'Regression':
                        pipe.fit(XD, yD_reg)
                        fit_s = time.time() - t0
                        pT = pipe.predict(XT)
                        mets = eval_regression_metrics(yT_reg, pT)
                        sub_mets = compute_per_subject_regression_metrics(T, pT)
                    else:
                        pipe.fit(XD, yD_cls)
                        fit_s = time.time() - t0
                        sT = pipe.predict_proba(XT)[:, 1]
                        pT = (sT > 0.5).astype(int)
                        mets = eval_classification_metrics(yT_cls, pT, sT)
                        sub_mets = {}
                        
                    # Extract selected feature names if applicable
                    sel_feats_list = PREDICTORS
                    if selector is not None:
                        if hasattr(pipe.named_steps['selector'], 'get_support'):
                            mask = pipe.named_steps['selector'].get_support()
                            sel_feats_list = [f for f, m in zip(PREDICTORS, mask) if m]
                        elif hasattr(pipe.named_steps['selector'], 'keep_indices_'):
                            keep_idx = pipe.named_steps['selector'].keep_indices_
                            sel_feats_list = [PREDICTORS[i] for i in keep_idx]
                            
                    feature_selection_log.append({
                        'seed': seed,
                        'task': task,
                        'preprocessor': p_name,
                        'selector': s_name,
                        'n_features_selected': len(sel_feats_list),
                        'selected_features': str(sel_feats_list)
                    })
                    
                    exp3_records.append({
                        'seed': seed,
                        'task': task,
                        'preprocessor': p_name,
                        'selector': s_name,
                        'n_features_selected': len(sel_feats_list),
                        'fit_s': fit_s,
                        **mets,
                        **sub_mets
                    })
                    
        print(f"Exp 3 Seed {seed}/{n_seeds-1} finished in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_exp3 = pd.DataFrame(exp3_records)
    df_feat_log = pd.DataFrame(feature_selection_log)
    
    out_runs = os.path.join(R, "exp3_feature_engineering_runs.csv")
    out_log = os.path.join(R, "exp3_selected_features_log.csv")
    out_sum = os.path.join(R, "exp3_feature_engineering_summary.csv")
    
    df_exp3.to_csv(out_runs, index=False)
    df_feat_log.to_csv(out_log, index=False)
    
    # Consolidated summary
    sum_rows = []
    for (task, p_name, s_name), grp in df_exp3.groupby(['task', 'preprocessor', 'selector']):
        row = {
            'task': task,
            'preprocessor': p_name,
            'selector': s_name,
            'n_features_avg': grp['n_features_selected'].mean(),
            'fit_s_avg': grp['fit_s'].mean()
        }
        if task == 'Regression':
            row.update({
                'mae_mean': grp['mae'].mean(), 'mae_std': grp['mae'].std(),
                'rmse_mean': grp['rmse'].mean(), 'rmse_std': grp['rmse'].std(),
                'r2_mean': grp['r2'].mean(), 'r2_std': grp['r2'].std(),
                'mean_subj_mae_mean': grp['mean_subj_mae'].mean()
            })
        else:
            row.update({
                'roc_auc_mean': grp['auc'].mean(), 'roc_auc_std': grp['auc'].std(),
                'bacc_mean': grp['bacc'].mean(), 'bacc_std': grp['bacc'].std(),
                'mcc_mean': grp['mcc'].mean(), 'mcc_std': grp['mcc'].std(),
                'f1m_mean': grp['f1m'].mean(), 'f1m_std': grp['f1m'].std()
            })
        sum_rows.append(row)
        
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 3 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_log}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_3(ns)

"""
Phase 4B - Experiment 1: Advanced Centralized Classification
Evaluates:
  1. Extra Trees (Balanced)
  2. Random Forest (Balanced)
  3. CatBoost Classifier (Balanced)
  4. XGBoost Classifier (Balanced)
  5. HistGradientBoosting (Balanced)
  6. RBF Support Vector Classifier (Calibrated)
  7. Regularized Logistic Regression (Balanced)
Includes probability calibration (Platt sigmoid scaling on 5-fold inner cross-validation out-of-fold scores strictly on development subjects).
Reports Brier score and Log Loss alongside ROC-AUC, PR-AUC, BalAcc, Sensitivity, Specificity, MCC.
"""

import os, sys, time, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from common import make_sets
from phase4b_common import load_data, PREDICTORS, eval_classification_metrics

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(R, exist_ok=True)
df_all = load_data()

def run_experiment_1(n_seeds=10):
    print("=" * 80)
    print("PHASE 4B - EXPERIMENT 1: ADVANCED CENTRALIZED CLASSIFICATION")
    print("Features: 19 Permitted Predictors, Protocol: GroupKFold(5) on Dev Cohort")
    print("Probability Calibration: Platt Sigmoid (Inner Folds), Metrics: Brier + Log Loss")
    print("=" * 80)
    
    exp1_runs = []
    
    for seed in range(n_seeds):
        t0_seed = time.time()
        D, V, T, thr = make_sets(df_all, seed)
        
        XD = D[PREDICTORS].values
        XV = V[PREDICTORS].values
        XT = T[PREDICTORS].values
        yD = D.y.values
        yV = V.y.values
        yT = T.y.values
        groups_D = D['subject#'].values
        
        # Preprocessing fitted strictly on development
        sc = StandardScaler().fit(XD)
        ZD = sc.transform(XD)
        ZT = sc.transform(XT)
        
        # Base candidate models with balanced class weighting
        models = {
            'Extra Trees (Balanced)': ExtraTreesClassifier(
                n_estimators=150, max_depth=8, min_samples_leaf=2,
                class_weight='balanced', random_state=seed, n_jobs=-1
            ),
            'Random Forest (Balanced)': RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_leaf=2,
                class_weight='balanced', random_state=seed, n_jobs=-1
            ),
            'CatBoost (Balanced)': CatBoostClassifier(
                iterations=150, depth=5, auto_class_weights='Balanced',
                random_state=seed, verbose=0, thread_count=-1
            ),
            'XGBoost (Balanced)': XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                random_state=seed, eval_metric='logloss', n_jobs=-1
            ),
            'HistGradientBoosting (Balanced)': HistGradientBoostingClassifier(
                max_iter=150, class_weight='balanced', random_state=seed
            ),
            'RBF SVM (Calibrated)': SVC(
                kernel='rbf', C=1.0, class_weight='balanced', random_state=seed
            ),
            'Logistic Regression (Balanced)': LogisticRegression(
                C=1.0, class_weight='balanced', max_iter=1000, random_state=seed
            )
        }
        
        cv_inner = GroupKFold(n_splits=5)
        
        for m_name, model in models.items():
            t0 = time.time()
            # 1. Fit main model on development
            model.fit(ZD, yD)
            
            # 2. Platt Sigmoid Calibration using 5-fold cross-validation strictly on development
            if hasattr(model, "predict_proba"):
                s_oof = cross_val_predict(model, ZD, yD, cv=cv_inner, groups=groups_D,
                                          method='predict_proba', n_jobs=-1)[:, 1]
                s_T = model.predict_proba(ZT)[:, 1]
            else:
                s_oof = cross_val_predict(model, ZD, yD, cv=cv_inner, groups=groups_D,
                                          method='decision_function', n_jobs=-1)
                s_T = model.decision_function(ZT)
                
            calibrator = LogisticRegression().fit(s_oof.reshape(-1, 1), yD)
            prob_T = calibrator.predict_proba(s_T.reshape(-1, 1))[:, 1]
            pred_T = (prob_T > 0.5).astype(int)
            fit_s = time.time() - t0
            
            mets = eval_classification_metrics(yT, pred_T, prob_T)
            
            exp1_runs.append({
                'seed': seed,
                'task': 'Classification',
                'model': m_name,
                'n_train_records': len(ZD),
                'n_calib_records': 0,
                'n_val_records': len(V),
                'n_test_records': len(ZT),
                'calibration': 'Platt Sigmoid (OOF GroupKFold-5)',
                'fit_s': fit_s,
                **mets
            })
            
        print(f"Exp 1 Seed {seed}/{n_seeds-1} completed in {time.time()-t0_seed:.1f}s.", flush=True)
        
    df_runs = pd.DataFrame(exp1_runs)
    out_runs = os.path.join(R, "exp1_advanced_classification_runs.csv")
    out_sum = os.path.join(R, "exp1_advanced_classification_summary.csv")
    df_runs.to_csv(out_runs, index=False)
    
    # Consolidated summary
    sum_rows = []
    for m_name, grp in df_runs.groupby('model'):
        sum_rows.append({
            'model': m_name,
            'fit_s_avg': grp['fit_s'].mean(),
            'roc_auc_mean': grp['auc'].mean(),
            'roc_auc_std': grp['auc'].std(),
            'prauc_mean': grp['prauc'].mean(),
            'bacc_mean': grp['bacc'].mean(),
            'bacc_std': grp['bacc'].std(),
            'sens_mean': grp['sens'].mean(),
            'spec_mean': grp['spec'].mean(),
            'f1m_mean': grp['f1m'].mean(),
            'f1m_std': grp['f1m'].std(),
            'mcc_mean': grp['mcc'].mean(),
            'mcc_std': grp['mcc'].std(),
            'brier_mean': grp['brier'].mean(),
            'brier_std': grp['brier'].std(),
            'log_loss_mean': grp['log_loss'].mean(),
            'log_loss_std': grp['log_loss'].std()
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_sum, index=False)
    
    print("=== EXPERIMENT 1 COMPLETED SUCCESSFULLY ===")
    print(f"Outputs written to:\n  {out_runs}\n  {out_sum}")
    return df_sum

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_experiment_1(ns)

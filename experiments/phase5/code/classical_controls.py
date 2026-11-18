"""
Phase 5: Matched Classical Controls
Provides:
  1. ExtraTreesClassifier (9 features, balanced)
  2. Classical RBF-SVM (9 features, calibrated)
  3. ElasticNet Regressor (9 features, alpha=0.1, l1_ratio=0.5)
  4. Classical RBF-SVR (9 features)
All models evaluate on the identical 9 features and subject-wise partitions.
"""

import time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import ElasticNet
from sklearn.calibration import CalibratedClassifierCV

from metrics import evaluate_classification, evaluate_regression

def run_classical_classification_controls(X_tr, y_tr, X_val, y_val, X_te, y_te, seed):
    """
    Fits and evaluates matched classical classification controls:
      - Extra Trees (Balanced)
      - Classical RBF-SVM (Calibrated)
    Returns: dict of model results for test and validation sets.
    """
    results = {}
    
    # 1. Extra Trees Classifier (Matched to Phase 4B Exp 3 benchmark)
    t0 = time.perf_counter()
    et = ExtraTreesClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=2,
        class_weight='balanced', random_state=seed, n_jobs=2
    )
    et.fit(X_tr, y_tr)
    fit_time_et = time.perf_counter() - t0
    
    # Validation evaluation
    probs_val_raw = et.predict_proba(X_val)
    val_probs_et = probs_val_raw[:, 1] if probs_val_raw.shape[1] > 1 else np.zeros(len(X_val))
    val_metrics_et = evaluate_classification(y_val, val_probs_et)
    
    # Test evaluation
    probs_te_raw = et.predict_proba(X_te)
    te_probs_et = probs_te_raw[:, 1] if probs_te_raw.shape[1] > 1 else np.zeros(len(X_te))
    te_metrics_et = evaluate_classification(y_te, te_probs_et)
    te_metrics_et['fit_time'] = fit_time_et
    
    results['Extra Trees (9 Feats)'] = {
        'model': et,
        'val_metrics': val_metrics_et,
        'test_metrics': te_metrics_et,
        'test_probs': te_probs_et
    }
    
    # 2. Classical RBF-SVM (Calibrated)
    t0 = time.perf_counter()
    # Tune C on validation data from small grid {0.5, 1.0, 2.0}
    best_c = 1.0
    best_val_auc = -1.0
    for c_cand in [0.5, 1.0, 2.0]:
        base_svm = SVC(C=c_cand, kernel='rbf', gamma='scale', probability=True, random_state=seed)
        base_svm.fit(X_tr, y_tr)
        val_probs_cand = base_svm.predict_proba(X_val)
        val_p = val_probs_cand[:, 1] if val_probs_cand.shape[1] > 1 else np.zeros(len(X_val))
        val_auc_cand = evaluate_classification(y_val, val_p)['roc_auc']
        if val_auc_cand > best_val_auc:
            best_val_auc = val_auc_cand
            best_c = c_cand
            
    svm = SVC(C=best_c, kernel='rbf', gamma='scale', probability=True, random_state=seed)
    svm.fit(X_tr, y_tr)
    fit_time_svm = time.perf_counter() - t0
    
    val_probs_raw = svm.predict_proba(X_val)
    val_probs_svm = val_probs_raw[:, 1] if val_probs_raw.shape[1] > 1 else np.zeros(len(X_val))
    val_metrics_svm = evaluate_classification(y_val, val_probs_svm)
    
    te_probs_raw = svm.predict_proba(X_te)
    te_probs_svm = te_probs_raw[:, 1] if te_probs_raw.shape[1] > 1 else np.zeros(len(X_te))
    te_metrics_svm = evaluate_classification(y_te, te_probs_svm)
    te_metrics_svm['fit_time'] = fit_time_svm
    te_metrics_svm['best_c'] = best_c
    
    results['Classical RBF-SVM (9 Feats)'] = {
        'model': svm,
        'val_metrics': val_metrics_svm,
        'test_metrics': te_metrics_svm,
        'test_probs': te_probs_svm
    }
    
    return results

def run_classical_regression_controls(X_tr, y_tr, X_val, y_val, X_te, y_te, df_te, seed):
    """
    Fits and evaluates matched classical regression controls:
      - ElasticNet (alpha=0.1, l1_ratio=0.5)
      - Classical RBF-SVR
    Returns: dict of model results for test and validation sets.
    """
    results = {}
    
    # 1. ElasticNet Regressor (Matched to Phase 4B Exp 3 benchmark)
    t0 = time.perf_counter()
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=seed)
    enet.fit(X_tr, y_tr)
    fit_time_enet = time.perf_counter() - t0
    
    val_preds_enet = enet.predict(X_val)
    val_metrics_enet = evaluate_regression(y_val, val_preds_enet)
    
    te_preds_enet = enet.predict(X_te)
    te_metrics_enet = evaluate_regression(y_te, te_preds_enet, df_records=df_te)
    te_metrics_enet['fit_time'] = fit_time_enet
    
    results['ElasticNet (9 Feats)'] = {
        'model': enet,
        'val_metrics': val_metrics_enet,
        'test_metrics': te_metrics_enet,
        'test_preds': te_preds_enet
    }
    
    # 2. Classical RBF-SVR
    t0 = time.perf_counter()
    # Tune C on validation data from small grid {5.0, 10.0, 20.0}
    best_c = 10.0
    best_val_mae = float('inf')
    for c_cand in [5.0, 10.0, 20.0]:
        svr_cand = SVR(C=c_cand, kernel='rbf', gamma='scale', epsilon=0.1)
        svr_cand.fit(X_tr, y_tr)
        val_preds_cand = svr_cand.predict(X_val)
        val_mae_cand = evaluate_regression(y_val, val_preds_cand)['mae']
        if val_mae_cand < best_val_mae:
            best_val_mae = val_mae_cand
            best_c = c_cand
            
    svr = SVR(C=best_c, kernel='rbf', gamma='scale', epsilon=0.1)
    svr.fit(X_tr, y_tr)
    fit_time_svr = time.perf_counter() - t0
    
    val_preds_svr = svr.predict(X_val)
    val_metrics_svr = evaluate_regression(y_val, val_preds_svr)
    
    te_preds_svr = svr.predict(X_te)
    te_metrics_svr = evaluate_regression(y_te, te_preds_svr, df_records=df_te)
    te_metrics_svr['fit_time'] = fit_time_svr
    te_metrics_svr['best_c'] = best_c
    
    results['Classical RBF-SVR (9 Feats)'] = {
        'model': svr,
        'val_metrics': val_metrics_svr,
        'test_metrics': te_metrics_svr,
        'test_preds': te_preds_svr
    }
    
    return results

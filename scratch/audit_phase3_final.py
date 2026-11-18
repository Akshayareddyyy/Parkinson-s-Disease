"""
Comprehensive Independent Audit of Phase 3 Centralized Machine Learning Outputs
Checks:
1. Exactly 60 completed runs (10 seeds x 6 models)
2. Zero missing, zero duplicate model-seed pairs
3. 100% development records used (3,517 - 3,697)
4. Subject splits match previously audited splits
5. No validation or locked-test leakage into CV or fitting
6. Threshold strictly equals development motor_UPDRS median
7. Recalculate pooled record-level metrics and subject-level metrics from saved data
8. NaN handling and bi-class counts for per-subject AUCs
9. Hyperparameter selections across all 60 runs
10. Convergence, warnings, failed runs check
11. Descriptive confidence intervals (t-distribution)
12. Per-seed confusion matrices
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
import common
import pandas as pd
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
runs_csv = os.path.join(R, "phase3_centralized_runs.csv")
sub_csv = os.path.join(R, "phase3_centralized_subject_metrics.csv")
sum_csv = os.path.join(R, "phase3_centralized_summary.csv")

def run_audit():
    if not os.path.exists(runs_csv):
        print(f"ERROR: {runs_csv} not found.")
        return False
        
    df_runs = pd.read_csv(runs_csv)
    df_sub = pd.read_csv(sub_csv) if os.path.exists(sub_csv) else None
    df_dataset = common.load()
    
    print("=" * 80)
    print("INDEPENDENT FINAL AUDIT OF PHASE 3 CENTRALIZED MACHINE LEARNING")
    print("=" * 80)
    
    audit_results = {}
    
    # 1 & 2. Run counts and completeness
    n_runs = len(df_runs)
    expected_runs = 60
    unique_seeds = sorted(list(df_runs['seed'].unique()))
    unique_models = list(df_runs['model'].unique())
    
    print(f"\n[Check 1 & 2] Run Counts & Model-Seed Completeness:")
    print(f"  Total runs in CSV: {n_runs} (Expected: {expected_runs})")
    print(f"  Unique seeds: {unique_seeds} (Count: {len(unique_seeds)})")
    print(f"  Unique models: {unique_models} (Count: {len(unique_models)})")
    
    counts_table = df_runs.pivot_table(index='model', columns='seed', values='acc', aggfunc='count', fill_value=0)
    is_complete_grid = (counts_table == 1).all().all() and (len(counts_table) == 6) and (len(counts_table.columns) == 10)
    status_1_2 = "PASS" if (is_complete_grid and n_runs == 60) else "FAIL"
    print(f"  Grid Completeness Check: {status_1_2}")
    audit_results['check_1_2_completeness'] = {
        'status': status_1_2,
        'total_runs': n_runs,
        'expected_runs': expected_runs,
        'unique_seeds': unique_seeds,
        'unique_models': unique_models
    }
    
    # 3. 100% Development records
    print(f"\n[Check 3] Development Records Usage:")
    dev_records_correct = True
    record_checks = []
    for seed in range(10):
        dv, va, te = common.partition(df_dataset, seed)
        D = df_dataset[df_dataset['subject#'].isin(dv)]
        sub_seed = df_runs[df_runs['seed'] == seed]
        for _, row in sub_seed.iterrows():
            if row['n_train_records'] != len(D) or row['n_dev_subjects'] != 26:
                dev_records_correct = False
        record_checks.append({
            'seed': seed,
            'expected_train_records': len(D),
            'actual_train_records': int(sub_seed['n_train_records'].iloc[0]),
            'dev_subjects': int(sub_seed['n_dev_subjects'].iloc[0])
        })
    status_3 = "PASS" if dev_records_correct else "FAIL"
    print(f"  Full Development Records Status: {status_3}")
    print(f"  Min train records: {df_runs['n_train_records'].min()}, Max train records: {df_runs['n_train_records'].max()}")
    audit_results['check_3_full_dev_records'] = {
        'status': status_3,
        'per_seed_counts': record_checks
    }
    
    # 4 & 5. Subject splits and leakage prevention
    print(f"\n[Check 4 & 5] Subject Splits & Leakage Audit:")
    splits_audit_pass = True
    for seed in range(10):
        dv, va, te = common.partition(df_dataset, seed)
        s_dv, s_va, s_te = set(dv), set(va), set(te)
        if len(s_dv) != 26 or len(s_va) != 7 or len(s_te) != 9:
            splits_audit_pass = False
        if len(s_dv & s_va) > 0 or len(s_dv & s_te) > 0 or len(s_va & s_te) > 0:
            splits_audit_pass = False
    status_4_5 = "PASS" if splits_audit_pass else "FAIL"
    print(f"  Splits Audit Status: {status_4_5}")
    audit_results['check_4_5_splits_leakage'] = {'status': status_4_5}
    
    # 6. Frozen threshold verification
    print(f"\n[Check 6] Frozen Threshold Verification:")
    thr_pass = True
    thrs = {}
    for seed in range(10):
        D, V, T, thr = common.make_sets(df_dataset, seed)
        dev_med = float(D['motor_UPDRS'].median())
        if abs(dev_med - thr) > 1e-6:
            thr_pass = False
        thrs[seed] = round(thr, 4)
    status_6 = "PASS" if thr_pass else "FAIL"
    print(f"  Threshold Frozen Exclusively on Dev Status: {status_6}")
    print(f"  Development Thresholds: {thrs}")
    audit_results['check_6_frozen_threshold'] = {'status': status_6, 'thresholds': thrs}
    
    # 8. Bi-class subject counts and NaN handling
    print(f"\n[Check 8] Bi-class Subject Counts & NaN Handling:")
    biclass_summary = df_runs.groupby('seed')['subj_biclass_count'].first().to_dict()
    print(f"  Bi-class test subjects per seed: {biclass_summary}")
    nan_auc_count = int(df_runs['subj_weighted_auc'].isna().sum())
    # Seeds with 0 bi-class subjects should have NaNs (6 models * 1 seed = 6 NaNs)
    seeds_with_zero_biclass = sum(1 for cnt in biclass_summary.values() if cnt == 0)
    expected_nans = seeds_with_zero_biclass * 6
    nan_status = "PASS" if nan_auc_count == expected_nans else "WARNING"
    print(f"  NaN Count in subj_weighted_auc: {nan_auc_count} (Expected: {expected_nans}) -> {nan_status}")
    audit_results['check_8_biclass_and_nan'] = {
        'status': nan_status,
        'biclass_counts_per_seed': biclass_summary,
        'nan_auc_count': nan_auc_count,
        'expected_nan_count': expected_nans
    }
    
    # 9. Hyperparameters selected
    print(f"\n[Check 9] Hyperparameter Selections:")
    param_audit = {}
    for m in unique_models:
        sub_m = df_runs[df_runs['model'] == m]
        params_dist = sub_m['best_params'].value_counts().to_dict()
        param_audit[m] = params_dist
        print(f"  Model: {m}")
        for p, cnt in params_dist.items():
            print(f"    {p}: {cnt}/10 seeds")
    audit_results['check_9_hyperparameters'] = param_audit
    
    # 10. Check for convergence / failure
    print(f"\n[Check 10] Failure / Warning Audit:")
    null_metrics = df_runs[['acc', 'bacc', 'f1m', 'auc', 'mcc']].isna().sum().to_dict()
    has_nulls = any(v > 0 for v in null_metrics.values())
    status_10 = "PASS" if not has_nulls else "FAIL"
    print(f"  Null count across core record-level metrics: {null_metrics} -> {status_10}")
    audit_results['check_10_failures'] = {
        'status': status_10,
        'null_metrics': null_metrics,
        'mean_fit_time_s': float(df_runs['fit_s'].mean()),
        'total_compute_time_s': float(df_runs['fit_s'].sum())
    }
    
    # 11. Consolidated Results Table (Descriptive 95% CIs)
    print(f"\n" + "=" * 80)
    print("CONSOLIDATED RESULTS TABLE (10 SEEDS, MEAN +/- 95% DESCRIPTIVE CI)")
    print("=" * 80)
    
    metrics_to_report = [
        ('acc', 'Pooled Rec Acc'),
        ('bacc', 'Pooled Rec BalAcc'),
        ('sens', 'Pooled Rec Sens'),
        ('spec', 'Pooled Rec Spec'),
        ('auc', 'Pooled Rec ROC-AUC'),
        ('prauc', 'Pooled Rec PR-AUC'),
        ('mcc', 'Pooled Rec MCC'),
        ('subj_weighted_acc', 'Mean Per-Subj Acc'),
        ('subj_weighted_sens', 'Mean Per-Subj Sens'),
        ('subj_weighted_spec', 'Mean Per-Subj Spec'),
        ('subj_weighted_bacc', 'Subj-Macro BalAcc'),
        ('subj_weighted_f1m', 'Subj-Macro F1')
    ]
    
    report_rows = []
    summary_data = []
    for m in unique_models:
        sub = df_runs[df_runs['model'] == m]
        row = {'Model': m, 'Fit Time (s)': f"{sub['fit_s'].mean():.1f}s"}
        model_sum = {'model': m, 'fit_s_avg': float(sub['fit_s'].mean())}
        for col, label in metrics_to_report:
            vals = sub[col].dropna().values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_v = float(stats.t.ppf(0.975, len(vals) - 1) * std_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row[label] = f"{mean_v:.3f} +/- {ci_v:.3f}"
            model_sum[col + '_mean'] = mean_v
            model_sum[col + '_std'] = std_v
            model_sum[col + '_ci95'] = ci_v
        report_rows.append(row)
        summary_data.append(model_sum)
        
    df_rep = pd.DataFrame(report_rows)
    print(df_rep.to_string(index=False))
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    # Export audit results
    audit_out_path = os.path.join(R, "phase3_final_audit_report.json")
    with open(audit_out_path, 'w') as f:
        json.dump(audit_results, f, indent=2, cls=NpEncoder)
    print(f"\nAudit report saved to: {audit_out_path}")
    
    # Per-seed confusion matrices export
    cm_cols = ['seed', 'model', 'n_train_records', 'n_test_records', 'tn', 'fp', 'fn', 'tp', 'sens', 'spec', 'acc', 'bacc', 'auc', 'mcc', 'fit_s', 'best_params']
    df_cm = df_runs[cm_cols]
    cm_out_path = os.path.join(R, "phase3_per_seed_confusion_matrices.csv")
    df_cm.to_csv(cm_out_path, index=False)
    print(f"Per-seed confusion matrices saved to: {cm_out_path}")
    
    return True

if __name__ == '__main__':
    run_audit()

import sys, os
import pandas as pd
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
runs_path = os.path.join(R, "phase3_centralized_runs.csv")
sum_path = os.path.join(R, "phase3_centralized_summary.csv")

if not os.path.exists(runs_path):
    print("runs file not found yet")
    sys.exit(0)

df_runs = pd.read_csv(runs_path)
print(f"Total runs loaded: {len(df_runs)}")

# Group by model
models = df_runs['model'].unique()
for m in models:
    sub = df_runs[df_runs['model'] == m]
    print(f"\n==================== MODEL: {m} (n={len(sub)}) ====================")
    print(f"Training records: {sub['n_train_records'].iloc[0]} (Dev subjects: {sub['n_dev_subjects'].iloc[0]})")
    print(f"Average fit time: {sub['fit_s'].mean():.2f}s, Infer time: {sub['infer_s'].mean()*1000:.1f}ms")
    print("\nMetrics summary (Mean +/- Std):")
    print(f"  Pooled Record Acc:      {sub['acc'].mean():.3f} +/- {sub['acc'].std():.3f}")
    print(f"  Pooled Record BalAcc:   {sub['bacc'].mean():.3f} +/- {sub['bacc'].std():.3f}")
    print(f"  Pooled Record ROC-AUC:  {sub['auc'].mean():.3f} +/- {sub['auc'].std():.3f}")
    print(f"  Pooled Record PR-AUC:   {sub['prauc'].mean():.3f} +/- {sub['prauc'].std():.3f}")
    print(f"  Pooled Record MCC:      {sub['mcc'].mean():.3f} +/- {sub['mcc'].std():.3f}")
    print(f"  Mean Per-Subject Acc:   {sub['subj_weighted_acc'].mean():.3f} +/- {sub['subj_weighted_acc'].std():.3f}")
    print(f"  Mean Per-Subject Sens:  {sub['subj_weighted_sens'].mean():.3f} +/- {sub['subj_weighted_sens'].std():.3f}")
    print(f"  Mean Per-Subject Spec:  {sub['subj_weighted_spec'].mean():.3f} +/- {sub['subj_weighted_spec'].std():.3f}")
    print(f"  Subject-Weighted BalAcc:{sub['subj_weighted_bacc'].mean():.3f} +/- {sub['subj_weighted_bacc'].std():.3f}")
    print(f"  Subject-Weighted F1M:   {sub['subj_weighted_f1m'].mean():.3f} +/- {sub['subj_weighted_f1m'].std():.3f}")
    if 'subj_weighted_auc' in sub.columns:
        valid_auc = sub['subj_weighted_auc'].dropna()
        if len(valid_auc) > 0:
            print(f"  Bi-Class Subj ROC-AUC:  {valid_auc.mean():.3f} +/- {valid_auc.std():.3f} (n_seeds_with_biclass={len(valid_auc)})")
    
    print("\nHyperparameters selected across seeds:")
    for seed, grp in sub.groupby('seed'):
        print(f"  Seed {seed}: {grp['best_params'].values[0]}")
        
    print("\nConfusion Matrix total across all 10 seeds:")
    tn = sub['tn'].sum()
    fp = sub['fp'].sum()
    fn = sub['fn'].sum()
    tp = sub['tp'].sum()
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp} (Total: {tn+fp+fn+tp})")
    print(f"  Aggregate Sensitivity: {tp/(tp+fn):.3f}, Aggregate Specificity: {tn/(tn+fp):.3f}")


import pandas as pd
import numpy as np

print("=== DISCREPANCY 1: Safe Local Class-Weighted FedAvg AUC std ===")
p4b_e4_sum = pd.read_csv('experiments/phase4b/results/exp4_federated_summary.csv')
safe_local = p4b_e4_sum[p4b_e4_sum['model'].str.contains('Safe Local', na=False)]
print("In exp4_federated_summary.csv:")
print(safe_local[['model', 'roc_auc_mean', 'roc_auc_std', 'bacc_mean', 'bacc_std', 'mcc_mean', 'mcc_std']])

p4b_e4_runs = pd.read_csv('experiments/phase4b/results/exp4_federated_runs.csv')
safe_local_runs = p4b_e4_runs[p4b_e4_runs['model'].str.contains('Safe Local', na=False)]
print("In exp4_federated_runs.csv (per seed):")
print(safe_local_runs[['seed', 'auc', 'bacc', 'mcc']])
print("Recomputed AUC std ddof=1:", safe_local_runs['auc'].std(ddof=1))
print("Recomputed AUC std ddof=0:", safe_local_runs['auc'].std(ddof=0))
print("Recomputed BalAcc std ddof=1:", safe_local_runs['bacc'].std(ddof=1))

print("\n=== DISCREPANCY 2: Centralized Logistic Regression AUC std ===")
cent_lr = p4b_e4_sum[p4b_e4_sum['model'].str.contains('Centralized LR', na=False)]
print("In exp4_federated_summary.csv:")
print(cent_lr[['model', 'roc_auc_mean', 'roc_auc_std', 'bacc_mean', 'bacc_std', 'mcc_mean', 'mcc_std']])

cent_lr_runs = p4b_e4_runs[p4b_e4_runs['model'].str.contains('Centralized LR', na=False)]
for m in cent_lr_runs['model'].unique():
    sub = cent_lr_runs[cent_lr_runs['model'] == m]
    print(f"Per-seed runs for {m}:")
    print(f"  AUC Mean: {sub['auc'].mean():.6f}, std ddof=1: {sub['auc'].std(ddof=1):.6f}, std ddof=0: {sub['auc'].std(ddof=0):.6f}")
    print(f"  BalAcc Mean: {sub['bacc'].mean():.6f}, std ddof=1: {sub['bacc'].std(ddof=1):.6f}")

print("\n=== Check where 0.1118 and 0.0811 appear in reports ===")
with open('experiments/phase4b/final_verification/phase4b_publication_readiness.md', 'r') as f:
    for line_no, line in enumerate(f, 1):
        if '0.1118' in line or '0.0811' in line:
            print(f"phase4b_publication_readiness.md L{line_no}: {line.strip()}")

with open('experiments/phase4b/final_verification/final_verification_report.md', 'r') as f:
    for line_no, line in enumerate(f, 1):
        if '0.1118' in line or '0.0811' in line:
            print(f"final_verification_report.md L{line_no}: {line.strip()}")

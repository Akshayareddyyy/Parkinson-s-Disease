import pandas as pd
import numpy as np

print("=== DISCREPANCY 3: FedAvg 50% client participation AUC ===")
p4_sum = pd.read_csv('experiments/phase4/results/phase4_federated_summary.csv')
p4_50 = p4_sum[p4_sum['model'].str.contains('50%', na=False)]
print("In phase4_federated_summary.csv for 50% clients:")
print(p4_50[['model', 'topology', 'roc_auc_mean', 'roc_auc_std', 'bacc_mean', 'bacc_std', 'mcc_mean']])

p4_100 = p4_sum[p4_sum['model'].str.contains('100%', na=False)]
print("\nIn phase4_federated_summary.csv for 100% clients:")
print(p4_100[['model', 'topology', 'roc_auc_mean', 'roc_auc_std', 'bacc_mean', 'bacc_std', 'mcc_mean']])

p4_runs = pd.read_csv('experiments/phase4/results/phase4_federated_runs.csv')
print("\nUnique models in phase4_federated_runs.csv:")
print(p4_runs['model'].unique())
for m in p4_runs['model'].unique():
    if '50%' in m or '100%' in m or 'FedAvg' in m:
        sub = p4_runs[p4_runs['model'] == m]
        print(f"Model: {m:35s} | N={len(sub)} | AUC Mean: {sub['auc'].mean():.6f} +/- {sub['auc'].std():.6f}")

print("\nLet's check phase4_fl_convergence_history.csv:")
conv = pd.read_csv('experiments/phase4/results/phase4_fl_convergence_history.csv')
print("Columns in convergence history:", conv.columns.tolist())
print(conv[conv['model'].str.contains('50%|subsample|dropout', case=False, na=False)][['round', 'model', 'val_auc', 'test_auc']].tail(10) if 'model' in conv.columns else conv.head(5))

print("\nLet's check independent_verification in Phase 4:")
with open('experiments/phase4/independent_verification/independent_verification_report.md', 'r') as f:
    for line_no, line in enumerate(f, 1):
        if '0.6175' in line or '0.6287' in line:
            print(f"independent_verification_report.md L{line_no}: {line.strip()}")


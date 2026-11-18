import pandas as pd
import numpy as np

print("=== 1 & 2. PHASE 4B EXP 4 FEDERATED SUMMARY ===")
p4b_e4 = pd.read_csv('experiments/phase4b/results/exp4_federated_summary.csv')
for idx, r in p4b_e4.iterrows():
    print(f"Row {idx:2d} | Task: {r['task']:14s} | Topo: {str(r['topology']):15s} | Model: {str(r['model']):35s} | AUC: {r['roc_auc_mean']:.6f} +/- {r['roc_auc_std']:.6f}")

print("\n=== PHASE 4B EXP 4 FEDERATED RUNS (PER-SEED) FOR SAFE LOCAL AND CENTRALIZED LR ===")
p4b_e4_runs = pd.read_csv('experiments/phase4b/results/exp4_federated_runs.csv')
print("Unique models in Phase 4B Exp 4 runs:", p4b_e4_runs['model'].unique())
for model_name in p4b_e4_runs['model'].unique():
    sub = p4b_e4_runs[p4b_e4_runs['model'] == model_name]
    if 'roc_auc' in sub.columns and sub['roc_auc'].notna().any():
        print(f"Model: {model_name:35s} | N={len(sub)} | AUC Mean: {sub['roc_auc'].mean():.6f} | Std: {sub['roc_auc'].std():.6f} (ddof=1) | Std (ddof=0): {sub['roc_auc'].std(ddof=0):.6f}")

print("\n=== PHASE 4 FEDERATED SUMMARY ===")
p4 = pd.read_csv('experiments/phase4/results/phase4_federated_summary.csv')
for idx, r in p4.iterrows():
    print(f"Row {idx:2d} | Task: {r['task']:14s} | Topo: {str(r['topology']):25s} | Model: {str(r['model']):30s} | AUC: {r['roc_auc_mean']:.6f} +/- {r['roc_auc_std']:.6f}")

print("\n=== PHASE 4 FEDERATED RUNS (PER-SEED) ===")
p4_runs = pd.read_csv('experiments/phase4/results/phase4_federated_runs.csv')
for model_name in p4_runs['model'].unique():
    sub = p4_runs[p4_runs['model'] == model_name]
    if 'roc_auc' in sub.columns and sub['roc_auc'].notna().any():
        for top in sub['topology'].unique():
            sub_top = sub[sub['topology'] == top]
            print(f"Model: {model_name:25s} | Topo: {str(top):25s} | N={len(sub_top)} | AUC Mean: {sub_top['roc_auc'].mean():.6f} | Std: {sub_top['roc_auc'].std():.6f}")


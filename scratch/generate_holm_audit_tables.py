"""
Phase 3 Holm-Bonferroni Multiple Comparison Audit & Corrected Test Tables
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
import pandas as pd
import numpy as np

def holm_adjust(raw_p_list):
    p = np.asarray(raw_p_list, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    
    multipliers = np.arange(n, 0, -1)
    adjusted_sorted = np.minimum(1.0, sorted_p * multipliers)
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    
    inv_order = np.empty(n, dtype=int)
    inv_order[order] = np.arange(n)
    adj_p = adjusted_sorted[inv_order]
    
    for val in np.unique(p):
        mask = (p == val)
        adj_p[mask] = np.max(adj_p[mask])
        
    return adj_p

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

baselines = ['LR', 'Linear SVM', 'Random Forest', 'HistGradientBoosting', 'GradientBoosting']
p_auc = [0.43164062, 0.43164062, 0.00390625, 0.13085938, 0.06445312]
p_bacc = [0.32226562, 0.43164062, 0.03710938, 0.27539062, 0.27539062]

# Per-metric family (m=5)
adj_auc_5 = holm_adjust(p_auc)
adj_bacc_5 = holm_adjust(p_bacc)

# Combined omnibus family (m=10)
all_p = p_auc + p_bacc
adj_all_10 = holm_adjust(all_p)
adj_auc_10 = adj_all_10[:5]
adj_bacc_10 = adj_all_10[5:]

rows = []
for i, b in enumerate(baselines):
    rows.append({
        'comparison': f"Extra Trees vs {b}",
        'metric': 'ROC-AUC',
        'raw_p_unrounded': p_auc[i],
        'raw_p_str': f"{p_auc[i]:.4f}",
        'holm_adj_p_m5': round(float(adj_auc_5[i]), 4),
        'holm_adj_p_m10': round(float(adj_auc_10[i]), 4),
        'confirmatory': False
    })
    rows.append({
        'comparison': f"Extra Trees vs {b}",
        'metric': 'Balanced Accuracy',
        'raw_p_unrounded': p_bacc[i],
        'raw_p_str': f"{p_bacc[i]:.4f}",
        'holm_adj_p_m5': round(float(adj_bacc_5[i]), 4),
        'holm_adj_p_m10': round(float(adj_bacc_10[i]), 4),
        'confirmatory': False
    })

df_tests = pd.DataFrame(rows)
out_csv = os.path.join(R, "phase3_corrected_statistical_tests.csv")
df_tests.to_csv(out_csv, index=False)

audit_data = {
    'audit_title': "Phase 3 Holm-Bonferroni Multiple Comparison Audit",
    'status': "AUDITED AND CORRECTED",
    'family_definitions': {
        'family_1_per_metric': "m = 5 comparisons against Extra Trees per evaluation metric (ROC-AUC and Balanced Accuracy evaluated independently).",
        'family_2_combined_omnibus': "m = 10 comparisons against Extra Trees pooling both metrics simultaneously."
    },
    'tie_handling_and_monotonicity': "Monotonicity is strictly enforced via cumulative maximum (cummax). Identical raw p-values receive mathematically identical adjusted p-values.",
    'methodological_caveat': "The 10 partitions reuse the same 42 unique subjects, violating independence. Holm correction controls family-wise error but does not resolve structural partition dependence. All tests remain strictly exploratory.",
    'detailed_comparisons': rows
}

out_json = os.path.join(R, "phase3_holm_correction_audit.json")
with open(out_json, 'w') as f:
    json.dump(audit_data, f, indent=2)

print("Export completed successfully.")


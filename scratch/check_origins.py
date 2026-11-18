import pandas as pd
import numpy as np

# Let's inspect all columns of exp4_federated_summary.csv for row 5 (Safe Local) and row 0/1 (Centralized LR)
df = pd.read_csv('experiments/phase4b/results/exp4_federated_summary.csv')
print("=== Columns of exp4_federated_summary.csv ===")
for col in df.columns:
    print(f"{col:25s} | Safe Local: {df.loc[5, col]} | Cent LR (Std): {df.loc[1, col]}")

# Let's check CI calculations: 0.128895 / sqrt(10) * t_crit or 1.96
se_safe = 0.128895 / np.sqrt(10)
print(f"SE Safe Local: {se_safe:.6f}, 1.96*SE: {1.96*se_safe:.6f}, 2.262*SE: {2.262*se_safe:.6f}")

se_cent = 0.106601 / np.sqrt(10)
print(f"SE Cent LR: {se_cent:.6f}, 1.96*SE: {1.96*se_cent:.6f}, 2.262*SE: {2.262*se_cent:.6f}")

# Look at final_model_feature_parity_matrix.csv in phase 5
p5_matrix = pd.read_csv('experiments/phase5/final_model_feature_parity_matrix.csv')
print("\n=== final_model_feature_parity_matrix.csv in Phase 5 ===")
print(p5_matrix.to_string())

# Look at experiments/phase4b/final_verification/phase4b_publication_readiness.md


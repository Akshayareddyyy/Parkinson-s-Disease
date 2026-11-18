import pandas as pd
df = pd.read_csv('experiments/phase4b/results/exp4_federated_runs.csv')
safe = df[df['model'].str.contains('Safe Local', na=False)]
cent = df[df['model'].str.contains('Centralized LR (Standard)', na=False)]

print("Safe Local columns mean & std:")
for col in ['best_val_metric', 'auc', 'prauc', 'acc', 'bacc', 'sens', 'spec', 'f1m', 'mcc', 'brier', 'log_loss']:
    print(f"{col:15s} | Mean: {safe[col].mean():.6f} | Std: {safe[col].std():.6f}")

print("\nCentralized LR columns mean & std:")
for col in ['best_val_metric', 'auc', 'prauc', 'acc', 'bacc', 'sens', 'spec', 'f1m', 'mcc', 'brier', 'log_loss']:
    print(f"{col:15s} | Mean: {cent[col].mean():.6f} | Std: {cent[col].std():.6f}")


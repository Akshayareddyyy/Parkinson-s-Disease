import pandas as pd
import numpy as np
import hashlib

data_path = 'data/parkinsons_updrs.csv'
with open(data_path, 'rb') as f:
    sha = hashlib.sha256(f.read()).hexdigest()

df = pd.read_csv(data_path)
print("=" * 60)
print("DATASET INTEGRITY & DISCREPANCY AUDIT")
print("=" * 60)
print(f"File: {data_path}")
print(f"SHA256: {sha}")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
print(f"Unique Subjects: {df['subject#'].nunique()}")
print(f"Total Missing Values: {df.isna().sum().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")
print(f"motor_UPDRS min: {df['motor_UPDRS'].min():.6f}")
print(f"motor_UPDRS max: {df['motor_UPDRS'].max():.6f}")
print(f"total_UPDRS min: {df['total_UPDRS'].min():.6f}")
print(f"total_UPDRS max: {df['total_UPDRS'].max():.6f}")

VOICE = ['Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)',
         'Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']
FEATS = ['age', 'sex', 'test_time'] + VOICE
print(f"Eligible Predictors Count: {len(FEATS)}")

# Now investigate 39.511 vs 39.388
print("\nINVESTIGATING 39.511 vs 39.388:")
print(f"Top 5 highest motor_UPDRS values in dataset:")
top5 = df.nlargest(5, 'motor_UPDRS')[['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS']]
print(top5.to_string())

# Check total_UPDRS values as well
print(f"\nTop 5 highest total_UPDRS values in dataset:")
top5_tot = df.nlargest(5, 'total_UPDRS')[['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS']]
print(top5_tot.to_string())

# Search if 39.511 exists in any column
print("\nSearching if 39.511 exists in ANY column of the dataset:")
for col in df.columns:
    if df[col].dtype in [np.float64, np.float32]:
        close_matches = df[np.isclose(df[col], 39.511, atol=0.01)]
        if len(close_matches) > 0:
            print(f"  Found close match in column {col}: {close_matches[col].values}")


import pandas as pd
import glob

print("Checking where 0.6287 comes from in CSVs:")
for p in glob.glob('experiments/**/results/*.csv', recursive=True) + glob.glob('results/*.csv'):
    try:
        df = pd.read_csv(p)
        for col in df.select_dtypes(include=['float', 'int']).columns:
            matches = df[df[col].between(0.62865, 0.62875)]
            if len(matches) > 0:
                print(f"Match in {p} [{col}]:")
                print(matches.head(2))
    except Exception as e:
        pass


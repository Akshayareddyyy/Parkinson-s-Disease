import pandas as pd
import glob

for p in glob.glob('experiments/**/results/*.csv', recursive=True) + glob.glob('results/*.csv'):
    try:
        df = pd.read_csv(p)
        for col in df.select_dtypes(include=['float', 'int']).columns:
            for val in df[col].dropna():
                if abs(val - 0.1118) < 0.001:
                    print(f"0.1118 close match in {p} [{col}]: {val}")
                if abs(val - 0.0811) < 0.001:
                    print(f"0.0811 close match in {p} [{col}]: {val}")
    except:
        pass


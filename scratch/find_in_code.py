import glob

print("Searching for 0.1118 and 0.0811 in python scripts:")
for p in glob.glob('experiments/**/*.py', recursive=True) + glob.glob('code/*.py'):
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f, 1):
            if '0.1118' in line or '0.0811' in line:
                print(f"{p}:{i} -> {line.strip()}")


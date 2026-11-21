import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== SMOKE TEST: PARKINSON'S SEVERITY CODEBASE ===")

# 1. Test common.py and dataset loading
print("[1/5] Testing common.py and dataset loading...")
from common import BASE_DIR, DATA, FEATS, load, partition, make_sets, metrics
df = load()
assert len(df) == 5875, f"Expected 5875 rows, got {len(df)}"
assert df['subject#'].nunique() == 42, f"Expected 42 subjects, got {df['subject#'].nunique()}"
print(f"  Dataset successfully loaded from: {DATA}")
print(f"  Shape: {df.shape}, Unique subjects: {df['subject#'].nunique()}")

# 2. Test subject partitioning & make_sets
print("[2/5] Testing subject partitioning (seed 0)...")
dev_subs, val_subs, test_subs = partition(df, 0)
assert len(dev_subs) == 26 and len(val_subs) == 7 and len(test_subs) == 9
assert len(set(dev_subs) & set(val_subs)) == 0
assert len(set(dev_subs) & set(test_subs)) == 0
assert len(set(val_subs) & set(test_subs)) == 0
D, V, T, thr = make_sets(df, 0)
print(f"  Threshold tau (dev median): {thr:.2f}")
print(f"  Dev records: {len(D)}, Val records: {len(V)}, Test records: {len(T)}")
assert len(D) + len(V) + len(T) == 5875

# 3. Test classical ML & FL logic (1 quick fold / step)
print("[3/5] Testing classical ML & FL logic...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))
m.fit(D[FEATS].values, D.y.values)
preds = m.predict_proba(T[FEATS].values)[:, 1]
mets = metrics(T.y.values, (preds > 0.5).astype(int), preds)
print(f"  Quick LR test ROC-AUC: {mets['auc']:.3f}, BalAcc: {mets['bacc']:.3f}")

# 4. Test PennyLane QML circuits (single forward pass + step)
print("[4/5] Testing PennyLane QML circuit forward pass...")
import pennylane as qml
from pennylane import numpy as pnp
dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev, interface="autograd", diff_method="backprop")
def test_vqc(x, th):
    for i in range(4): qml.Hadamard(i); qml.RZ(x[:, i], i)
    for l in range(1):
        for i in range(4): qml.RY(th[l, i, 0], i); qml.RZ(th[l, i, 1], i)
    return qml.expval(qml.PauliZ(0))

x_dummy = pnp.random.uniform(0, pnp.pi, (2, 4))
th_dummy = pnp.random.uniform(0, 2*pnp.pi, (1, 4, 2), requires_grad=True)
res = test_vqc(x_dummy, th_dummy)
print(f"  PennyLane version: {qml.__version__}")
print(f"  Test circuit evaluated successfully, expval: {res}")

# 5. Test paper generation (DOCX & TeX)
print("[5/5] Testing paper generation (build_paper.py)...")
import build_paper
print(f"  Paper build completed! Generated outputs in {build_paper.OUT}")

print("=== ALL SMOKE TESTS PASSED SUCCESSFULLY! ===")


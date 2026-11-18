import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
import common
import pandas as pd
import numpy as np

df = common.load()
print(f"Total records in df: {len(df)}, Unique subjects: {df['subject#'].nunique()}")

splits_summary = []
for seed in range(10):
    dv, va, te = common.partition(df, seed)
    s_dv, s_va, s_te = set(dv), set(va), set(te)
    
    # Check subjects
    assert len(s_dv) == 26, f"Seed {seed}: Dev subjects count {len(s_dv)} != 26"
    assert len(s_va) == 7, f"Seed {seed}: Val subjects count {len(s_va)} != 7"
    assert len(s_te) == 9, f"Seed {seed}: Test subjects count {len(s_te)} != 9"
    assert len(s_dv & s_va) == 0, "Overlap Dev-Val"
    assert len(s_dv & s_te) == 0, "Overlap Dev-Test"
    assert len(s_va & s_te) == 0, "Overlap Val-Test"
    assert len(s_dv | s_va | s_te) == 42, "Not all 42 subjects accounted for"
    
    # Make sets
    D, V, T, thr = common.make_sets(df, seed)
    total_records = len(D) + len(V) + len(T)
    assert total_records == 5875, f"Seed {seed}: Total records {total_records} != 5875"
    
    # Check target balancing
    dev_median = float(D['motor_UPDRS'].median())
    assert abs(dev_median - thr) < 1e-6, f"Threshold mismatch: {dev_median} vs {thr}"
    
    dev_pos = D.y.mean()
    val_pos = V.y.mean()
    test_pos = T.y.mean()
    
    splits_summary.append({
        'seed': seed,
        'dev_records': len(D),
        'val_records': len(V),
        'test_records': len(T),
        'total_records': total_records,
        'dev_subjects': len(s_dv),
        'val_subjects': len(s_va),
        'test_subjects': len(s_te),
        'dev_median_thr': round(thr, 4),
        'dev_pos_ratio': round(dev_pos, 4),
        'val_pos_ratio': round(val_pos, 4),
        'test_pos_ratio': round(test_pos, 4)
    })

df_res = pd.DataFrame(splits_summary)
print(df_res.to_string(index=False))


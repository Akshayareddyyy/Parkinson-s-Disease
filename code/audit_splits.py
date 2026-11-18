import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import BASE_DIR, DATA, FEATS, load, partition, make_sets

def audit_partitions():
    print("=== PHASE 2: SUBJECT-WISE PARTITION & LEAKAGE AUDIT ===")
    df = load()
    NSEEDS = 10
    
    partition_audit = []
    
    for seed in range(NSEEDS):
        dv, va, te = partition(df, seed)
        
        # 1. Subject count verification
        assert len(dv) == 26, f"Seed {seed}: Expected 26 dev subjects, got {len(dv)}"
        assert len(va) == 7, f"Seed {seed}: Expected 7 val subjects, got {len(va)}"
        assert len(te) == 9, f"Seed {seed}: Expected 9 test subjects, got {len(te)}"
        assert len(set(dv) | set(va) | set(te)) == 42, f"Seed {seed}: Not all 42 subjects accounted for!"
        
        # 2. Subject overlap verification (strictly zero)
        assert len(set(dv) & set(va)) == 0, f"Seed {seed}: Overlap between Dev and Val!"
        assert len(set(dv) & set(te)) == 0, f"Seed {seed}: Overlap between Dev and Test!"
        assert len(set(va) & set(te)) == 0, f"Seed {seed}: Overlap between Val and Test!"
        
        # 3. Record-level isolation verification
        D, V, T, thr = make_sets(df, seed)
        assert len(D) + len(V) + len(T) == 5875, f"Seed {seed}: Dropped records! Sum is {len(D)+len(V)+len(T)}"
        
        d_subs = set(D['subject#'].unique())
        v_subs = set(V['subject#'].unique())
        t_subs = set(T['subject#'].unique())
        
        assert d_subs == set(dv), f"Seed {seed}: Mismatch in dev subject IDs"
        assert v_subs == set(va), f"Seed {seed}: Mismatch in val subject IDs"
        assert t_subs == set(te), f"Seed {seed}: Mismatch in test subject IDs"
        
        # Check target construction
        assert np.isclose(D.y.mean(), 0.50, atol=0.01), f"Seed {seed}: Dev target not balanced around median!"
        assert thr == D['motor_UPDRS'].median(), f"Seed {seed}: Threshold tau does not match dev median!"
        
        # Verify that validation and test labels use the dev threshold
        assert (V.y == (V['motor_UPDRS'] >= thr).astype(int)).all(), f"Seed {seed}: Val target leakage/inconsistency!"
        assert (T.y == (T['motor_UPDRS'] >= thr).astype(int)).all(), f"Seed {seed}: Test target leakage/inconsistency!"
        
        record = {
            'seed': seed,
            'dev_subjects_count': len(dv),
            'val_subjects_count': len(va),
            'test_subjects_count': len(te),
            'dev_records': len(D),
            'val_records': len(V),
            'test_records': len(T),
            'threshold_tau': float(thr),
            'dev_pos_rate': float(D.y.mean()),
            'val_pos_rate': float(V.y.mean()),
            'test_pos_rate': float(T.y.mean()),
            'dev_subjects': [int(x) for x in dv],
            'val_subjects': [int(x) for x in va],
            'test_subjects': [int(x) for x in te],
            'zero_overlap_verified': True
        }
        partition_audit.append(record)
        
    df_audit = pd.DataFrame(partition_audit)
    
    out_json = os.path.join(BASE_DIR, "results", "split_verification_report.json")
    out_csv = os.path.join(BASE_DIR, "results", "split_verification_summary.csv")
    
    with open(out_json, 'w') as f:
        json.dump(partition_audit, f, indent=2)
        
    cols_summary = ['seed', 'dev_subjects_count', 'val_subjects_count', 'test_subjects_count',
                    'dev_records', 'val_records', 'test_records', 'threshold_tau',
                    'dev_pos_rate', 'val_pos_rate', 'test_pos_rate', 'zero_overlap_verified']
    df_audit[cols_summary].to_csv(out_csv, index=False)
    
    print(f"Split audit completed successfully for all {NSEEDS} partitions!")
    print(f"  All assertions passed: 0 subject overlap, 0 record leakage, threshold fitted solely on D.")
    print(f"  Dev records span: [{df_audit.dev_records.min()}, {df_audit.dev_records.max()}]")
    print(f"  Val records span: [{df_audit.val_records.min()}, {df_audit.val_records.max()}]")
    print(f"  Test records span: [{df_audit.test_records.min()}, {df_audit.test_records.max()}]")
    print(f"  Summary saved to: {out_csv}")
    return partition_audit

if __name__ == '__main__':
    audit_partitions()


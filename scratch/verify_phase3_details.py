import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
import common
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

df = common.load()

print("=== CHECKING TEST SUBJECT CLASS DISTRIBUTIONS ===")
for seed in range(10):
    D, V, T, thr = common.make_sets(df, seed)
    mono_cnt = 0
    bi_cnt = 0
    mono_pos = 0
    mono_neg = 0
    for sid, grp in T.groupby('subject#'):
        classes = grp.y.nunique()
        if classes == 1:
            mono_cnt += 1
            if grp.y.iloc[0] == 1:
                mono_pos += 1
            else:
                mono_neg += 1
        else:
            bi_cnt += 1
    print(f"Seed {seed}: 9 Test Subjects -> {bi_cnt} bi-class, {mono_cnt} mono-class (Pos: {mono_pos}, Neg: {mono_neg})")

print("\n=== CHECKING GROUPKFOLD ON DEVELOPMENT SETS ===")
for seed in range(10):
    D, V, T, thr = common.make_sets(df, seed)
    cv = GroupKFold(n_splits=5)
    XD = D[common.FEATS].values
    yD = D.y.values
    groups = D['subject#'].values
    
    fold_train_counts = []
    fold_val_counts = []
    fold_train_subjects = []
    fold_val_subjects = []
    
    for tr_idx, val_idx in cv.split(XD, yD, groups=groups):
        tr_s = set(groups[tr_idx])
        val_s = set(groups[val_idx])
        assert len(tr_s & val_s) == 0, "GroupKFold subject leakage across folds!"
        fold_train_counts.append(len(tr_idx))
        fold_val_counts.append(len(val_idx))
        fold_train_subjects.append(len(tr_s))
        fold_val_subjects.append(len(val_s))
        
    print(f"Seed {seed}: Dev records={len(D)}, Dev subjects={len(set(groups))}")
    print(f"  Folds train recs: {fold_train_counts}")
    print(f"  Folds val recs:   {fold_val_counts}")
    print(f"  Folds train subs: {fold_train_subjects}")
    print(f"  Folds val subs:   {fold_val_subjects}")


import os, sys, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from common import BASE_DIR, DATA, FEATS, VOICE, load, partition, make_sets

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def audit_dataset():
    print("=== PHASE 1: COMPREHENSIVE DATASET & METHODOLOGY AUDIT ===")
    assert os.path.exists(DATA), f"Dataset not found at: {DATA}"
    sha256 = compute_sha256(DATA)
    df = load()
    
    total_records = len(df)
    total_subjects = df['subject#'].nunique()
    cols = list(df.columns)
    
    # Missing, duplicates, nulls
    missing_per_col = df.isnull().sum().to_dict()
    total_missing = sum(missing_per_col.values())
    dup_rows = int(df.duplicated().sum())
    
    # Records per subject
    recs_per_sub = df.groupby('subject#').size()
    
    # UPDRS statistics
    motor_stats = {
        'min': float(df['motor_UPDRS'].min()),
        'q25': float(df['motor_UPDRS'].quantile(0.25)),
        'median': float(df['motor_UPDRS'].median()),
        'mean': float(df['motor_UPDRS'].mean()),
        'q75': float(df['motor_UPDRS'].quantile(0.75)),
        'max': float(df['motor_UPDRS'].max()),
        'std': float(df['motor_UPDRS'].std()),
        'skew': float(stats.skew(df['motor_UPDRS'])),
        'kurt': float(stats.kurtosis(df['motor_UPDRS']))
    }
    
    total_updrs_stats = {
        'min': float(df['total_UPDRS'].min()),
        'q25': float(df['total_UPDRS'].quantile(0.25)),
        'median': float(df['total_UPDRS'].median()),
        'mean': float(df['total_UPDRS'].mean()),
        'q75': float(df['total_UPDRS'].quantile(0.75)),
        'max': float(df['total_UPDRS'].max()),
        'std': float(df['total_UPDRS'].std()),
        'skew': float(stats.skew(df['total_UPDRS'])),
        'kurt': float(stats.kurtosis(df['total_UPDRS']))
    }
    
    corr_updrs, p_corr = stats.pearsonr(df['motor_UPDRS'], df['total_UPDRS'])
    
    # Feature exclusion verification
    assert 'motor_UPDRS' not in FEATS, "CRITICAL ERROR: motor_UPDRS leaked into FEATS!"
    assert 'total_UPDRS' not in FEATS, "CRITICAL ERROR: total_UPDRS leaked into FEATS!"
    assert 'subject#' not in FEATS, "CRITICAL ERROR: subject# leaked into FEATS!"
    assert len(FEATS) == 19, f"CRITICAL ERROR: Expected 19 features, got {len(FEATS)}"
    
    # Target distribution across subjects
    sub_means = df.groupby('subject#')['motor_UPDRS'].mean()
    overall_median = df['motor_UPDRS'].median()
    sub_target_rates = df.groupby('subject#').apply(lambda g: (g['motor_UPDRS'] >= overall_median).mean()).to_dict()
    
    audit_data = {
        'filepath': DATA,
        'sha256': sha256,
        'total_records': total_records,
        'total_subjects': total_subjects,
        'columns': cols,
        'total_columns': len(cols),
        'missing_values_total': total_missing,
        'missing_per_col': missing_per_col,
        'duplicate_rows': dup_rows,
        'records_per_subject': {
            'min': int(recs_per_sub.min()),
            'max': int(recs_per_sub.max()),
            'mean': float(recs_per_sub.mean()),
            'median': float(recs_per_sub.median()),
            'std': float(recs_per_sub.std())
        },
        'motor_UPDRS_distribution': motor_stats,
        'total_UPDRS_distribution': total_updrs_stats,
        'pearson_correlation_motor_total_UPDRS': float(corr_updrs),
        'permitted_predictors': FEATS,
        'num_predictors': len(FEATS),
        'excluded_columns': ['subject#', 'motor_UPDRS', 'total_UPDRS'],
        'subject_level_target_distribution_sample': {int(k): round(float(v), 3) for k, v in list(sub_target_rates.items())[:10]}
    }
    
    out_path = os.path.join(BASE_DIR, "results", "dataset_audit_report.json")
    with open(out_path, 'w') as f:
        json.dump(audit_data, f, indent=2)
        
    print(f"Audit completed successfully!")
    print(f"  Records: {total_records}, Subjects: {total_subjects}, Features: {len(FEATS)}")
    print(f"  SHA-256: {sha256}")
    print(f"  Missing values: {total_missing}, Duplicates: {dup_rows}")
    print(f"  motor_UPDRS: range [{motor_stats['min']:.2f}, {motor_stats['max']:.2f}], median {motor_stats['median']:.2f}")
    print(f"  total_UPDRS: range [{total_updrs_stats['min']:.2f}, {total_updrs_stats['max']:.2f}], correlation r = {corr_updrs:.4f}")
    print(f"  Audit report written to: {out_path}")
    return audit_data

if __name__ == '__main__':
    audit_dataset()


"""
Phase 5: Common Utilities, Configuration & Path Management
Provides:
  - Canonical data loading & 10 audited subject-wise partitions
  - 9-feature manifold definition
  - Environment variable setup for Windows stability
  - Checkpointing & path utilities
"""

import os, sys, time, gc, psutil, importlib.util
import numpy as np
import pandas as pd
import torch

# Base directory navigation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables for Windows portability & memory stability
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.tmp_mpl'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

# Explicitly load canonical common.py from ROOT/code/common.py to avoid name shadowing
canonical_common_path = os.path.join(BASE_DIR, 'code', 'common.py')
spec = importlib.util.spec_from_file_location("canonical_common", canonical_common_path)
canonical_common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_common)

canonical_load = canonical_common.load
canonical_partition = canonical_common.partition
canonical_make_sets = canonical_common.make_sets

# Verified 9-feature manifold
FEATS_9 = ['age', 'sex', 'test_time', 'Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
CONT_FEATS_8 = ['age', 'test_time', 'Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']

# Full 19 features for historical contextual comparison
FEATS_19 = [
    'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)',
    'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]

# Phase 5 directory structure
PHASE5_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PHASE5_DIR, 'results')
FIGURES_DIR = os.path.join(PHASE5_DIR, 'figures')
LOGS_DIR = os.path.join(PHASE5_DIR, 'logs')
AUDIT_DIR = os.path.join(PHASE5_DIR, 'audit')
CHECKPOINTS_DIR = os.path.join(PHASE5_DIR, 'checkpoints')

for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, AUDIT_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

def load_data():
    return canonical_load()

def get_partition_sets(df, seed):
    """
    Returns D (Dev: 26 subj), V (Val: 7 subj), T (Test: 9 subj), thr (Dev median).
    All records preserved; no downsampling.
    """
    return canonical_make_sets(df, seed)

def get_process_memory_mb():
    """Returns current host process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def release_memory():
    """Forces garbage collection and releases PyTorch allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

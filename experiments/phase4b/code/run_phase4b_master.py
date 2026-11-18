"""
Phase 4B: Master Orchestration Script
Runs all 5 Phase 4B experiments across 10 audited subject-wise partitions:
  - Experiment 1: Advanced Centralized Classification
  - Experiment 2: Advanced Continuous Regression
  - Experiment 3: Acoustic Feature Engineering & Transformations
  - Experiment 4: Federated Learning Improvement & Stabilization
  - Experiment 5: Personalized Longitudinal Delta Prediction
Followed by generation of all publication figures.
"""

import os, sys, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'code'))

from run_exp1_advanced_classification import run_experiment_1
from run_exp2_advanced_regression import run_experiment_2
from run_exp3_feature_engineering import run_experiment_3
from run_exp4_federated_improvement import run_experiment_4
from run_exp5_personalized_prediction import run_experiment_5
from plot_phase4b_figures import plot_all

def run_master(n_seeds=10):
    t_start = time.time()
    print("=" * 80)
    print("PHASE 4B: COMPLETE PREDICTIVE PERFORMANCE IMPROVEMENT CAMPAIGN")
    print(f"Total Partitions: {n_seeds} Audited Seeds (26 Dev / 7 Val / 9 Locked-Test)")
    print("Strict Fairness: Zero Test Leakage, 100% Record Accounting, Matched Controls")
    print("=" * 80)
    
    stages = [
        ("Exp 1: Advanced Classification", run_experiment_1),
        ("Exp 2: Advanced Regression", run_experiment_2),
        ("Exp 3: Feature Engineering", run_experiment_3),
        ("Exp 4: Federated Learning", run_experiment_4),
        ("Exp 5: Personalized Prediction", run_experiment_5)
    ]
    
    timing_records = []
    
    for name, fn in stages:
        t0 = time.time()
        print("\n" + "#" * 80)
        print(f"LAUNCHING {name.upper()}")
        print("#" * 80 + "\n")
        try:
            fn(n_seeds)
            elapsed = time.time() - t0
            timing_records.append((name, "SUCCESS", elapsed))
            print(f"\n[DONE] {name} completed in {elapsed/60.0:.2f} min ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            timing_records.append((name, f"FAILED: {e}", elapsed))
            print(f"\n[ERROR] {name} failed: {e}")
            raise e
            
    # Generate publication figures
    print("\n" + "#" * 80)
    print("GENERATING PUBLICATION FIGURES")
    print("#" * 80 + "\n")
    plot_all()
    
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("PHASE 4B CAMPAIGN EXECUTION COMPLETE")
    print(f"Total Duration: {total_elapsed/60.0:.2f} min ({total_elapsed:.1f}s)")
    print("=" * 80)
    print(f"{'Experiment':<35} | {'Status':<15} | {'Time (s)':<10}")
    print("-" * 65)
    for name, stat, el in timing_records:
        print(f"{name:<35} | {stat:<15} | {el:<10.1f}")
    print("=" * 80)

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_master(ns)


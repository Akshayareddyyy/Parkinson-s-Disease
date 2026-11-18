"""
Phase 3B: Master Orchestration Script
Sequentially executes:
  1. Experiment 1: Subject-Balanced Learning (10 seeds)
  2. Experiment 2: Improved Classical Algorithms (10 seeds)
  3. Experiment 3: Feature Engineering & Preprocessing (10 seeds)
  4. Experiment 4: Nested GroupKFold Cross-Validation (5 seeds)
  5. Experiment 5: Subject-Aware Normalization & Personalization (10 seeds)
  6. Visualization generation
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_exp1_subject_weighting import run_experiment_1
from run_exp2_improved_algorithms import run_experiment_2
from run_exp3_feature_engineering import run_experiment_3
from run_exp4_nested_cv import run_experiment_4
from run_exp5_personalized_calibration import run_experiment_5
from plot_phase3b_figures import plot_all_figures

def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("LAUNCHING PHASE 3B: SCIENTIFICALLY VALID MODEL IMPROVEMENT STUDY", flush=True)
    print("=" * 80, flush=True)
    
    # -----------------------------------------------------------------
    # STEP 1: EXPERIMENT 1 - Subject Weighting
    # -----------------------------------------------------------------
    print("\n>>> STARTING EXPERIMENT 1 (Subject Weighting) <<<", flush=True)
    t0 = time.time()
    run_experiment_1(n_seeds=10)
    print(f"Experiment 1 completed in {time.time()-t0:.1f}s.", flush=True)
    
    # -----------------------------------------------------------------
    # STEP 2: EXPERIMENT 2 - Improved Classical Algorithms
    # -----------------------------------------------------------------
    print("\n>>> STARTING EXPERIMENT 2 (Improved Algorithms: CatBoost, XGBoost, etc.) <<<", flush=True)
    t0 = time.time()
    run_experiment_2(n_seeds=10)
    print(f"Experiment 2 completed in {time.time()-t0:.1f}s.", flush=True)
    
    # -----------------------------------------------------------------
    # STEP 3: EXPERIMENT 3 - Feature Engineering & Transformations
    # -----------------------------------------------------------------
    print("\n>>> STARTING EXPERIMENT 3 (Feature Engineering & Preprocessing) <<<", flush=True)
    t0 = time.time()
    run_experiment_3(n_seeds=10)
    print(f"Experiment 3 completed in {time.time()-t0:.1f}s.", flush=True)
    
    # -----------------------------------------------------------------
    # STEP 4: EXPERIMENT 4 - Nested GroupKFold
    # -----------------------------------------------------------------
    print("\n>>> STARTING EXPERIMENT 4 (Nested GroupKFold Cross-Validation) <<<", flush=True)
    t0 = time.time()
    run_experiment_4(n_seeds=5)
    print(f"Experiment 4 completed in {time.time()-t0:.1f}s.", flush=True)
    
    # -----------------------------------------------------------------
    # STEP 5: EXPERIMENT 5 - Subject Calibration & Personalization
    # -----------------------------------------------------------------
    print("\n>>> STARTING EXPERIMENT 5 (Personalized Subject Calibration) <<<", flush=True)
    t0 = time.time()
    run_experiment_5(n_seeds=10)
    print(f"Experiment 5 completed in {time.time()-t0:.1f}s.", flush=True)
    
    # -----------------------------------------------------------------
    # STEP 6: VISUALIZATIONS & PLOTS
    # -----------------------------------------------------------------
    print("\n>>> GENERATING PUBLICATION FIGURES <<<", flush=True)
    plot_all_figures()
    
    print("\n" + "=" * 80, flush=True)
    print(f"ALL PHASE 3B EXPERIMENTS COMPLETED IN {(time.time()-t_start)/60:.2f} MINUTES!", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()

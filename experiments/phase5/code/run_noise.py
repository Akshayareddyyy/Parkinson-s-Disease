"""
Phase 5 - Experiment 5: Simulated Quantum Noise Resilience
Evaluates frozen VQC and PQC-R models across all 10 seeds under:
  - Depolarizing noise (p in [1e-4, 1e-3, 5e-3, 1e-2])
  - Phase damping (gamma in [1e-3, 1e-2, 5e-2])
  - Finite measurement shots (S in [1000, 4000, 10000])
  - Ideal statevector simulation (reference)
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch

from common import (
    load_data, get_partition_sets, release_memory, get_process_memory_mb,
    RESULTS_DIR, CHECKPOINTS_DIR
)
from preprocessing import FoldContainedQuantumPreprocessor
from quantum_classifier import VQCModel
from quantum_regressor import PQCRModel
from noise_simulation import evaluate_vqc_under_noise, evaluate_pqcr_under_noise
from metrics import compute_ci

def run_noise_experiments():
    print("=" * 80)
    print("PHASE 5 — EXPERIMENT 5: SIMULATED QUANTUM NOISE RESILIENCE EVALUATION")
    print("=" * 80)
    
    df = load_data()
    seeds = list(range(10))
    
    noise_runs = []
    t_start = time.perf_counter()
    
    depol_levels = [1e-4, 1e-3, 5e-3, 1e-2]
    dephase_levels = [1e-3, 1e-2, 5e-2]
    shot_levels = [1000, 4000, 10000]
    
    for seed in seeds:
        print(f"\n>>> SEED {seed}/9: Evaluating Noise Sensitivity...")
        t_seed_start = time.perf_counter()
        
        # Load test partition
        df_dev, df_val, df_test, _ = get_partition_sets(df, seed)
        prep = FoldContainedQuantumPreprocessor()
        prep.fit(df_dev)
        
        X_te_q = prep.transform_quantum_angles(df_test)
        y_te_cls = df_test['y'].values.astype(int)
        y_te_reg = df_test['motor_UPDRS'].values.astype(float)
        
        # 1. Load frozen VQC checkpoint
        vqc_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, f"vqc_seed_{seed}.pt"), weights_only=False)
        vqc_model = VQCModel(n_qubits=9, depth_L=vqc_ckpt['depth_L'], seed=seed)
        vqc_model.load_state_dict(vqc_ckpt['state_dict'])
        vqc_model.eval()
        
        # 2. Load frozen PQC-R checkpoint
        pqcr_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, f"pqcr_seed_{seed}.pt"), weights_only=False)
        pqcr_model = PQCRModel(n_qubits=9, depth_L=pqcr_ckpt['depth_L'], seed=seed)
        pqcr_model.load_state_dict(pqcr_ckpt['state_dict'])
        pqcr_model.eval()
        
        # --- A. Ideal Statevector Reference ---
        with torch.no_grad():
            _, te_probs, _ = vqc_model(torch.tensor(X_te_q, dtype=torch.float64))
            vqc_ideal_auc = vqc_ckpt['test_metrics']['roc_auc']
            
            te_preds, _ = pqcr_model(torch.tensor(X_te_q, dtype=torch.float64))
            pqcr_ideal_mae = pqcr_ckpt['test_metrics']['mae']
            
        noise_runs.append({
            'seed': seed, 'model': '9-Qubit VQC', 'condition': 'Ideal Statevector',
            'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
            'primary_metric_name': 'ROC-AUC', 'primary_metric_val': vqc_ideal_auc
        })
        noise_runs.append({
            'seed': seed, 'model': '9-Qubit PQC-R', 'condition': 'Ideal Statevector',
            'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
            'primary_metric_name': 'MAE', 'primary_metric_val': pqcr_ideal_mae
        })
        print(f"    Ideal Baseline: VQC ROC-AUC = {vqc_ideal_auc:.4f} | PQC-R MAE = {pqcr_ideal_mae:.4f}")
        
        # --- B. Depolarizing Noise ---
        for p in depol_levels:
            m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='depol', noise_param=p)
            m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='depol', noise_param=p)
            
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit VQC', 'condition': f'Depolarizing (p={p})',
                'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc']
            })
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit PQC-R', 'condition': f'Depolarizing (p={p})',
                'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae']
            })
            print(f"      Depolarizing (p={p}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f}")
            
        # --- C. Phase Damping / Dephasing Noise ---
        for gamma in dephase_levels:
            m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='dephase', noise_param=gamma)
            m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='dephase', noise_param=gamma)
            
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit VQC', 'condition': f'Phase Damping (gamma={gamma})',
                'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc']
            })
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit PQC-R', 'condition': f'Phase Damping (gamma={gamma})',
                'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae']
            })
            print(f"      Phase Damping (gamma={gamma}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f}")
            
        # --- D. Finite Measurement Shots ---
        for s in shot_levels:
            m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='shots', shots=s)
            m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='shots', shots=s)
            
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit VQC', 'condition': f'Finite Shots (S={s})',
                'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc']
            })
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit PQC-R', 'condition': f'Finite Shots (S={s})',
                'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae']
            })
            print(f"      Finite Shots (S={s}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f}")
            
        print(f"    Seed {seed} noise evaluation completed in {time.perf_counter() - t_seed_start:.2f}s")
        release_memory()
        
    df_nruns = pd.DataFrame(noise_runs)
    nruns_path = os.path.join(RESULTS_DIR, 'exp5_noise_runs.csv')
    df_nruns.to_csv(nruns_path, index=False)
    print(f"\n[Saved] Noise simulation per-seed runs saved to: {nruns_path}")
    
    # Aggregate Summary
    summary_list = []
    for (model, cond), grp in df_nruns.groupby(['model', 'condition'], sort=False):
        mean_val, std_val, ci_val = compute_ci(grp['primary_metric_val'])
        summary_list.append({
            'model': model,
            'condition': cond,
            'noise_type': grp['noise_type'].iloc[0],
            'noise_param': grp['noise_param'].iloc[0],
            'shots': grp['shots'].iloc[0],
            'metric_name': grp['primary_metric_name'].iloc[0],
            'metric_mean': mean_val,
            'metric_std': std_val,
            'metric_ci_lower': ci_val[0],
            'metric_ci_upper': ci_val[1],
            'n_seeds': len(grp)
        })
        
    df_nsummary = pd.DataFrame(summary_list)
    nsummary_path = os.path.join(RESULTS_DIR, 'exp5_noise_summary.csv')
    df_nsummary.to_csv(nsummary_path, index=False)
    print(f"[Saved] Noise simulation aggregate summary saved to: {nsummary_path}")
    
    print("\n" + "=" * 80)
    print("SIMULATED NOISE SENSITIVITY SUMMARY:")
    print("=" * 80)
    print(df_nsummary[['model', 'condition', 'metric_name', 'metric_mean', 'metric_std']].to_string(index=False))
    print(f"\nTotal Noise Evaluation Runtime: {time.perf_counter() - t_start:.2f}s")
    return df_nsummary, df_nruns

if __name__ == '__main__':
    run_noise_experiments()


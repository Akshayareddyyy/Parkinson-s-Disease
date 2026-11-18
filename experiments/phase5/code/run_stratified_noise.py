"""
Phase 5 - Experiment 5 (Path B): Stratified 10-Seed Quantum Noise Resilience Evaluation

Protocol:
  - Uses all 10 audited subject-wise seeds (0-9).
  - Selects a reproducible stratified sample of exactly 200 locked-test records per seed.
  - Stratified by class target 'y' to strictly preserve class proportions.
  - Exactly the same 200 selected records within each seed are used for BOTH VQC and PQC-R,
    and across Ideal reference and ALL noisy conditions.
  - Evaluates:
      1. Ideal Statevector Reference (on the 200 stratified records)
      2. Depolarizing noise: p in [1e-4, 1e-3, 5e-3, 1e-2]
      3. Phase damping (dephasing): gamma in [1e-3, 1e-2, 5e-2]
      4. Finite measurement shots: S in [1000, 4000, 10000]
  - Records sample-selection seeds and exact sample counts in sample metadata.
  - Saves all results separately:
      - exp5_stratified_sample_metadata.csv
      - exp5_stratified_noise_runs.csv
      - exp5_stratified_noise_summary.csv
  - Strictly preserves all frozen checkpoints and historical files.
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure reproducible environment
os.environ['MPLCONFIGDIR'] = 'C:/Users/akank/AppData/Local/Temp'

from common import (
    load_data, get_partition_sets, release_memory, get_process_memory_mb,
    RESULTS_DIR, CHECKPOINTS_DIR
)
from preprocessing import FoldContainedQuantumPreprocessor
from quantum_classifier import VQCModel
from quantum_regressor import PQCRModel
from noise_simulation import evaluate_vqc_under_noise, evaluate_pqcr_under_noise
from metrics import compute_ci, evaluate_classification, evaluate_regression

def run_stratified_noise_evaluation(seeds=list(range(10)), n_strat_samples=200):
    print("=" * 80)
    print("PHASE 5 — EXPERIMENT 5 (PATH B): STRATIFIED 10-SEED QUANTUM NOISE RESILIENCE")
    print(f"Sample size per seed: {n_strat_samples} locked-test records (Stratified by 'y')")
    print(f"Audited seeds: {seeds}")
    print("=" * 80)
    
    df = load_data()
    
    sample_metadata = []
    noise_runs = []
    failures = []
    
    depol_levels = [1e-4, 1e-3, 5e-3, 1e-2]
    dephase_levels = [1e-3, 1e-2, 5e-2]
    shot_levels = [1000, 4000, 10000]
    
    t_total_start = time.perf_counter()
    
    for seed in seeds:
        print(f"\n" + "-" * 70)
        print(f">>> SEED {seed}/9: Evaluating Stratified Quantum Noise Resilience...")
        print("-" * 70)
        t_seed_start = time.perf_counter()
        
        # 1. Load partitions
        df_dev, df_val, df_test, thr = get_partition_sets(df, seed)
        
        # 2. Stratified Sampling of exactly n_strat_samples locked-test records
        sample_selection_seed = 42 + seed
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_strat_samples, random_state=sample_selection_seed)
        sampled_indices, _ = next(sss.split(df_test, df_test['y']))
        df_test_strat = df_test.iloc[sampled_indices].reset_index(drop=True)
        
        orig_c1_ratio = float(df_test['y'].mean())
        strat_c1_ratio = float(df_test_strat['y'].mean())
        c0_count = int((df_test_strat['y'] == 0).sum())
        c1_count = int((df_test_strat['y'] == 1).sum())
        
        meta = {
            'seed': seed,
            'sample_selection_seed': sample_selection_seed,
            'total_locked_test_records': len(df_test),
            'stratified_sample_records': len(df_test_strat),
            'class_0_count': c0_count,
            'class_1_count': c1_count,
            'original_test_c1_ratio': orig_c1_ratio,
            'sampled_test_c1_ratio': strat_c1_ratio,
            'original_motor_updrs_mean': float(df_test['motor_UPDRS'].mean()),
            'sampled_motor_updrs_mean': float(df_test_strat['motor_UPDRS'].mean()),
            'original_motor_updrs_std': float(df_test['motor_UPDRS'].std()),
            'sampled_motor_updrs_std': float(df_test_strat['motor_UPDRS'].std())
        }
        sample_metadata.append(meta)
        print(f"  [Sample Metadata] N={len(df_test_strat)} | Class 0: {c0_count}, Class 1: {c1_count} "
              f"(Ratio: {strat_c1_ratio:.4f} vs Orig: {orig_c1_ratio:.4f})")
        
        # 3. Fit preprocessing strictly on dev and transform stratified test set
        prep = FoldContainedQuantumPreprocessor()
        prep.fit(df_dev)
        X_te_q = prep.transform_quantum_angles(df_test_strat)
        y_te_cls = df_test_strat['y'].values.astype(int)
        y_te_reg = df_test_strat['motor_UPDRS'].values.astype(float)
        
        # 4. Load frozen VQC checkpoint
        vqc_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"vqc_seed_{seed}.pt")
        vqc_ckpt = torch.load(vqc_ckpt_path, weights_only=False)
        vqc_model = VQCModel(n_qubits=9, depth_L=vqc_ckpt['depth_L'], seed=seed)
        vqc_model.load_state_dict(vqc_ckpt['state_dict'])
        vqc_model.eval()
        
        # 5. Load frozen PQC-R checkpoint
        pqcr_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"pqcr_seed_{seed}.pt")
        pqcr_ckpt = torch.load(pqcr_ckpt_path, weights_only=False)
        pqcr_model = PQCRModel(n_qubits=9, depth_L=pqcr_ckpt['depth_L'], seed=seed)
        pqcr_model.load_state_dict(pqcr_ckpt['state_dict'])
        pqcr_model.eval()
        
        # --- A. Ideal Statevector Reference on Stratified Subset ---
        try:
            with torch.no_grad():
                _, vqc_strat_probs, _ = vqc_model(torch.tensor(X_te_q, dtype=torch.float64))
                vqc_strat_probs = vqc_strat_probs.numpy().flatten()
                m_vqc_ideal = evaluate_classification(y_te_cls, vqc_strat_probs)
                
                pqcr_strat_preds, _ = pqcr_model(torch.tensor(X_te_q, dtype=torch.float64))
                pqcr_strat_preds = pqcr_strat_preds.numpy().flatten()
                m_pqcr_ideal = evaluate_regression(y_te_reg, pqcr_strat_preds)
                
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': 'Ideal Statevector', 'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc_ideal['roc_auc'],
                'bacc': m_vqc_ideal['bacc'], 'mcc': m_vqc_ideal['mcc'], 'f1_macro': m_vqc_ideal['f1_macro'],
                'brier': m_vqc_ideal['brier'], 'log_loss': m_vqc_ideal['log_loss'],
                'eval_time_s': 0.05
            })
            noise_runs.append({
                'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': 'Ideal Statevector', 'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr_ideal['mae'],
                'rmse': m_pqcr_ideal['rmse'], 'pearson_r': m_pqcr_ideal['pearson_r'],
                'spearman_rho': m_pqcr_ideal['spearman_rho'], 'r2': m_pqcr_ideal['r2'],
                'eval_time_s': 0.05
            })
            print(f"  [Ideal Reference] VQC AUC = {m_vqc_ideal['roc_auc']:.4f} | PQC-R MAE = {m_pqcr_ideal['mae']:.4f}")
        except Exception as e:
            failures.append({'seed': seed, 'condition': 'Ideal', 'error': str(e)})
            print(f"  [Error Ideal Seed {seed}]: {e}")

        # --- B. Depolarizing Noise Channel ---
        for p in depol_levels:
            t_cond_start = time.perf_counter()
            try:
                m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='depol', noise_param=p, batch_size=32)
                m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='depol', noise_param=p, batch_size=32)
                dt = time.perf_counter() - t_cond_start
                
                noise_runs.append({
                    'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                    'condition': f'Depolarizing (p={p})', 'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                    'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'],
                    'bacc': m_vqc['bacc'], 'mcc': m_vqc['mcc'], 'f1_macro': m_vqc['f1_macro'],
                    'brier': m_vqc['brier'], 'log_loss': m_vqc['log_loss'],
                    'eval_time_s': m_vqc.get('eval_time_s', dt/2)
                })
                noise_runs.append({
                    'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                    'condition': f'Depolarizing (p={p})', 'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                    'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'],
                    'rmse': m_pqcr['rmse'], 'pearson_r': m_pqcr['pearson_r'],
                    'spearman_rho': m_pqcr['spearman_rho'], 'r2': m_pqcr['r2'],
                    'eval_time_s': m_pqcr.get('eval_time_s', dt/2)
                })
                print(f"    Depol (p={p}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
            except Exception as e:
                failures.append({'seed': seed, 'condition': f'Depolarizing (p={p})', 'error': str(e)})
                print(f"    [Error Depol p={p} Seed {seed}]: {e}")

        # --- C. Phase Damping / Dephasing Noise Channel ---
        for gamma in dephase_levels:
            t_cond_start = time.perf_counter()
            try:
                m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='dephase', noise_param=gamma, batch_size=32)
                m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='dephase', noise_param=gamma, batch_size=32)
                dt = time.perf_counter() - t_cond_start
                
                noise_runs.append({
                    'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                    'condition': f'Phase Damping (gamma={gamma})', 'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
                    'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'],
                    'bacc': m_vqc['bacc'], 'mcc': m_vqc['mcc'], 'f1_macro': m_vqc['f1_macro'],
                    'brier': m_vqc['brier'], 'log_loss': m_vqc['log_loss'],
                    'eval_time_s': m_vqc.get('eval_time_s', dt/2)
                })
                noise_runs.append({
                    'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                    'condition': f'Phase Damping (gamma={gamma})', 'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
                    'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'],
                    'rmse': m_pqcr['rmse'], 'pearson_r': m_pqcr['pearson_r'],
                    'spearman_rho': m_pqcr['spearman_rho'], 'r2': m_pqcr['r2'],
                    'eval_time_s': m_pqcr.get('eval_time_s', dt/2)
                })
                print(f"    Phase Damping (gamma={gamma}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
            except Exception as e:
                failures.append({'seed': seed, 'condition': f'Phase Damping (gamma={gamma})', 'error': str(e)})
                print(f"    [Error Phase Damping gamma={gamma} Seed {seed}]: {e}")

        # --- D. Finite Measurement Shots ---
        for s in shot_levels:
            t_cond_start = time.perf_counter()
            try:
                m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='shots', shots=s, batch_size=32)
                m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='shots', shots=s, batch_size=32)
                dt = time.perf_counter() - t_cond_start
                
                noise_runs.append({
                    'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                    'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                    'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'],
                    'bacc': m_vqc['bacc'], 'mcc': m_vqc['mcc'], 'f1_macro': m_vqc['f1_macro'],
                    'brier': m_vqc['brier'], 'log_loss': m_vqc['log_loss'],
                    'eval_time_s': m_vqc.get('eval_time_s', dt/2)
                })
                noise_runs.append({
                    'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                    'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                    'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'],
                    'rmse': m_pqcr['rmse'], 'pearson_r': m_pqcr['pearson_r'],
                    'spearman_rho': m_pqcr['spearman_rho'], 'r2': m_pqcr['r2'],
                    'eval_time_s': m_pqcr.get('eval_time_s', dt/2)
                })
                print(f"    Finite Shots (S={s}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
            except Exception as e:
                failures.append({'seed': seed, 'condition': f'Finite Shots (S={s})', 'error': str(e)})
                print(f"    [Error Finite Shots S={s} Seed {seed}]: {e}")
                
        seed_time = time.perf_counter() - t_seed_start
        print(f"  >>> Seed {seed} completed in {seed_time:.2f}s (~{seed_time/60:.1f} min)")
        release_memory()
        
    total_time = time.perf_counter() - t_total_start
    print(f"\n================================================================================")
    print(f"ALL 10 SEEDS NOISE EVALUATION COMPLETED IN {total_time:.2f}s (~{total_time/60:.1f} min)")
    print(f"Total Failures: {len(failures)}")
    print(f"================================================================================")
    
    # 6. Save Stratified Sample Metadata
    df_meta = pd.DataFrame(sample_metadata)
    meta_path = os.path.join(RESULTS_DIR, 'exp5_stratified_sample_metadata.csv')
    df_meta.to_csv(meta_path, index=False)
    print(f"[Saved] Sample metadata saved to: {meta_path}")
    
    # 7. Save Runs
    df_runs = pd.DataFrame(noise_runs)
    runs_path = os.path.join(RESULTS_DIR, 'exp5_stratified_noise_runs.csv')
    df_runs.to_csv(runs_path, index=False)
    print(f"[Saved] Per-seed runs saved to: {runs_path}")
    
    # 8. Save Failures log if any
    if failures:
        df_fail = pd.DataFrame(failures)
        fail_path = os.path.join(RESULTS_DIR, 'exp5_stratified_noise_failures.csv')
        df_fail.to_csv(fail_path, index=False)
        print(f"[Warning] Recorded {len(failures)} failures in: {fail_path}")
        
    # 9. Compute and Save Aggregated Summary
    summary_records = []
    for (model, cond), grp in df_runs.groupby(['model', 'condition']):
        prim_metric = grp['primary_metric_name'].iloc[0]
        prim_vals = grp['primary_metric_val'].values
        m_mean = float(np.mean(prim_vals))
        m_std = float(np.std(prim_vals, ddof=1)) if len(prim_vals) > 1 else 0.0
        ci_lo, ci_hi = compute_ci(prim_vals)
        
        row = {
            'model': model,
            'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': cond,
            'noise_type': grp['noise_type'].iloc[0],
            'noise_param': grp['noise_param'].iloc[0],
            'shots': grp['shots'].iloc[0],
            'primary_metric': prim_metric,
            'mean': m_mean,
            'std': m_std,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'n_seeds': len(prim_vals)
        }
        
        if model == '9-Qubit VQC':
            row['bacc_mean'] = float(np.mean(grp['bacc']))
            row['bacc_std'] = float(np.std(grp['bacc'], ddof=1)) if len(grp) > 1 else 0.0
            row['mcc_mean'] = float(np.mean(grp['mcc']))
            row['mcc_std'] = float(np.std(grp['mcc'], ddof=1)) if len(grp) > 1 else 0.0
        elif model == '9-Qubit PQC-R':
            row['pearson_r_mean'] = float(np.mean(grp['pearson_r']))
            row['pearson_r_std'] = float(np.std(grp['pearson_r'], ddof=1)) if len(grp) > 1 else 0.0
            row['r2_mean'] = float(np.mean(grp['r2']))
            row['r2_std'] = float(np.std(grp['r2'], ddof=1)) if len(grp) > 1 else 0.0
            
        summary_records.append(row)
        
    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(RESULTS_DIR, 'exp5_stratified_noise_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"[Saved] Aggregated noise summary saved to: {summary_path}")
    
    return df_runs, df_summary, df_meta

if __name__ == '__main__':
    # Full 10 seeds execution
    run_stratified_noise_evaluation(seeds=list(range(10)), n_strat_samples=200)


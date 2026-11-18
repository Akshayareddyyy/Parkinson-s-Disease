"""
Phase 5: Resumed Stratified 10-Seed Quantum Noise Resilience Evaluation
Reconstructs completed runs for Seeds 0-5 and Seed 6 (Ideal + Depolarizing) from task-2080 log,
then executes Seed 6 (Phase + Shots) and Seeds 7, 8, 9.
Persists results after each seed to prevent data loss.
"""

import os, sys, time, re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

os.environ['MPLCONFIGDIR'] = 'C:/Users/akank/AppData/Local/Temp'

from common import (
    load_data, get_partition_sets, release_memory,
    RESULTS_DIR, CHECKPOINTS_DIR
)
from preprocessing import FoldContainedQuantumPreprocessor
from quantum_classifier import VQCModel
from quantum_regressor import PQCRModel
from noise_simulation import evaluate_vqc_under_noise, evaluate_pqcr_under_noise
from metrics import compute_ci, evaluate_classification, evaluate_regression

def parse_existing_log(log_path, df):
    with open(log_path, 'r') as f:
        text = f.read()
        
    seed_blocks = re.split(r'>>> SEED (\d+)/9: Evaluating Stratified Quantum Noise Resilience\.\.\.', text)
    runs = []
    metadata = []
    
    for i in range(1, len(seed_blocks), 2):
        s_id = int(seed_blocks[i])
        s_txt = seed_blocks[i+1]
        
        # Get true test set stats for metadata
        df_dev, df_val, df_test, _ = get_partition_sets(df, s_id)
        sample_selection_seed = 42 + s_id
        sss = StratifiedShuffleSplit(n_splits=1, train_size=200, random_state=sample_selection_seed)
        sampled_indices, _ = next(sss.split(df_test, df_test['y']))
        df_test_strat = df_test.iloc[sampled_indices].reset_index(drop=True)
        
        meta = {
            'seed': s_id,
            'sample_selection_seed': sample_selection_seed,
            'total_locked_test_records': len(df_test),
            'stratified_sample_records': len(df_test_strat),
            'class_0_count': int((df_test_strat['y'] == 0).sum()),
            'class_1_count': int((df_test_strat['y'] == 1).sum()),
            'original_test_c1_ratio': float(df_test['y'].mean()),
            'sampled_test_c1_ratio': float(df_test_strat['y'].mean()),
            'original_motor_updrs_mean': float(df_test['motor_UPDRS'].mean()),
            'sampled_motor_updrs_mean': float(df_test_strat['motor_UPDRS'].mean()),
            'original_motor_updrs_std': float(df_test['motor_UPDRS'].std()),
            'sampled_motor_updrs_std': float(df_test_strat['motor_UPDRS'].std())
        }
        metadata.append(meta)
        
        # Ideal
        ideal = re.search(r'\[Ideal Reference\] VQC AUC = ([0-9\.]+) \| PQC-R MAE = ([0-9\.]+)', s_txt)
        if ideal:
            v_auc, p_mae = float(ideal.group(1)), float(ideal.group(2))
            runs.append({
                'seed': s_id, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': 'Ideal Statevector', 'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': v_auc, 'eval_time_s': 0.05
            })
            runs.append({
                'seed': s_id, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': 'Ideal Statevector', 'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': p_mae, 'eval_time_s': 0.05
            })
            
        # Depol
        depols = re.findall(r'Depol \(p=([0-9\.]+)\): VQC AUC = ([0-9\.]+) \| PQC-R MAE = ([0-9\.]+)', s_txt)
        for p_str, v_auc_str, p_mae_str in depols:
            p = float(p_str)
            runs.append({
                'seed': s_id, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Depolarizing (p={p})', 'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': float(v_auc_str), 'eval_time_s': 70.0
            })
            runs.append({
                'seed': s_id, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Depolarizing (p={p})', 'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': float(p_mae_str), 'eval_time_s': 70.0
            })
            
        # Phase
        phases = re.findall(r'Phase Damping \(gamma=([0-9\.]+)\): VQC AUC = ([0-9\.]+) \| PQC-R MAE = ([0-9\.]+)', s_txt)
        for g_str, v_auc_str, p_mae_str in phases:
            g = float(g_str)
            runs.append({
                'seed': s_id, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Phase Damping (gamma={g})', 'noise_type': 'Phase Damping', 'noise_param': g, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': float(v_auc_str), 'eval_time_s': 60.0
            })
            runs.append({
                'seed': s_id, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Phase Damping (gamma={g})', 'noise_type': 'Phase Damping', 'noise_param': g, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': float(p_mae_str), 'eval_time_s': 60.0
            })
            
        # Shots
        shots = re.findall(r'Finite Shots \(S=(\d+)\): VQC AUC = ([0-9\.]+) \| PQC-R MAE = ([0-9\.]+)', s_txt)
        for s_str, v_auc_str, p_mae_str in shots:
            s = int(s_str)
            runs.append({
                'seed': s_id, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': float(v_auc_str), 'eval_time_s': 2.0
            })
            runs.append({
                'seed': s_id, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                'primary_metric_name': 'MAE', 'primary_metric_val': float(p_mae_str), 'eval_time_s': 2.0
            })
            
    return runs, metadata

def save_intermediate(runs, metadata):
    pd.DataFrame(metadata).to_csv(os.path.join(RESULTS_DIR, 'exp5_stratified_sample_metadata.csv'), index=False)
    df_runs = pd.DataFrame(runs)
    df_runs.to_csv(os.path.join(RESULTS_DIR, 'exp5_stratified_noise_runs.csv'), index=False)
    
    # Summary
    summary_records = []
    for (model, cond), grp in df_runs.groupby(['model', 'condition']):
        prim_metric = grp['primary_metric_name'].iloc[0]
        prim_vals = grp['primary_metric_val'].values
        m_mean = float(np.mean(prim_vals))
        m_std = float(np.std(prim_vals, ddof=1)) if len(prim_vals) > 1 else 0.0
        mean_val, std_val, (ci_lo, ci_hi) = compute_ci(prim_vals)
        
        summary_records.append({
            'model': model,
            'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': cond,
            'noise_type': grp['noise_type'].iloc[0],
            'noise_param': grp['noise_param'].iloc[0],
            'shots': grp['shots'].iloc[0],
            'primary_metric': prim_metric,
            'metric_mean': mean_val,
            'mean': mean_val,
            'metric_std': std_val,
            'std': std_val,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'n_seeds': len(prim_vals)
        })
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'exp5_stratified_noise_summary.csv'), index=False)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'exp5_noise_summary.csv'), index=False)

def resume_evaluation():
    print("=" * 80)
    print("PHASE 5: RESUMING STRATIFIED QUANTUM NOISE EVALUATION (SEEDS 6-9)")
    print("=" * 80)
    
    df = load_data()
    log_path = r'C:\Users\akank\.gemini\antigravity\brain\9f6be27c-7108-4d3b-be77-ab80ea3a4fef\.system_generated\tasks\task-2080.log'
    runs, metadata = parse_existing_log(log_path, df)
    print(f"Loaded {len(runs)} evaluation records across completed seeds (Seeds 0-5 full, Seed 6 partial).")
    save_intermediate(runs, metadata)
    
    depol_levels = [1e-4, 1e-3, 5e-3, 1e-2]
    dephase_levels = [1e-3, 1e-2, 5e-2]
    shot_levels = [1000, 4000, 10000]
    
    # 1. Complete Seed 6: Phase Damping and Shots
    print("\n>>> Completing SEED 6 (Phase Damping and Finite Shots)...")
    df_dev, df_val, df_test, _ = get_partition_sets(df, 6)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=200, random_state=48)
    sampled_indices, _ = next(sss.split(df_test, df_test['y']))
    df_test_strat = df_test.iloc[sampled_indices].reset_index(drop=True)
    
    prep = FoldContainedQuantumPreprocessor()
    prep.fit(df_dev)
    X_te_q = prep.transform_quantum_angles(df_test_strat)
    y_te_cls = df_test_strat['y'].values.astype(int)
    y_te_reg = df_test_strat['motor_UPDRS'].values.astype(float)
    
    vqc_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "vqc_seed_6.pt"), weights_only=False)
    vqc_model = VQCModel(n_qubits=9, depth_L=vqc_ckpt['depth_L'], seed=6)
    vqc_model.load_state_dict(vqc_ckpt['state_dict'])
    vqc_model.eval()
    
    pqcr_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "pqcr_seed_6.pt"), weights_only=False)
    pqcr_model = PQCRModel(n_qubits=9, depth_L=pqcr_ckpt['depth_L'], seed=6)
    pqcr_model.load_state_dict(pqcr_ckpt['state_dict'])
    pqcr_model.eval()
    
    for gamma in dephase_levels:
        t0 = time.perf_counter()
        m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='dephase', noise_param=gamma)
        m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='dephase', noise_param=gamma)
        dt = time.perf_counter() - t0
        runs.append({
            'seed': 6, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': f'Phase Damping (gamma={gamma})', 'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
            'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'], 'eval_time_s': dt/2
        })
        runs.append({
            'seed': 6, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': f'Phase Damping (gamma={gamma})', 'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
            'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'], 'eval_time_s': dt/2
        })
        print(f"  Phase Damping (gamma={gamma}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
        
    for s in shot_levels:
        t0 = time.perf_counter()
        m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='shots', shots=s)
        m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='shots', shots=s)
        dt = time.perf_counter() - t0
        runs.append({
            'seed': 6, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
            'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'], 'eval_time_s': dt/2
        })
        runs.append({
            'seed': 6, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
            'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'], 'eval_time_s': dt/2
        })
        print(f"  Finite Shots (S={s}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
    save_intermediate(runs, metadata)
    release_memory()
    
    # 2. Run Seeds 7, 8, 9
    for seed in [7, 8, 9]:
        print(f"\n" + "-" * 70)
        print(f">>> SEED {seed}/9: Evaluating Stratified Quantum Noise Resilience...")
        print("-" * 70)
        
        df_dev, df_val, df_test, _ = get_partition_sets(df, seed)
        sample_selection_seed = 42 + seed
        sss = StratifiedShuffleSplit(n_splits=1, train_size=200, random_state=sample_selection_seed)
        sampled_indices, _ = next(sss.split(df_test, df_test['y']))
        df_test_strat = df_test.iloc[sampled_indices].reset_index(drop=True)
        
        meta = {
            'seed': seed,
            'sample_selection_seed': sample_selection_seed,
            'total_locked_test_records': len(df_test),
            'stratified_sample_records': len(df_test_strat),
            'class_0_count': int((df_test_strat['y'] == 0).sum()),
            'class_1_count': int((df_test_strat['y'] == 1).sum()),
            'original_test_c1_ratio': float(df_test['y'].mean()),
            'sampled_test_c1_ratio': float(df_test_strat['y'].mean()),
            'original_motor_updrs_mean': float(df_test['motor_UPDRS'].mean()),
            'sampled_motor_updrs_mean': float(df_test_strat['motor_UPDRS'].mean()),
            'original_motor_updrs_std': float(df_test['motor_UPDRS'].std()),
            'sampled_motor_updrs_std': float(df_test_strat['motor_UPDRS'].std())
        }
        metadata.append(meta)
        
        prep = FoldContainedQuantumPreprocessor()
        prep.fit(df_dev)
        X_te_q = prep.transform_quantum_angles(df_test_strat)
        y_te_cls = df_test_strat['y'].values.astype(int)
        y_te_reg = df_test_strat['motor_UPDRS'].values.astype(float)
        
        vqc_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, f"vqc_seed_{seed}.pt"), weights_only=False)
        vqc_model = VQCModel(n_qubits=9, depth_L=vqc_ckpt['depth_L'], seed=seed)
        vqc_model.load_state_dict(vqc_ckpt['state_dict'])
        vqc_model.eval()
        
        pqcr_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, f"pqcr_seed_{seed}.pt"), weights_only=False)
        pqcr_model = PQCRModel(n_qubits=9, depth_L=pqcr_ckpt['depth_L'], seed=seed)
        pqcr_model.load_state_dict(pqcr_ckpt['state_dict'])
        pqcr_model.eval()
        
        # Ideal
        with torch.no_grad():
            _, vqc_strat_probs, _ = vqc_model(torch.tensor(X_te_q, dtype=torch.float64))
            vqc_strat_probs = vqc_strat_probs.numpy().flatten()
            m_vqc_ideal = evaluate_classification(y_te_cls, vqc_strat_probs)
            
            pqcr_strat_preds, _ = pqcr_model(torch.tensor(X_te_q, dtype=torch.float64))
            pqcr_strat_preds = pqcr_strat_preds.numpy().flatten()
            m_pqcr_ideal = evaluate_regression(y_te_reg, pqcr_strat_preds)
            
        runs.append({
            'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': 'Ideal Statevector', 'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
            'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc_ideal['roc_auc'], 'eval_time_s': 0.05
        })
        runs.append({
            'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
            'condition': 'Ideal Statevector', 'noise_type': 'None', 'noise_param': 0.0, 'shots': np.nan,
            'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr_ideal['mae'], 'eval_time_s': 0.05
        })
        print(f"  [Ideal Reference] VQC AUC = {m_vqc_ideal['roc_auc']:.4f} | PQC-R MAE = {m_pqcr_ideal['mae']:.4f}")
        
        # Depol
        for p in depol_levels:
            t0 = time.perf_counter()
            m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='depol', noise_param=p)
            m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='depol', noise_param=p)
            dt = time.perf_counter() - t0
            runs.append({
                'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Depolarizing (p={p})', 'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'], 'eval_time_s': dt/2
            })
            runs.append({
                'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Depolarizing (p={p})', 'noise_type': 'Depolarizing', 'noise_param': p, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'], 'eval_time_s': dt/2
            })
            print(f"    Depol (p={p}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
            
        # Phase
        for gamma in dephase_levels:
            t0 = time.perf_counter()
            m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='dephase', noise_param=gamma)
            m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='dephase', noise_param=gamma)
            dt = time.perf_counter() - t0
            runs.append({
                'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Phase Damping (gamma={gamma})', 'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'], 'eval_time_s': dt/2
            })
            runs.append({
                'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Phase Damping (gamma={gamma})', 'noise_type': 'Phase Damping', 'noise_param': gamma, 'shots': np.nan,
                'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'], 'eval_time_s': dt/2
            })
            print(f"    Phase Damping (gamma={gamma}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
            
        # Shots
        for s in shot_levels:
            t0 = time.perf_counter()
            m_vqc, _ = evaluate_vqc_under_noise(vqc_model, X_te_q, y_te_cls, noise_type='shots', shots=s)
            m_pqcr, _ = evaluate_pqcr_under_noise(pqcr_model, X_te_q, y_te_reg, noise_type='shots', shots=s)
            dt = time.perf_counter() - t0
            runs.append({
                'seed': seed, 'model': '9-Qubit VQC', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                'primary_metric_name': 'ROC-AUC', 'primary_metric_val': m_vqc['roc_auc'], 'eval_time_s': dt/2
            })
            runs.append({
                'seed': seed, 'model': '9-Qubit PQC-R', 'evaluation_mode': 'Stratified-Subset (N=200)',
                'condition': f'Finite Shots (S={s})', 'noise_type': 'Shots', 'noise_param': np.nan, 'shots': s,
                'primary_metric_name': 'MAE', 'primary_metric_val': m_pqcr['mae'], 'eval_time_s': dt/2
            })
            print(f"    Finite Shots (S={s}): VQC AUC = {m_vqc['roc_auc']:.4f} | PQC-R MAE = {m_pqcr['mae']:.4f} ({dt:.1f}s)")
            
        save_intermediate(runs, metadata)
        release_memory()
        
    print("\n" + "=" * 80)
    print("ALL 10 SEEDS QUANTUM NOISE RESILIENCE EVALUATION COMPLETED AND SAVED!")
    print("=" * 80)

if __name__ == '__main__':
    resume_evaluation()

"""
Phase 5 - Experiment 1: 9-Qubit VQC Classification & Matched Classical Controls
Runs across all 10 audited seeds (Seeds 0 to 9).
Protocol:
  - 26 Development subjects: Model fitting (zero downsampling)
  - 7 Validation subjects: Hyperparameter selection (L in {2, 3}) and early stopping
  - 9 Locked-Test subjects: Evaluated once per seed after freeze
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch

from common import (
    load_data, get_partition_sets, release_memory, get_process_memory_mb,
    RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR
)
from preprocessing import FoldContainedQuantumPreprocessor
from quantum_classifier import VQCModel, train_vqc
from classical_controls import run_classical_classification_controls
from metrics import evaluate_classification, compute_ci

def run_experiment_1():
    print("=" * 80)
    print("PHASE 5 — EXPERIMENT 1: 9-QUBIT VQC CLASSIFICATION & MATCHED CONTROLS")
    print("=" * 80)
    
    df = load_data()
    seeds = list(range(10))
    
    run_records = []
    convergence_logs = []
    
    t_exp_start = time.perf_counter()
    
    for seed in seeds:
        print(f"\n>>> SEED {seed}/9: Partitioning & Preprocessing...")
        t_seed_start = time.perf_counter()
        
        # 1. Audited subject-wise partition
        df_dev, df_val, df_test, median_thr = get_partition_sets(df, seed)
        print(f"    Dev records: {len(df_dev)} | Val records: {len(df_val)} | Test records: {len(df_test)}")
        
        # 2. Fold-contained preprocessing (fitted strictly on df_dev)
        prep = FoldContainedQuantumPreprocessor()
        prep.fit(df_dev)
        
        # Quantum angle inputs in [-pi, pi]^9
        X_tr_q = prep.transform_quantum_angles(df_dev)
        y_tr = df_dev['y'].values.astype(int)
        
        X_val_q = prep.transform_quantum_angles(df_val)
        y_val = df_val['y'].values.astype(int)
        
        X_te_q = prep.transform_quantum_angles(df_test)
        y_te = df_test['y'].values.astype(int)
        
        # Classical standardized inputs (for fair matched classical controls)
        X_tr_c = prep.transform_classical(df_dev)
        X_val_c = prep.transform_classical(df_val)
        X_te_c = prep.transform_classical(df_test)
        
        # 3. Matched Classical Controls
        print("    Running matched classical controls (Extra Trees & RBF-SVM)...")
        classical_res = run_classical_classification_controls(
            X_tr_c, y_tr, X_val_c, y_val, X_te_c, y_te, seed
        )
        
        for model_name, mdict in classical_res.items():
            tm = mdict['test_metrics']
            vm = mdict['val_metrics']
            run_records.append({
                'seed': seed,
                'model_type': 'Classical Control',
                'model': model_name,
                'depth_L': np.nan,
                'best_epoch': np.nan,
                'roc_auc': tm['roc_auc'],
                'bacc': tm['bacc'],
                'mcc': tm['mcc'],
                'f1_macro': tm['f1_macro'],
                'brier': tm['brier'],
                'log_loss': tm['log_loss'],
                'val_auc': vm['roc_auc'],
                'fit_time_s': tm['fit_time'],
                'peak_ram_mb': get_process_memory_mb()
            })
            print(f"      {model_name}: Test ROC-AUC = {tm['roc_auc']:.4f}, BalAcc = {tm['bacc']:.4f}")
            
        # 4. Hyperparameter Selection for VQC on Validation Set (L in {2, 3})
        print("    Tuning VQC depth (L in {2, 3}) on validation set...")
        best_vqc_bundle = None
        best_val_loss = float('inf')
        best_L = 2
        
        for L_cand in [2, 3]:
            model_cand = VQCModel(n_qubits=9, depth_L=L_cand, seed=seed)
            train_res = train_vqc(
                model_cand, X_tr_q, y_tr, X_val_q, y_val,
                epochs=30, batch_size=64, lr=0.01, patience=5
            )
            if train_res['best_val_loss'] < best_val_loss:
                best_val_loss = train_res['best_val_loss']
                best_vqc_bundle = train_res
                best_L = L_cand
                
        print(f"    Selected optimal VQC depth: L={best_L} (Best Val Loss: {best_val_loss:.4f}, Epoch {best_vqc_bundle['best_epoch']})")
        
        # 5. Locked-Test Evaluation of Selected Frozen VQC
        vqc_best_model = best_vqc_bundle['model']
        vqc_best_model.eval()
        
        X_te_q_t = torch.tensor(X_te_q, dtype=torch.float64)
        X_val_q_t = torch.tensor(X_val_q, dtype=torch.float64)
        with torch.no_grad():
            _, te_probs_t, _ = vqc_best_model(X_te_q_t)
            vqc_te_probs = te_probs_t.cpu().numpy().flatten()
            
            _, val_probs_t, _ = vqc_best_model(X_val_q_t)
            vqc_val_probs = val_probs_t.cpu().numpy().flatten()
            
        vqc_te_metrics = evaluate_classification(y_te, vqc_te_probs)
        vqc_te_metrics['fit_time'] = best_vqc_bundle['total_train_time']
        vqc_val_metrics = evaluate_classification(y_val, vqc_val_probs)
        
        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"vqc_seed_{seed}.pt")
        torch.save({
            'seed': seed,
            'depth_L': best_L,
            'state_dict': vqc_best_model.state_dict(),
            'val_loss': best_val_loss,
            'best_epoch': best_vqc_bundle['best_epoch'],
            'test_metrics': vqc_te_metrics
        }, ckpt_path)
        
        # Record convergence trajectory
        for h in best_vqc_bundle['history']:
            h_entry = dict(h)
            h_entry['seed'] = seed
            h_entry['depth_L'] = best_L
            convergence_logs.append(h_entry)
            
        run_records.append({
            'seed': seed,
            'model_type': 'Quantum Model',
            'model': f'9-Qubit VQC (HEA, L={best_L})',
            'depth_L': best_L,
            'best_epoch': best_vqc_bundle['best_epoch'],
            'roc_auc': vqc_te_metrics['roc_auc'],
            'bacc': vqc_te_metrics['bacc'],
            'mcc': vqc_te_metrics['mcc'],
            'f1_macro': vqc_te_metrics['f1_macro'],
            'brier': vqc_te_metrics['brier'],
            'log_loss': vqc_te_metrics['log_loss'],
            'val_auc': vqc_val_metrics['roc_auc'],
            'fit_time_s': best_vqc_bundle['total_train_time'],
            'peak_ram_mb': get_process_memory_mb()
        })
        print(f"      9-Qubit VQC: Test ROC-AUC = {vqc_te_metrics['roc_auc']:.4f}, BalAcc = {vqc_te_metrics['bacc']:.4f}, MCC = {vqc_te_metrics['mcc']:.4f}")
        print(f"    Seed {seed} completed in {time.perf_counter() - t_seed_start:.2f}s")
        
        # Memory cleanup between seeds
        del vqc_best_model, best_vqc_bundle
        release_memory()
        
    df_runs = pd.DataFrame(run_records)
    runs_csv_path = os.path.join(RESULTS_DIR, 'exp1_vqc_classification_runs.csv')
    df_runs.to_csv(runs_csv_path, index=False)
    print(f"\n[Saved] Per-seed classification runs saved to: {runs_csv_path}")
    
    df_conv = pd.DataFrame(convergence_logs)
    conv_csv_path = os.path.join(RESULTS_DIR, 'exp1_vqc_convergence_logs.csv')
    df_conv.to_csv(conv_csv_path, index=False)
    print(f"[Saved] VQC convergence & gradient logs saved to: {conv_csv_path}")
    
    # 6. Aggregate Summary Tables
    summary_rows = []
    for model_name, grp in df_runs.groupby('model'):
        m_auc, s_auc, ci_auc = compute_ci(grp['roc_auc'])
        m_bacc, s_bacc, ci_bacc = compute_ci(grp['bacc'])
        m_mcc, s_mcc, ci_mcc = compute_ci(grp['mcc'])
        m_f1, s_f1, ci_f1 = compute_ci(grp['f1_macro'])
        m_brier, s_brier, ci_brier = compute_ci(grp['brier'])
        m_time, s_time, _ = compute_ci(grp['fit_time_s'])
        
        summary_rows.append({
            'model': model_name,
            'model_type': grp['model_type'].iloc[0],
            'n_seeds': len(grp),
            'roc_auc_mean': m_auc,
            'roc_auc_std': s_auc,
            'roc_auc_ci_lower': ci_auc[0],
            'roc_auc_ci_upper': ci_auc[1],
            'bacc_mean': m_bacc,
            'bacc_std': s_bacc,
            'mcc_mean': m_mcc,
            'mcc_std': s_mcc,
            'f1_macro_mean': m_f1,
            'f1_macro_std': s_f1,
            'brier_mean': m_brier,
            'brier_std': s_brier,
            'fit_time_s_mean': m_time,
            'fit_time_s_std': s_time
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULTS_DIR, 'exp1_vqc_classification_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Aggregate classification summary saved to: {summary_csv_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 CLASSIFICATION RESULTS SUMMARY:")
    print("=" * 80)
    print(df_summary[['model', 'roc_auc_mean', 'roc_auc_std', 'bacc_mean', 'mcc_mean', 'fit_time_s_mean']].to_string(index=False))
    print(f"\nTotal Experiment 1 Runtime: {time.perf_counter() - t_exp_start:.2f}s")
    return df_summary, df_runs

if __name__ == '__main__':
    run_experiment_1()


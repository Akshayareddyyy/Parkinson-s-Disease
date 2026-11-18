"""
Phase 5 - Experiment 2: 9-Qubit PQC-R Continuous Progression & Matched Controls
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
from quantum_regressor import PQCRModel, train_pqcr
from classical_controls import run_classical_regression_controls
from metrics import evaluate_regression, compute_ci

def run_experiment_2():
    print("=" * 80)
    print("PHASE 5 — EXPERIMENT 2: 9-QUBIT PQC-R REGRESSION & MATCHED CONTROLS")
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
        
        # 2. Fold-contained preprocessing
        prep = FoldContainedQuantumPreprocessor()
        prep.fit(df_dev)
        
        X_tr_q = prep.transform_quantum_angles(df_dev)
        y_tr = df_dev['motor_UPDRS'].values.astype(float)
        
        X_val_q = prep.transform_quantum_angles(df_val)
        y_val = df_val['motor_UPDRS'].values.astype(float)
        
        X_te_q = prep.transform_quantum_angles(df_test)
        y_te = df_test['motor_UPDRS'].values.astype(float)
        
        X_tr_c = prep.transform_classical(df_dev)
        X_val_c = prep.transform_classical(df_val)
        X_te_c = prep.transform_classical(df_test)
        
        # 3. Matched Classical Controls
        print("    Running matched classical controls (ElasticNet & RBF-SVR)...")
        classical_res = run_classical_regression_controls(
            X_tr_c, y_tr, X_val_c, y_val, X_te_c, y_te, df_test, seed
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
                'mae': tm['mae'],
                'rmse': tm['rmse'],
                'r2': tm['r2'],
                'pearson_r': tm['pearson_r'],
                'spearman_rho': tm['spearman_rho'],
                'val_mae': vm['mae'],
                'fit_time_s': tm['fit_time'],
                'peak_ram_mb': get_process_memory_mb()
            })
            print(f"      {model_name}: Test MAE = {tm['mae']:.4f}, r = {tm['pearson_r']:.4f}")
            
        # 4. Hyperparameter Selection for PQC-R on Validation Set (L in {2, 3})
        print("    Tuning PQC-R depth (L in {2, 3}) on validation set...")
        best_pqcr_bundle = None
        best_val_loss = float('inf')
        best_L = 2
        
        for L_cand in [2, 3]:
            model_cand = PQCRModel(n_qubits=9, depth_L=L_cand, seed=seed)
            train_res = train_pqcr(
                model_cand, X_tr_q, y_tr, X_val_q, y_val,
                epochs=30, batch_size=64, lr=0.01, patience=5
            )
            if train_res['best_val_loss'] < best_val_loss:
                best_val_loss = train_res['best_val_loss']
                best_pqcr_bundle = train_res
                best_L = L_cand
                
        print(f"    Selected optimal PQC-R depth: L={best_L} (Best Val Loss: {best_val_loss:.4f}, Epoch {best_pqcr_bundle['best_epoch']})")
        
        # 5. Locked-Test Evaluation of Selected Frozen PQC-R
        pqcr_best_model = best_pqcr_bundle['model']
        pqcr_best_model.eval()
        
        X_te_q_t = torch.tensor(X_te_q, dtype=torch.float64)
        X_val_q_t = torch.tensor(X_val_q, dtype=torch.float64)
        with torch.no_grad():
            preds_t, _ = pqcr_best_model(X_te_q_t)
            pqcr_te_preds = preds_t.cpu().numpy().flatten()
            
            val_preds_t, _ = pqcr_best_model(X_val_q_t)
            pqcr_val_preds = val_preds_t.cpu().numpy().flatten()
            
        pqcr_te_metrics = evaluate_regression(y_te, pqcr_te_preds, df_records=df_test)
        pqcr_te_metrics['fit_time'] = best_pqcr_bundle['total_train_time']
        pqcr_val_metrics = evaluate_regression(y_val, pqcr_val_preds)
        
        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"pqcr_seed_{seed}.pt")
        torch.save({
            'seed': seed,
            'depth_L': best_L,
            'state_dict': pqcr_best_model.state_dict(),
            'val_loss': best_val_loss,
            'best_epoch': best_pqcr_bundle['best_epoch'],
            'test_metrics': pqcr_te_metrics
        }, ckpt_path)
        
        for h in best_pqcr_bundle['history']:
            h_entry = dict(h)
            h_entry['seed'] = seed
            h_entry['depth_L'] = best_L
            convergence_logs.append(h_entry)
            
        run_records.append({
            'seed': seed,
            'model_type': 'Quantum Model',
            'model': f'9-Qubit PQC-R (HEA, L={best_L})',
            'depth_L': best_L,
            'best_epoch': best_pqcr_bundle['best_epoch'],
            'mae': pqcr_te_metrics['mae'],
            'rmse': pqcr_te_metrics['rmse'],
            'r2': pqcr_te_metrics['r2'],
            'pearson_r': pqcr_te_metrics['pearson_r'],
            'spearman_rho': pqcr_te_metrics['spearman_rho'],
            'val_mae': pqcr_val_metrics['mae'],
            'fit_time_s': best_pqcr_bundle['total_train_time'],
            'peak_ram_mb': get_process_memory_mb()
        })
        print(f"      9-Qubit PQC-R: Test MAE = {pqcr_te_metrics['mae']:.4f}, r = {pqcr_te_metrics['pearson_r']:.4f}, R2 = {pqcr_te_metrics['r2']:.4f}")
        print(f"    Seed {seed} completed in {time.perf_counter() - t_seed_start:.2f}s")
        
        del pqcr_best_model, best_pqcr_bundle
        release_memory()
        
    df_runs = pd.DataFrame(run_records)
    runs_csv_path = os.path.join(RESULTS_DIR, 'exp2_pqcr_regression_runs.csv')
    df_runs.to_csv(runs_csv_path, index=False)
    print(f"\n[Saved] Per-seed regression runs saved to: {runs_csv_path}")
    
    df_conv = pd.DataFrame(convergence_logs)
    conv_csv_path = os.path.join(RESULTS_DIR, 'exp2_pqcr_convergence_logs.csv')
    df_conv.to_csv(conv_csv_path, index=False)
    print(f"[Saved] PQC-R convergence & gradient logs saved to: {conv_csv_path}")
    
    # Aggregate Summary
    summary_rows = []
    for model_name, grp in df_runs.groupby('model'):
        m_mae, s_mae, ci_mae = compute_ci(grp['mae'])
        m_rmse, s_rmse, ci_rmse = compute_ci(grp['rmse'])
        m_pr, s_pr, ci_pr = compute_ci(grp['pearson_r'])
        m_sr, s_sr, ci_sr = compute_ci(grp['spearman_rho'])
        m_r2, s_r2, ci_r2 = compute_ci(grp['r2'])
        m_time, s_time, _ = compute_ci(grp['fit_time_s'])
        
        summary_rows.append({
            'model': model_name,
            'model_type': grp['model_type'].iloc[0],
            'n_seeds': len(grp),
            'mae_mean': m_mae,
            'mae_std': s_mae,
            'mae_ci_lower': ci_mae[0],
            'mae_ci_upper': ci_mae[1],
            'rmse_mean': m_rmse,
            'rmse_std': s_rmse,
            'pearson_r_mean': m_pr,
            'pearson_r_std': s_pr,
            'spearman_rho_mean': m_sr,
            'spearman_rho_std': s_sr,
            'r2_mean': m_r2,
            'r2_std': s_r2,
            'fit_time_s_mean': m_time,
            'fit_time_s_std': s_time
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULTS_DIR, 'exp2_pqcr_regression_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Aggregate regression summary saved to: {summary_csv_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 REGRESSION RESULTS SUMMARY:")
    print("=" * 80)
    print(df_summary[['model', 'mae_mean', 'mae_std', 'pearson_r_mean', 'r2_mean', 'fit_time_s_mean']].to_string(index=False))
    print(f"\nTotal Experiment 2 Runtime: {time.perf_counter() - t_exp_start:.2f}s")
    return df_summary, df_runs

if __name__ == '__main__':
    run_experiment_2()


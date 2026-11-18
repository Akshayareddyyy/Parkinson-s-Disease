"""
Phase 5: Publication-Quality Figure Generation
Generates 5 figures in experiments/phase5/figures/:
  1. fig1_vqc_convergence_gradients.png: VQC Loss Convergence & Gradient Diagnostics
  2. fig2_classification_quantum_vs_classical.png: VQC vs Extra Trees vs RBF-SVM
  3. fig3_regression_quantum_vs_classical.png: PQC-R vs ElasticNet vs RBF-SVR
  4. fig4_simulated_noise_resilience.png: Noise degradation profiles (Depol, Dephase, Shots)
  5. fig5_runtime_memory_comparison.png: Computational complexity profiles
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE5_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PHASE5_DIR, 'results')
FIGURES_DIR = os.path.join(PHASE5_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-quality styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def plot_all_figures():
    print("Generating Phase 5 figures...")
    
    # -------------------------------------------------------------------------
    # FIGURE 1: VQC Convergence & Gradient Trajectory
    # -------------------------------------------------------------------------
    conv_csv = os.path.join(RESULTS_DIR, 'exp1_vqc_convergence_logs.csv')
    if os.path.exists(conv_csv):
        df_conv = pd.read_csv(conv_csv)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # Plot Loss Curves
        for seed, grp in df_conv.groupby('seed'):
            axes[0].plot(grp['epoch'], grp['train_loss'], color='#2b5c8f', alpha=0.3, lw=1.2)
            axes[0].plot(grp['epoch'], grp['val_loss'], color='#d95f02', alpha=0.3, lw=1.2, ls='--')
            
        mean_tr = df_conv.groupby('epoch')['train_loss'].mean()
        mean_val = df_conv.groupby('epoch')['val_loss'].mean()
        axes[0].plot(mean_tr.index, mean_tr.values, color='#1b3c65', lw=2.5, label='Mean Train Loss (BCE)')
        axes[0].plot(mean_val.index, mean_val.values, color='#b34700', lw=2.5, ls='--', label='Mean Val Loss (BCE)')
        
        axes[0].set_title("9-Qubit VQC Loss Convergence", fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
        axes[0].grid(True, ls=':', alpha=0.6)
        axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
        
        # Plot Gradient Diagnostics
        for seed, grp in df_conv.groupby('seed'):
            axes[1].plot(grp['epoch'], grp['mean_grad_norm'], color='#7570b3', alpha=0.3, lw=1.2)
            
        mean_grad = df_conv.groupby('epoch')['mean_grad_norm'].mean()
        axes[1].plot(mean_grad.index, mean_grad.values, color='#494480', lw=2.5, label=r'Mean $\|\nabla_{\theta} \mathcal{L}\|_2$ Norm')
        axes[1].axhline(y=1e-4, color='#e41a1c', ls=':', lw=1.5, label='Vanishing Threshold ($10^{-4}$)')
        
        axes[1].set_title("Variational Parameter Gradient Diagnostics", fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel(r"Gradient $L_2$ Norm", fontsize=11)
        axes[1].set_yscale('log')
        axes[1].grid(True, ls=':', alpha=0.6)
        axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)
        
        plt.tight_layout()
        f1_path = os.path.join(FIGURES_DIR, 'fig1_vqc_convergence_gradients.png')
        plt.savefig(f1_path)
        plt.close()
        print(f"[Figure 1 Saved] {f1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Classification Quantum vs Classical Comparison
    # -------------------------------------------------------------------------
    cls_runs_csv = os.path.join(RESULTS_DIR, 'exp1_vqc_classification_runs.csv')
    if os.path.exists(cls_runs_csv):
        df_cls = pd.read_csv(cls_runs_csv)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        models = df_cls['model'].unique()
        colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
        
        # ROC-AUC Boxplot / Scatter
        auc_data = [df_cls[df_cls['model'] == m]['roc_auc'].values for m in models]
        bp1 = axes[0].boxplot(auc_data, labels=[m.replace(' (HEA, L=2)', '').replace(' (9 Feats)', '') for m in models],
                              patch_artist=True, medianprops=dict(color='black', lw=2))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        axes[0].set_title("Test ROC-AUC Comparison (10 Seeds)", fontsize=12, fontweight='bold', pad=10)
        axes[0].set_ylabel("ROC-AUC", fontsize=11)
        axes[0].grid(True, ls=':', alpha=0.6)
        
        # Balanced Accuracy Boxplot
        bacc_data = [df_cls[df_cls['model'] == m]['bacc'].values for m in models]
        bp2 = axes[1].boxplot(bacc_data, labels=[m.replace(' (HEA, L=2)', '').replace(' (9 Feats)', '') for m in models],
                              patch_artist=True, medianprops=dict(color='black', lw=2))
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        axes[1].set_title("Test Balanced Accuracy Comparison (10 Seeds)", fontsize=12, fontweight='bold', pad=10)
        axes[1].set_ylabel("Balanced Accuracy", fontsize=11)
        axes[1].grid(True, ls=':', alpha=0.6)
        
        plt.tight_layout()
        f2_path = os.path.join(FIGURES_DIR, 'fig2_classification_quantum_vs_classical.png')
        plt.savefig(f2_path)
        plt.close()
        print(f"[Figure 2 Saved] {f2_path}")

    # -------------------------------------------------------------------------
    # FIGURE 3: Regression Quantum vs Classical Comparison
    # -------------------------------------------------------------------------
    reg_runs_csv = os.path.join(RESULTS_DIR, 'exp2_pqcr_regression_runs.csv')
    if os.path.exists(reg_runs_csv):
        df_reg = pd.read_csv(reg_runs_csv)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        models = df_reg['model'].unique()
        colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
        
        # MAE Comparison
        mae_data = [df_reg[df_reg['model'] == m]['mae'].values for m in models]
        bp1 = axes[0].boxplot(mae_data, labels=[m.replace(' (HEA, L=2)', '').replace(' (9 Feats)', '') for m in models],
                              patch_artist=True, medianprops=dict(color='black', lw=2))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_title("Test MAE Comparison (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
        axes[0].set_ylabel("Mean Absolute Error (UPDRS)", fontsize=11)
        axes[0].grid(True, ls=':', alpha=0.6)
        
        # Pearson Correlation Comparison
        pr_data = [df_reg[df_reg['model'] == m]['pearson_r'].values for m in models]
        bp2 = axes[1].boxplot(pr_data, labels=[m.replace(' (HEA, L=2)', '').replace(' (9 Feats)', '') for m in models],
                              patch_artist=True, medianprops=dict(color='black', lw=2))
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title("Pearson Correlation r (Higher is Better)", fontsize=12, fontweight='bold', pad=10)
        axes[1].set_ylabel("Pearson Correlation r", fontsize=11)
        axes[1].grid(True, ls=':', alpha=0.6)
        
        plt.tight_layout()
        f3_path = os.path.join(FIGURES_DIR, 'fig3_regression_quantum_vs_classical.png')
        plt.savefig(f3_path)
        plt.close()
        print(f"[Figure 3 Saved] {f3_path}")

    # -------------------------------------------------------------------------
    # FIGURE 4: Simulated Quantum Noise Resilience
    # -------------------------------------------------------------------------
    noise_csv = os.path.join(RESULTS_DIR, 'exp5_noise_summary.csv')
    if os.path.exists(noise_csv):
        df_noise = pd.read_csv(noise_csv)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # VQC Noise Sensitivity
        vqc_noise = df_noise[df_noise['model'] == '9-Qubit VQC']
        depol_vqc = vqc_noise[vqc_noise['noise_type'] == 'Depolarizing'].sort_values('noise_param')
        dephase_vqc = vqc_noise[vqc_noise['noise_type'] == 'Phase Damping'].sort_values('noise_param')
        ideal_vqc = vqc_noise[vqc_noise['condition'] == 'Ideal Statevector']['metric_mean'].values[0]
        
        axes[0].axhline(y=ideal_vqc, color='black', ls='--', lw=1.5, label=f'Ideal Reference ({ideal_vqc:.3f})')
        axes[0].plot(depol_vqc['noise_param'], depol_vqc['metric_mean'], marker='o', color='#e41a1c', lw=2, label='Depolarizing Noise (p)')
        axes[0].plot(dephase_vqc['noise_param'], dephase_vqc['metric_mean'], marker='s', color='#377eb8', lw=2, label='Phase Damping ($\gamma$)')
        
        axes[0].set_title("VQC Noise Resilience (Simulated Channels)", fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel("Noise Parameter ($p$ or $\gamma$)", fontsize=11)
        axes[0].set_ylabel("Test ROC-AUC", fontsize=11)
        axes[0].set_xscale('log')
        axes[0].grid(True, ls=':', alpha=0.6)
        axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
        
        # PQC-R Noise Sensitivity
        pqcr_noise = df_noise[df_noise['model'] == '9-Qubit PQC-R']
        depol_pqcr = pqcr_noise[pqcr_noise['noise_type'] == 'Depolarizing'].sort_values('noise_param')
        dephase_pqcr = pqcr_noise[pqcr_noise['noise_type'] == 'Phase Damping'].sort_values('noise_param')
        ideal_pqcr = pqcr_noise[pqcr_noise['condition'] == 'Ideal Statevector']['metric_mean'].values[0]
        
        axes[1].axhline(y=ideal_pqcr, color='black', ls='--', lw=1.5, label=f'Ideal Reference ({ideal_pqcr:.3f})')
        axes[1].plot(depol_pqcr['noise_param'], depol_pqcr['metric_mean'], marker='o', color='#e41a1c', lw=2, label='Depolarizing Noise (p)')
        axes[1].plot(dephase_pqcr['noise_param'], dephase_pqcr['metric_mean'], marker='s', color='#377eb8', lw=2, label='Phase Damping ($\gamma$)')
        
        axes[1].set_title("PQC-R Noise Resilience (Simulated Channels)", fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel("Noise Parameter ($p$ or $\gamma$)", fontsize=11)
        axes[1].set_ylabel("Test MAE (Lower is Better)", fontsize=11)
        axes[1].set_xscale('log')
        axes[1].grid(True, ls=':', alpha=0.6)
        axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)
        
        plt.tight_layout()
        f4_path = os.path.join(FIGURES_DIR, 'fig4_simulated_noise_resilience.png')
        plt.savefig(f4_path)
        plt.close()
        print(f"[Figure 4 Saved] {f4_path}")

    # -------------------------------------------------------------------------
    # FIGURE 5: Runtime & Peak Memory Comparison
    # -------------------------------------------------------------------------
    if os.path.exists(cls_runs_csv) and os.path.exists(reg_runs_csv):
        df_c = pd.read_csv(cls_runs_csv)
        df_r = pd.read_csv(reg_runs_csv)
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        model_names = ['Extra Trees', 'Classical RBF-SVM', '9-Qubit VQC', 'ElasticNet', 'Classical RBF-SVR', '9-Qubit PQC-R']
        mean_times = [
            df_c[df_c['model'].str.contains('Extra Trees')]['fit_time_s'].mean(),
            df_c[df_c['model'].str.contains('RBF-SVM')]['fit_time_s'].mean(),
            df_c[df_c['model'].str.contains('VQC')]['fit_time_s'].mean(),
            df_r[df_r['model'].str.contains('ElasticNet')]['fit_time_s'].mean(),
            df_r[df_r['model'].str.contains('RBF-SVR')]['fit_time_s'].mean(),
            df_r[df_r['model'].str.contains('PQC-R')]['fit_time_s'].mean()
        ]
        
        bar_colors = ['#1b9e77', '#7570b3', '#d95f02', '#1b9e77', '#7570b3', '#d95f02']
        axes[0].barh(model_names, mean_times, color=bar_colors, alpha=0.85)
        axes[0].set_title("Mean Training Runtime Per Seed (Seconds)", fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel("Time (seconds)", fontsize=11)
        axes[0].grid(True, ls=':', alpha=0.6)
        
        mean_rams = [
            df_c[df_c['model'].str.contains('Extra Trees')]['peak_ram_mb'].mean(),
            df_c[df_c['model'].str.contains('RBF-SVM')]['peak_ram_mb'].mean(),
            df_c[df_c['model'].str.contains('VQC')]['peak_ram_mb'].mean(),
            df_r[df_r['model'].str.contains('ElasticNet')]['peak_ram_mb'].mean(),
            df_r[df_r['model'].str.contains('RBF-SVR')]['peak_ram_mb'].mean(),
            df_r[df_r['model'].str.contains('PQC-R')]['peak_ram_mb'].mean()
        ]
        axes[1].barh(model_names, mean_rams, color=bar_colors, alpha=0.85)
        axes[1].set_title("Peak Host Process RSS Memory (MB)", fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel("Memory (MB)", fontsize=11)
        axes[1].grid(True, ls=':', alpha=0.6)
        
        plt.tight_layout()
        f5_path = os.path.join(FIGURES_DIR, 'fig5_runtime_memory_comparison.png')
        plt.savefig(f5_path)
        plt.close()
        print(f"[Figure 5 Saved] {f5_path}")

if __name__ == '__main__':
    plot_all_figures()


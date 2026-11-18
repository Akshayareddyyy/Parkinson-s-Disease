"""
Phase 4B: Publication Figure Generation Script
Generates high-resolution publication-quality figures:
  1. fig1_advanced_classification.png: ROC-AUC, Balanced Accuracy, & Brier Score
  2. fig2_advanced_regression.png: Continuous Regression MAE & Pearson Correlation r
  3. fig3_feature_engineering_impact.png: Impact of Acoustic Interactions & Transformations
  4. fig4_federated_improvement.png: Federated Optimization, Class-Weighting & Topologies
  5. fig5_personalized_calibration_windows.png: Window size k & Normalization Schemes
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['axes.edgecolor'] = '#333333'
matplotlib.rcParams['axes.linewidth'] = 0.8

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

def plot_fig1():
    csv_path = os.path.join(RESULTS_DIR, "exp1_advanced_classification_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping Fig 1: {csv_path} not found")
        return
    df = pd.read_csv(csv_path).sort_values('roc_auc_mean', ascending=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(df))
    
    # Plot ROC-AUC
    bars1 = ax1.barh(y_pos - 0.2, df['roc_auc_mean'], height=0.35, color='#1f77b4', alpha=0.85, label='ROC-AUC')
    bars2 = ax1.barh(y_pos + 0.2, df['bacc_mean'], height=0.35, color='#2ca02c', alpha=0.85, label='Balanced Accuracy')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['model'], fontsize=11)
    ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.4, 0.75)
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.6, label='Chance (0.50)')
    ax1.set_title('Phase 4B Exp 1: Advanced Centralized Classification Performance', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='lower right', frameon=True)
    
    # Add data labels
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=9, color='#1f77b4')
    for bar in bars2:
        w = bar.get_width()
        ax1.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=9, color='#2ca02c')
        
    plt.tight_layout()
    out_p = os.path.join(FIGS_DIR, "fig1_advanced_classification.png")
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")

def plot_fig2():
    csv_path = os.path.join(RESULTS_DIR, "exp2_advanced_regression_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping Fig 2: {csv_path} not found")
        return
    df = pd.read_csv(csv_path).sort_values('mae_mean', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharey=True)
    y_pos = np.arange(len(df))
    
    # Left: MAE (lower is better)
    colors = ['#d62728' if 'Baseline' in m else '#1f77b4' for m in df['model']]
    ax1.barh(y_pos, df['mae_mean'], color=colors, alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['model'], fontsize=10)
    ax1.set_xlabel('Mean Absolute Error (UPDRS Points - Lower is Better)', fontsize=11, fontweight='bold')
    ax1.set_title('Out-of-Sample Regression MAE', fontsize=12, fontweight='bold')
    for i, v in enumerate(df['mae_mean']):
        ax1.text(v + 0.05, i, f'{v:.2f}', va='center', ha='left', fontsize=9)
    ax1.set_xlim(0, max(df['mae_mean']) * 1.15)
    
    # Right: Pearson r
    ax2.barh(y_pos, df['pearson_r_mean'], color='#ff7f0e', alpha=0.85, height=0.6)
    ax2.set_xlabel('Pearson Correlation r', fontsize=11, fontweight='bold')
    ax2.set_title('Correlation with Ground Truth', fontsize=12, fontweight='bold')
    ax2.axvline(0.0, color='gray', linestyle='--', alpha=0.6)
    for i, v in enumerate(df['pearson_r_mean']):
        ax2.text(v + 0.01, i, f'{v:.2f}', va='center', ha='left', fontsize=9)
    ax2.set_xlim(-0.05, max(0.4, max(df['pearson_r_mean']) * 1.25))
    
    plt.suptitle('Phase 4B Exp 2: Continuous motor-UPDRS Regression Benchmark', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_p = os.path.join(FIGS_DIR, "fig2_advanced_regression.png")
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")

def plot_fig3():
    csv_path = os.path.join(RESULTS_DIR, "exp3_feature_engineering_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping Fig 3: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Classification pipeline comparison
    df_cls = df[df['task'] == 'Classification'].sort_values('roc_auc_mean', ascending=True)
    y_cls = np.arange(len(df_cls))
    ax1.barh(y_cls, df_cls['roc_auc_mean'], color='#2ca02c', alpha=0.85, height=0.55)
    ax1.set_yticks(y_cls)
    ax1.set_yticklabels(df_cls['pipeline'], fontsize=10)
    ax1.set_xlabel('ROC-AUC', fontsize=11, fontweight='bold')
    ax1.set_xlim(0.5, 0.7)
    ax1.set_title('Extra Trees Classification', fontsize=12, fontweight='bold')
    for i, v in enumerate(df_cls['roc_auc_mean']):
        ax1.text(v + 0.003, i, f'{v:.3f}', va='center', ha='left', fontsize=9)
        
    # Regression pipeline comparison
    df_reg = df[df['task'] == 'Regression'].sort_values('mae_mean', ascending=False)
    y_reg = np.arange(len(df_reg))
    ax2.barh(y_reg, df_reg['mae_mean'], color='#1f77b4', alpha=0.85, height=0.55)
    ax2.set_yticks(y_reg)
    ax2.set_yticklabels(df_reg['pipeline'], fontsize=10)
    ax2.set_xlabel('MAE (UPDRS Points - Lower is Better)', fontsize=11, fontweight='bold')
    ax2.set_xlim(6.0, 8.0)
    ax2.set_title('ElasticNet Regression', fontsize=12, fontweight='bold')
    for i, v in enumerate(df_reg['mae_mean']):
        ax2.text(v + 0.02, i, f'{v:.2f}', va='center', ha='left', fontsize=9)
        
    plt.suptitle('Phase 4B Exp 3: Acoustic Domain Engineering & Transformation Impact', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_p = os.path.join(FIGS_DIR, "fig3_feature_engineering_impact.png")
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")

def plot_fig4():
    csv_path = os.path.join(RESULTS_DIR, "exp4_federated_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping Fig 4: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    df_cls = df[df['task'] == 'Classification'].sort_values('roc_auc_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(df_cls))
    
    colors = []
    for topo in df_cls['topology']:
        if topo == 'Centralized':
            colors.append('#7f7f7f')
        elif topo == 'Clinic-5':
            colors.append('#9467bd')
        else:
            colors.append('#1f77b4')
            
    bars = ax.barh(y_pos, df_cls['roc_auc_mean'], color=colors, alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{m} ({t})" for m, t in zip(df_cls['model'], df_cls['topology'])], fontsize=10)
    ax.set_xlabel('Locked-Test ROC-AUC', fontsize=12, fontweight='bold')
    ax.set_xlim(0.45, 0.72)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Phase 4B Exp 4: Federated Learning Improvement & Stabilization', fontsize=13, fontweight='bold', pad=12)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=9)
        
    plt.tight_layout()
    out_p = os.path.join(FIGS_DIR, "fig4_federated_improvement.png")
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")

def plot_fig5():
    csv_path = os.path.join(RESULTS_DIR, "exp5_personalized_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping Fig 5: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 1. Effect of calibration window k on Absolute MAE (using Z-score)
    df_z = df[df['norm_type'] == 'z_score']
    for m_name in ['Extra Trees Regressor', 'Random Forest Regressor', 'Ridge Regressor', 'Zero Delta Dummy']:
        sub = df_z[df_z['model'] == m_name].sort_values('k_window')
        if len(sub) > 0:
            ax1.plot(sub['k_window'], sub['abs_mae_mean'], marker='o', linewidth=2, label=m_name)
            
    ax1.set_xlabel('Calibration Window k (Recordings)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Absolute motor-UPDRS MAE (Points)', fontsize=11, fontweight='bold')
    ax1.set_title('Impact of Calibration Horizon k (Z-Score)', fontsize=12, fontweight='bold')
    ax1.set_xticks([3, 5, 10, 15])
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Normalization comparison at k=10
    df_k10 = df[df['k_window'] == 10]
    norm_models = ['Extra Trees Regressor', 'Random Forest Regressor', 'Ridge Regressor']
    df_bar = df_k10[df_k10['model'].isin(norm_models)]
    
    x_idx = np.arange(len(norm_models))
    width = 0.25
    
    norms = ['z_score', 'robust_iqr', 'min_max']
    norm_labels = ['Z-Score', 'Robust (IQR)', 'Min-Max']
    palette = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    for i, (n_key, n_lbl, col) in enumerate(zip(norms, norm_labels, palette)):
        vals = []
        for m in norm_models:
            v = df_bar[(df_bar['model'] == m) & (df_bar['norm_type'] == n_key)]['abs_mae_mean'].values
            vals.append(v[0] if len(v) > 0 else 0.0)
        ax2.bar(x_idx + i * width, vals, width=width, label=n_lbl, color=col, alpha=0.85)
        
    ax2.set_xticks(x_idx + width)
    ax2.set_xticklabels(norm_models, fontsize=10)
    ax2.set_ylabel('Absolute motor-UPDRS MAE (Points)', fontsize=11, fontweight='bold')
    ax2.set_title('Normalization Scheme Comparison (k=10)', fontsize=12, fontweight='bold')
    ax2.set_ylim(2.0, 3.5)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Phase 4B Exp 5: Personalized Longitudinal Progression Dynamics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_p = os.path.join(FIGS_DIR, "fig5_personalized_calibration_windows.png")
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")

def plot_all():
    print("=" * 60)
    print("GENERATING PHASE 4B PUBLICATION FIGURES")
    print("=" * 60)
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    print("=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)

if __name__ == '__main__':
    plot_all()


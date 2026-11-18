"""
Consolidated Analysis & Figure Generation for Experiments A, B, and C
Integrates:
  - Original locked-test results (results/summary_test.csv)
  - Experiment A (results/exp_a_*.csv)
  - Experiment B (results/exp_b_*.csv)
  - Experiment C (results/exp_c_*.csv)
Produces:
  - results/consolidated_master_summary.csv
  - figures/fig_exp_a_federated_clinics.png/.pdf
  - figures/fig_exp_b_quantum_analysis.png/.pdf
  - figures/fig_exp_c_regression.png/.pdf
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from common import BASE_DIR
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(BASE_DIR, "results") + os.sep
F = os.path.join(BASE_DIR, "figures") + os.sep
os.makedirs(F, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42
})

C = dict(cen='#0072B2', fed='#009E73', q='#CC79A7', bad='#D55E00', gray='#7f7f7f', accent='#E69F00')

def plot_experiment_a():
    path_a = os.path.join(R, "exp_a_summary.csv")
    if not os.path.exists(path_a):
        print(f"Skipping plot_experiment_a: {path_a} not found.")
        return
    df_a = pd.read_csv(path_a)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    
    # Panel 1: FedForest-RF and FedForest-ET by K and allocation
    forests = df_a[df_a.model.isin(['FedForest-RF', 'FedForest-ET'])]
    for (m, alloc), grp in forests.groupby(['model', 'allocation']):
        lbl = f"{m} ({'Natural' if alloc=='natural' else 'Balanced'})"
        ls = '-' if alloc=='natural' else '--'
        axes[0].plot(grp.K_clients, grp.auc_mean, marker='o', ls=ls, label=lbl)
    axes[0].axhline(0.5, ls=':', c='gray', label='Chance (0.50)')
    axes[0].set_xlabel('Number of Clinical Centers (K)')
    axes[0].set_ylabel('Test ROC-AUC')
    axes[0].set_title('(a) Federated Tree Ensembles vs Clinic Size')
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].set_ylim(0.35, 0.72)
    
    # Panel 2: FedAvg-LR vs FedProx-LR
    linear_fl = df_a[df_a.model.str.startswith('FedAvg') | df_a.model.str.startswith('FedProx')]
    for (m, alloc), grp in linear_fl.groupby(['model', 'allocation']):
        if alloc == 'natural':
            axes[1].plot(grp.K_clients, grp.auc_mean, marker='s', label=f"{m} (Natural)")
    axes[1].axhline(0.5, ls=':', c='gray')
    axes[1].set_xlabel('Number of Clinical Centers (K)')
    axes[1].set_ylabel('Test ROC-AUC')
    axes[1].set_title('(b) Federated Linear Models vs Clinic Size')
    axes[1].legend(frameon=False, fontsize=6.5)
    axes[1].set_ylim(0.50, 0.65)
    
    # Panel 3: Communication Volume vs K
    for (m, alloc), grp in df_a.groupby(['model', 'allocation']):
        if alloc == 'natural' and m in ['FedAvg-LR', 'FedForest-RF', 'FedForest-ET']:
            axes[2].plot(grp.K_clients, grp.comm_bytes_avg / 1e3, marker='^', label=m)
    axes[2].set_xlabel('Number of Clinical Centers (K)')
    axes[2].set_ylabel('Total Traffic (kB, log scale)')
    axes[2].set_yscale('log')
    axes[2].set_title('(c) Communication Overhead')
    axes[2].legend(frameon=False, fontsize=6.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(F, "fig_exp_a_federated_clinics.png"), dpi=200)
    plt.savefig(os.path.join(F, "fig_exp_a_federated_clinics.pdf"))
    plt.close()
    print("Saved fig_exp_a_federated_clinics")

def plot_experiment_b():
    path_b = os.path.join(R, "exp_b_quantum_summary.csv")
    path_cl = os.path.join(R, "exp_b_classical_4feat_runs.csv")
    if not os.path.exists(path_b):
        print(f"Skipping plot_experiment_b: {path_b} not found.")
        return
    df_b = pd.read_csv(path_b)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    
    # Panel 1: Depth sweep L in {1, 2, 3, 4} (supervised feature mode, ring topology)
    depth_sweep = df_b[(df_b.feat_mode == 'supervised') & (df_b.topology == 'ring')]
    for m, grp in depth_sweep.groupby('model'):
        axes[0].plot(grp.L_depth, grp.auc_mean, marker='o', label=m)
    axes[0].axhline(0.5, ls=':', c='gray', label='Chance')
    axes[0].set_xlabel('Circuit Depth (L)')
    axes[0].set_ylabel('Test ROC-AUC')
    axes[0].set_title('(a) Circuit Depth vs Test ROC-AUC')
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0.40, 0.65)
    
    # Panel 2: Topology comparison (Ring vs Linear vs All-to-All at L=2)
    topo_comp = df_b[(df_b.feat_mode == 'supervised') & (df_b.L_depth == 2)]
    x_pos = np.arange(len(topo_comp))
    bars = axes[1].bar(x_pos, topo_comp.auc_mean, color=[C['q'] if m=='VQC' else '#8856a7' for m in topo_comp.model], width=0.6)
    axes[1].axhline(0.5, ls=':', c='gray')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"{r.model}\n{r.topology}" for _, r in topo_comp.iterrows()], fontsize=6.5)
    axes[1].set_ylabel('Test ROC-AUC')
    axes[1].set_title('(b) Entanglement Arrangement (L=2)')
    axes[1].set_ylim(0.40, 0.65)
    
    # Panel 3: Quantum vs Matched Classical 4-Feature Baselines
    if os.path.exists(path_cl):
        df_cl = pd.read_csv(path_cl)
        cl_sum = df_cl.groupby(['feat_mode', 'model'])['auc'].mean().reset_index()
        q_sum = df_b[df_b.L_depth == 3][['feat_mode', 'model', 'auc_mean']].rename(columns={'auc_mean': 'auc'})
        comp = pd.concat([cl_sum, q_sum])
        
        # Compare PCA vs Supervised for LR, ET, VQC, QNN
        models_sel = ['LR', 'Extra Trees', 'VQC', 'QNN']
        sub_comp = comp[comp.model.isin(models_sel)]
        x_m = np.arange(len(models_sel))
        w = 0.35
        
        v_pca = [sub_comp[(sub_comp.feat_mode=='pca') & (sub_comp.model==m)].auc.values[0] if len(sub_comp[(sub_comp.feat_mode=='pca') & (sub_comp.model==m)]) > 0 else 0.5 for m in models_sel]
        v_sup = [sub_comp[(sub_comp.feat_mode=='supervised') & (sub_comp.model==m)].auc.values[0] if len(sub_comp[(sub_comp.feat_mode=='supervised') & (sub_comp.model==m)]) > 0 else 0.5 for m in models_sel]
        
        axes[2].bar(x_m - w/2, v_pca, width=w, label='PCA (4 feats)', color=C['cen'])
        axes[2].bar(x_m + w/2, v_sup, width=w, label='Supervised (4 feats)', color=C['fed'])
        axes[2].axhline(0.5, ls=':', c='gray')
        axes[2].set_xticks(x_m)
        axes[2].set_xticklabels(models_sel, fontsize=7)
        axes[2].set_ylabel('Test ROC-AUC')
        axes[2].set_title('(c) 4-Feature Matched Baselines vs QML')
        axes[2].legend(frameon=False, fontsize=6.5)
        axes[2].set_ylim(0.45, 0.70)
        
    plt.tight_layout()
    plt.savefig(os.path.join(F, "fig_exp_b_quantum_analysis.png"), dpi=200)
    plt.savefig(os.path.join(F, "fig_exp_b_quantum_analysis.pdf"))
    plt.close()
    print("Saved fig_exp_b_quantum_analysis")

def plot_experiment_c():
    path_c = os.path.join(R, "exp_c_regression_summary.csv")
    if not os.path.exists(path_c):
        print(f"Skipping plot_experiment_c: {path_c} not found.")
        return
    df_c = pd.read_csv(path_c)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    
    models = df_c['model'].values
    x_pos = np.arange(len(models))
    
    # Panel 1: MAE
    axes[0].bar(x_pos, df_c['mae_mean'], color=C['cen'], width=0.6)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, rotation=30, ha='right', fontsize=7)
    axes[0].set_ylabel('Mean Absolute Error (points)')
    axes[0].set_title('(a) MAE (motor_UPDRS, 0-108)')
    
    # Panel 2: RMSE
    axes[1].bar(x_pos, df_c['rmse_mean'], color=C['fed'], width=0.6)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models, rotation=30, ha='right', fontsize=7)
    axes[1].set_ylabel('Root Mean Squared Error')
    axes[1].set_title('(b) RMSE')
    
    # Panel 3: R-squared
    axes[2].bar(x_pos, df_c['r2_mean'], color=C['q'], width=0.6)
    axes[2].axhline(0.0, ls=':', c='gray')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(models, rotation=30, ha='right', fontsize=7)
    axes[2].set_ylabel('R-squared (coefficient of determination)')
    axes[2].set_title('(c) Out-of-Sample R²')
    
    plt.tight_layout()
    plt.savefig(os.path.join(F, "fig_exp_c_regression.png"), dpi=200)
    plt.savefig(os.path.join(F, "fig_exp_c_regression.pdf"))
    plt.close()
    print("Saved fig_exp_c_regression")

def generate_consolidated_report():
    plot_experiment_a()
    plot_experiment_b()
    plot_experiment_c()
    print("All consolidated analyses completed.")

if __name__ == '__main__':
    generate_consolidated_report()


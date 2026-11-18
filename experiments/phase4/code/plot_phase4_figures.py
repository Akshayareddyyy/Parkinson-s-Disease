"""
Phase 4: Publication Visualization Generator for Federated Learning
Generates high-resolution publication figures in experiments/phase4/figures/
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(BASE_DIR, 'code'))
os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, '.tmp_mpl')

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def plot_all_figures():
    print("Generating Phase 4 publication figures...")
    
    # -------------------------------------------------------------------------
    # Figure 1: Matched Model Comparisons (Centralized vs Federated)
    # -------------------------------------------------------------------------
    f_sum = os.path.join(R, "phase4_federated_summary.csv")
    if os.path.exists(f_sum):
        df_sum = pd.read_csv(f_sum)
        
        # Panel A: Classification Matched Pairs
        df_cls = df_sum[df_sum['task'] == 'Classification'].copy()
        cls_pairs = [
            ('Centralized LR', 'Centralized Logistic Regression', 'Centralized'),
            ('FedAvg-LR (26 Clients)', 'FedAvg-LR (100% clients)', 'Federated'),
            ('FedProx-LR (mu=0.1)', 'FedProx-LR (mu=0.1)', 'Federated'),
            ('Centralized RF', 'Centralized Random Forest', 'Centralized'),
            ('CED-Forest (RF)', 'CED-Forest (Random Forest)', 'Federated (CED)'),
            ('Centralized ET', 'Centralized Extra Trees', 'Centralized'),
            ('CED-Forest (ET)', 'CED-Forest (Extra Trees)', 'Federated (CED)')
        ]
        
        plot_names = []
        aucs = []
        auc_errs = []
        colors = []
        
        for p_label, m_name, m_type in cls_pairs:
            row = df_cls[df_cls['model'] == m_name]
            if len(row) > 0:
                plot_names.append(p_label)
                aucs.append(row['roc_auc_mean'].values[0])
                auc_errs.append(row['roc_auc_std'].values[0])
                colors.append('#2b5c8f' if 'Centralized' in m_type else ('#55A868' if 'CED' in m_type else '#e7298a'))
                
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(plot_names))
        ax.barh(y_pos, aucs, xerr=auc_errs, align='center', color=colors, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_names)
        ax.invert_yaxis()
        ax.set_xlabel('Test ROC-AUC')
        ax.set_title('Strict Matched Comparison: Centralized vs. Federated Learning')
        ax.axvline(0.50, color='gray', linestyle='--', alpha=0.7, label='Chance (0.50)')
        ax.axvline(0.629, color='crimson', linestyle=':', label='Phase 3 Centralized Benchmark (0.629)')
        ax.legend(loc='lower right')
        plt.tight_layout()
        out_fig1 = os.path.join(FIG_DIR, "fig_phase4_matched_comparisons.png")
        plt.savefig(out_fig1)
        plt.close()
        print(f"Saved: {out_fig1}")
        
    # -------------------------------------------------------------------------
    # Figure 2: Convergence Curves (Round-by-Round History)
    # -------------------------------------------------------------------------
    f_hist = os.path.join(R, "phase4_fl_convergence_history.csv")
    if os.path.exists(f_hist):
        df_hist = pd.read_csv(f_hist)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by model and round
        for m_name, grp in df_hist.groupby('model'):
            round_agg = grp.groupby('round').mean(numeric_only=True)
            ax1.plot(round_agg.index, round_agg['train_loss'], label=m_name)
            ax2.plot(round_agg.index, round_agg['val_auc'], label=m_name)
            
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Global Training Loss (Cross-Entropy)')
        ax1.set_title('(A) Federated Loss Convergence')
        ax1.legend(loc='upper right', fontsize=8)
        
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Server Validation ROC-AUC')
        ax2.set_title('(B) Validation ROC-AUC Trajectory')
        ax2.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        out_fig2 = os.path.join(FIG_DIR, "fig_phase4_convergence.png")
        plt.savefig(out_fig2)
        plt.close()
        print(f"Saved: {out_fig2}")
        
    # -------------------------------------------------------------------------
    # Figure 3: Impact of Client Heterogeneity & Topology
    # -------------------------------------------------------------------------
    if os.path.exists(f_sum):
        df_sum = pd.read_csv(f_sum)
        df_topo = df_sum[df_sum['model'].str.contains('FedAvg-LR|Centralized Logistic')].copy()
        
        fig, ax = plt.subplots(figsize=(9, 4.5))
        y_pos = np.arange(len(df_topo))
        ax.barh(y_pos, df_topo['roc_auc_mean'], xerr=df_topo['roc_auc_std'], align='center', color='#7570b3', capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_topo['model'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Test ROC-AUC')
        ax.set_title('Impact of Client Allocation Topology on Federated Performance')
        plt.tight_layout()
        out_fig3 = os.path.join(FIG_DIR, "fig_phase4_client_heterogeneity.png")
        plt.savefig(out_fig3)
        plt.close()
        print(f"Saved: {out_fig3}")
        
    # -------------------------------------------------------------------------
    # Figure 4: Communication Cost vs Performance Trade-off
    # -------------------------------------------------------------------------
    if os.path.exists(f_sum):
        df_sum = pd.read_csv(f_sum)
        df_fl = df_sum[df_sum['task'] == 'Classification'].copy()
        
        fig, ax = plt.subplots(figsize=(9, 5))
        for _, row in df_fl.iterrows():
            kb = row['comm_bytes_avg'] / 1024.0
            auc = row['roc_auc_mean']
            name = row['model']
            color = '#2b5c8f' if 'Centralized' in row['methodology'] else '#d95f02'
            ax.scatter(kb, auc, s=120, color=color, alpha=0.8)
            ax.annotate(name, (kb, auc), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        ax.set_xlabel('Communication Overhead (Kilobytes Transmitted)')
        ax.set_ylabel('Test ROC-AUC')
        ax.set_title('Communication Overhead vs. Generalization Trade-Off')
        plt.tight_layout()
        out_fig4 = os.path.join(FIG_DIR, "fig_phase4_communication_tradeoff.png")
        plt.savefig(out_fig4)
        plt.close()
        print(f"Saved: {out_fig4}")

    print("All Phase 4 figures generated successfully.")

if __name__ == '__main__':
    plot_all_figures()

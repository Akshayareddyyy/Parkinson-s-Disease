"""
Phase 3B: Visualization generator for comparison figures
Generates publication-quality charts in experiments/phase3b/figures/
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
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def plot_all_figures():
    print("Generating Phase 3B publication figures...")
    
    # -----------------------------------------------------------------
    # Figure 1: Experiment 1 - Subject Weighting Impact
    # -----------------------------------------------------------------
    f1_csv = os.path.join(R, "exp1_subject_weighting_summary.csv")
    if os.path.exists(f1_csv):
        df1 = pd.read_csv(f1_csv)
        df1_cls = df1[df1['task'] == 'Classification']
        
        models = df1_cls['model'].unique()
        ord_auc = [df1_cls[(df1_cls['model']==m) & (df1_cls['weighting_mode']=='Ordinary')]['roc_auc_mean'].values[0] for m in models]
        bal_auc = [df1_cls[(df1_cls['model']==m) & (df1_cls['weighting_mode']=='Subject-Balanced')]['roc_auc_mean'].values[0] for m in models]
        
        ord_err = [df1_cls[(df1_cls['model']==m) & (df1_cls['weighting_mode']=='Ordinary')]['roc_auc_std'].values[0] for m in models]
        bal_err = [df1_cls[(df1_cls['model']==m) & (df1_cls['weighting_mode']=='Subject-Balanced')]['roc_auc_std'].values[0] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 5))
        rects1 = ax.bar(x - width/2, ord_auc, width, yerr=ord_err, label='Ordinary Training ($w_{ij}=1$)', color='#4C72B0', capsize=4)
        rects2 = ax.bar(x + width/2, bal_auc, width, yerr=bal_err, label='Subject-Balanced Training ($w_{ij}=1/n_j$)', color='#55A868', capsize=4)
        
        ax.set_ylabel('Test ROC-AUC')
        ax.set_title('Impact of Subject-Balanced Sample Weighting on Binary Severity Classification')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylim(0.40, 0.75)
        ax.axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='Chance Baseline (0.50)')
        ax.axhline(0.629, color='crimson', linestyle=':', label='Phase 3 ET Benchmark (0.629)')
        ax.legend(loc='lower right')
        plt.tight_layout()
        out_fig1 = os.path.join(FIG_DIR, "fig_exp1_sample_weighting_effect.png")
        plt.savefig(out_fig1)
        plt.close()
        print(f"Saved: {out_fig1}")

    # -----------------------------------------------------------------
    # Figure 2: Experiment 2 - Continuous Regression Model Comparison
    # -----------------------------------------------------------------
    f2_csv = os.path.join(R, "exp2_improved_algorithms_summary.csv")
    if os.path.exists(f2_csv):
        df2 = pd.read_csv(f2_csv)
        df2_reg = df2[df2['task'] == 'Regression'].sort_values('mae_mean')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Panel A: MAE Comparison
        y_pos = np.arange(len(df2_reg))
        ax1.barh(y_pos, df2_reg['mae_mean'], xerr=df2_reg['mae_std'], align='center', color='#2b5c8f', capsize=3)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df2_reg['model'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Mean Absolute Error (motor_UPDRS points)')
        ax1.set_title('(A) Regression Error Comparison (MAE)')
        
        # Panel B: Pearson Correlation
        ax2.barh(y_pos, df2_reg['pearson_r_mean'], align='center', color='#d95f02')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(df2_reg['model'])
        ax2.invert_yaxis()
        ax2.set_xlabel('Pearson Correlation (r)')
        ax2.set_title('(B) Correlation with True Progression')
        
        plt.tight_layout()
        out_fig2 = os.path.join(FIG_DIR, "fig_exp2_regression_model_comparison.png")
        plt.savefig(out_fig2)
        plt.close()
        print(f"Saved: {out_fig2}")

    # -----------------------------------------------------------------
    # Figure 3: Experiment 5 - Personalized vs Global Prediction
    # -----------------------------------------------------------------
    f5_csv = os.path.join(R, "exp5_subject_normalization_summary.csv")
    if os.path.exists(f5_csv):
        df5 = pd.read_csv(f5_csv)
        fig, ax = plt.subplots(figsize=(9, 5))
        
        scenarios = []
        maes = []
        errs = []
        for _, row in df5.iterrows():
            scenarios.append(f"{row['scenario']} ({row['target_type']})\n[{row['model']}, k={row['calibration_k']}]")
            maes.append(row['mae_mean'])
            errs.append(row['mae_std'])
            
        y_pos = np.arange(len(scenarios))
        ax.barh(y_pos, maes, xerr=errs, align='center', color='#7570b3', capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scenarios, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Absolute Error (MAE)')
        ax.set_title('Personalized Subject-Calibrated Monitoring vs Global Zero-Shot')
        plt.tight_layout()
        out_fig3 = os.path.join(FIG_DIR, "fig_exp5_personalized_vs_global.png")
        plt.savefig(out_fig3)
        plt.close()
        print(f"Saved: {out_fig3}")
        
    # -----------------------------------------------------------------
    # Figure 4: Experiment 3 - Feature Engineering & Selection Impact
    # -----------------------------------------------------------------
    f3_csv = os.path.join(R, "exp3_feature_engineering_summary.csv")
    if os.path.exists(f3_csv):
        df3 = pd.read_csv(f3_csv)
        df3_cls = df3[df3['task'] == 'Classification'].copy()
        df3_cls['pipeline'] = df3_cls['preprocessor'] + ' + ' + df3_cls['selector']
        df3_cls = df3_cls.sort_values('roc_auc_mean', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        y_pos = np.arange(len(df3_cls))
        ax.barh(y_pos, df3_cls['roc_auc_mean'], xerr=df3_cls['roc_auc_std'], align='center', color='#1b9e77', capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df3_cls['pipeline'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Test ROC-AUC')
        ax.set_title('Top 10 Feature Transformations & Selection Pipelines (Extra Trees)')
        ax.axvline(0.629, color='crimson', linestyle=':', label='Phase 3 ET Baseline (0.629)')
        ax.legend(loc='lower right')
        plt.tight_layout()
        out_fig4 = os.path.join(FIG_DIR, "fig_exp3_feature_engineering_impact.png")
        plt.savefig(out_fig4)
        plt.close()
        print(f"Saved: {out_fig4}")

    print("All Phase 3B figures generated successfully.")

if __name__ == '__main__':
    plot_all_figures()

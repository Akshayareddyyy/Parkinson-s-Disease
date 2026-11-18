import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR
import numpy as np,pandas as pd
R=os.path.join(BASE_DIR,"results")+os.sep
O=os.path.join(BASE_DIR,"paper_final","tables6")+os.sep
os.makedirs(O,exist_ok=True)
S=pd.read_csv(R+"summary_test.csv").set_index('model'); sig=pd.read_csv(R+"significance.csv"); cost=pd.read_csv(R+"cost.csv").set_index('model')
nm={'LR':'LR','SVM':'Linear SVM','RF':'Random Forest','ET':'Extra Trees','FedAvg-LR':'FedAvg-LR','FedAvg-LR (50% clients)':'FedAvg-LR (50\\%)','FedProx-LR (mu=0.1)':'FedProx-LR','FedForest-RF':'Fed-RF','FedForest-ET':'Fed-ET','VQC':'VQC (sim.)','QNN':'QNN (sim.)'}
order=['LR','SVM','RF','ET','FedAvg-LR','FedAvg-LR (50% clients)','FedProx-LR (mu=0.1)','FedForest-RF','FedForest-ET','VQC','QNN']
f=lambda m,k:f"{S.loc[m,k]:.3f}"
rows=""
for i,m in enumerate(order):
    if i in(4,9): rows+="\\midrule\n"
    rows+=nm[m]+" & "+" & ".join(f(m,k) for k in ['acc','bacc','prec','rec','f1m','f1w','prauc','mcc'])+f" & {S.loc[m,'auc']:.3f}$\\pm${S.loc[m,'auc_ci']:.3f}\\\\\n"
open(O+"t_all.tex",'w').write("\\begin{table*}[!t]\\centering\\caption{Locked-test results (mean over 10 subject-wise partitions; ROC-AUC with 95\\% $t$-interval). Top: centralized; middle: federated; bottom: simulated quantum. Precision/recall refer to the more-severe class.}\\label{tab:all}\\scriptsize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lccccccccc}\\toprule Model & Acc. & Bal.Acc. & Prec. & Rec. & Macro-F1 & Wtd-F1 & PR-AUC & MCC & ROC-AUC\\\\\\midrule\n"+rows+"\\bottomrule\\end{tabular}\\end{table*}\n")
t=sig[sig.test.str.startswith('AUC')].set_index('model'); w=sig[sig.test.str.startswith('Wilcoxon')].set_index('model')
rows=""
for m in order:
    d=f"{w.loc[m,'mean_diff']:+.3f} & {w.loc[m,'p_holm']:.3f}" if m in w.index else "-- & --"
    rows+=f"{nm[m]} & {t.loc[m,'p_holm']:.3f} & {d}\\\\\n"
open(O+"t_stats.tex",'w').write("\\begin{table}[!t]\\centering\\caption{Holm-adjusted $p$-values over 10 partitions: ROC-AUC against 0.5 (one-sample $t$-test) and paired Wilcoxon test against Extra Trees ($\\Delta$AUC = ET $-$ model).}\\label{tab:stats}\\scriptsize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lccc}\\toprule Model & $p$ vs 0.5 & $\\Delta$AUC & $p$ vs ET\\\\\\midrule\n"+rows+"\\bottomrule\\end{tabular}\\end{table}\n")

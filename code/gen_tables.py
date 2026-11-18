import sys,os,json,pickle,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,pandas as pd; from common import *
R=os.path.join(BASE_DIR,"results")+os.sep
O=os.path.join(BASE_DIR,"paper_final","tables")+os.sep
os.makedirs(O,exist_ok=True)
S=pd.read_csv(R+"summary_test.csv").set_index('model'); cost=pd.read_csv(R+"cost.csv").set_index('model'); sig=pd.read_csv(R+"significance.csv")
nm={'LR':'Logistic Regression','SVM':'Linear SVM','RF':'Random Forest','ET':'Extra Trees','FedAvg-LR':'FedAvg-LR (K=26)','FedAvg-LR (50% clients)':'FedAvg-LR (50\\% clients)','FedProx-LR (mu=0.1)':'FedProx-LR ($\\mu$=0.1)','FedForest-RF':'Fed-RF (ensemble)','FedForest-ET':'Fed-ET (ensemble)','VQC':'VQC','QNN':'QNN'}
def f(m,k,d=3): return f"{S.loc[m,k]:.{d}f}$\\pm${S.loc[m,k+'_ci']:.{d}f}"
def res_table(models,cap,lab,fn):
    h="\\begin{table*}[!t]\\centering\\caption{%s}\\label{%s}\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{l"+"c"*9+"}\\toprule\nModel & Acc. & Bal. Acc. & Prec. & Rec. & Macro-F1 & Wtd-F1 & ROC-AUC & PR-AUC & MCC\\\\\\midrule\n"
    b="".join(nm[m]+" & "+" & ".join(f(m,k) for k in ['acc','bacc','prec','rec','f1m','f1w','auc','prauc','mcc'])+"\\\\\n" for m in models)
    open(O+fn,'w').write(h%(cap,lab)+b+"\\bottomrule\\end{tabular}\\end{table*}\n")
res_table(['LR','SVM','RF','ET'],"Centralized models on the locked-test subjects (mean $\\pm$ 95\\% $t$-interval over 10 subject-wise partitions). Positive class: higher motor severity.","tab:central","t_central.tex")
res_table(['FedAvg-LR','FedAvg-LR (50% clients)','FedProx-LR (mu=0.1)','FedForest-RF','FedForest-ET'],"Federated models on the locked-test subjects (mean $\\pm$ 95\\% $t$-interval, 10 partitions). One client per development subject.","tab:fed","t_fed.tex")
res_table(['VQC','QNN'],"Simulated quantum models on the locked-test subjects (mean $\\pm$ 95\\% $t$-interval, 10 partitions; 800 balanced training records per run).","tab:quantum","t_quantum.tex")
# overall
ms=['LR','SVM','RF','ET','FedAvg-LR','FedProx-LR (mu=0.1)','FedForest-RF','FedForest-ET','VQC','QNN']
def c(m):
    r=cost.loc[m]; parts=[]
    if not np.isnan(r.fit_s): parts.append(f"{r.fit_s:.3g} s")
    if not np.isnan(r.comm_bytes): parts.append(f"{r.comm_bytes/1e3:.0f} kB")
    return ", ".join(parts) if parts else "--"
b="".join(f"{nm[m]} & {S.loc[m,'bacc']:.3f} & {S.loc[m,'auc']:.3f} & {S.loc[m,'mcc']:.3f} & {c(m)}\\\\\n" for m in ms)
open(O+"t_overall.tex",'w').write("\\begin{table}[!t]\\centering\\caption{Overall comparison on locked-test subjects (means over 10 partitions) with training time or total communication volume.}\\label{tab:overall}\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lcccc}\\toprule Model & Bal.Acc. & AUC & MCC & Train time / comm.\\\\\\midrule\n"+b+"\\bottomrule\\end{tabular}\\end{table}\n")
# stats
t=sig[sig.test.str.startswith('AUC')]; w=sig[sig.test.str.startswith('Wilcoxon')]
rows="".join(f"{nm[r.model]} & {S.loc[r.model,'auc']:.3f} & {r.p:.4f} & {r.p_holm:.3f}\\\\\n" for r in t.itertuples())
rows2="".join(f"{nm[r.model]} & {r.mean_diff:+.3f} & {r.p:.4f} & {r.p_holm:.3f}\\\\\n" for r in w.itertuples())
open(O+"t_stats.tex",'w').write("\\begin{table}[!t]\\centering\\caption{Statistical evaluation over 10 partitions. Top: one-sample $t$-test of ROC-AUC against 0.5. Bottom: two-sided Wilcoxon signed-rank test of Extra Trees against each model (positive difference = ET higher). Holm correction within each block.}\\label{tab:stats}\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccc}\\toprule Model & Mean AUC & $p$ & $p_{\\mathrm{Holm}}$\\\\\\midrule\n"+rows+"\\midrule Model & $\\Delta$AUC (ET$-$model) & $p$ & $p_{\\mathrm{Holm}}$\\\\\\midrule\n"+rows2+"\\bottomrule\\end{tabular}\\end{table}\n")
# leakage
A=pd.concat([pd.read_csv(R+"classical_fl_runs.csv"),pd.read_csv(R+"rowsplit_runs.csv")]); A['variant']=A['variant'].fillna('clean'); A=A[A.split=='test']
vn=[('clean','Subject-wise, clean features'),('record_wise_split','Record-wise split, clean features'),('with_total_UPDRS','Subject-wise, + total\\_UPDRS'),('with_motor_UPDRS','Subject-wise, + motor\\_UPDRS')]
rows=""
for k,l in vn:
    for m in ['LR','RF']:
        g=A[(A.variant==k)&(A.model==m)]; rows+=f"{l} & {m} & {g.acc.mean():.3f} & {g.auc.mean():.3f} & {g.mcc.mean():.3f}\\\\\n"
open(O+"t_leak.tex",'w').write("\\begin{table}[!t]\\centering\\caption{Leakage ablation on identical test records (means over 10 partitions). Only the first row is a valid estimate of generalisation to unseen patients.}\\label{tab:leak}\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llccc}\\toprule Setting & Model & Acc. & AUC & MCC\\\\\\midrule\n"+rows+"\\bottomrule\\end{tabular}\\end{table}\n")
# dataset distribution
df=load(); rows=""; ns=collections.defaultdict(list)
for s in range(10):
    D,V,T,thr=make_sets(df,s)
    for n_,x in (('Development',D),('Validation',V),('Locked test',T)): ns[n_].append((x['subject#'].nunique(),len(x),x.y.mean()))
D,V,T,thr=make_sets(df,0)
for n_,x in (('Development',D),('Validation',V),('Locked test',T)):
    a=np.array(ns[n_]); rows+=f"{n_} & {x['subject#'].nunique()} & {len(x)} & {x.y.mean():.3f} & {int(a[:,1].min())}--{int(a[:,1].max())} & {a[:,2].min():.2f}--{a[:,2].max():.2f}\\\\\n"
open(O+"t_data.tex",'w').write("\\begin{table}[!t]\\centering\\caption{Subject-wise partition. Columns 2--4 refer to partition 0 (threshold = %.2f); the last two columns give the range over the 10 partitions.}\\label{tab:data}\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccccc}\\toprule Set & Subjects & Records & Positive rate & Records (range) & Pos. rate (range)\\\\\\midrule\n"%thr+rows+"\\bottomrule\\end{tabular}\\end{table}\n")
# hyperparameters (selected values)
preds,hp=pickle.load(open(R+"preds_seed0.pkl","rb")); cnt=collections.defaultdict(collections.Counter)
import glob
print(hp)

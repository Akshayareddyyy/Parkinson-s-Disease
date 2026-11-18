import sys,os,json,pickle,glob; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR,DATA,load
import numpy as np,pandas as pd,matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix,precision_recall_fscore_support
import pennylane as qml
R=os.path.join(BASE_DIR,"results")+os.sep
F=os.path.join(BASE_DIR,"figures")+os.sep
os.makedirs(F,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'legend.fontsize':6,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
 'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight','pdf.fonttype':42})
C=dict(cen='#0072B2',fed='#009E73',q='#CC79A7',bad='#D55E00',gray='#7f7f7f')
def col(m): return C['q'] if m in('VQC','QNN') else (C['fed'] if m.startswith('Fed') else C['cen'])
def save(n): plt.savefig(F+n+'.pdf'); plt.savefig(F+n+'.png',dpi=200); plt.close()
S=pd.read_csv(R+"summary_test.csv").set_index('model')
preds,hp=pickle.load(open(R+"preds_seed0.pkl","rb")); preds.update(pickle.load(open(R+"preds_quantum_seed0.pkl","rb")))
part=json.load(open(R+"partition_seed0.json")); cs=json.load(open(R+"client_stats_seed0.json"))
# ---------- diagram helpers
def box(ax,x,y,w,h,t,c='#EAF2FB',ec='#2b4a6f',fs=6.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=c,ec=ec,lw=.7)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,wrap=True)
def arr(ax,x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,lw=.7,color='#333'))
def canvas(w,h,xl,yl): fig,ax=plt.subplots(figsize=(w,h)); ax.set_xlim(0,xl); ax.set_ylim(0,yl); ax.axis('off'); return fig,ax
def chain(ax,labels,y,x0,w,h,gap,colors=None):
    for i,l in enumerate(labels):
        x=x0+i*(w+gap); box(ax,x,y,w,h,l,c=(colors[i] if colors else '#EAF2FB'))
        if i: arr(ax,x-gap,y+h/2,x,y+h/2)
# Fig1 overall
fig,ax=canvas(7.1,2.5,14.2,5.0)
xs=[.2,3.75,7.3,10.85]; W=3.1
for x,t,c in zip(xs,["Parkinson's Telemonitoring\n42 subjects, 5,875 recordings","Target definition\nseverity = 1[motor_UPDRS\n>= dev-set median]","Input features\n19 predictors\n(age, sex, voice)","Subject-wise split\n26 dev / 7 val / 9 test"],['#F3F3F3','#FFF2CC','#FFE0D6','#EAF2FB']): box(ax,x,3.4,W,1.3,t,c,fs=6)
for i in range(3): arr(ax,xs[i]+W,4.05,xs[i+1],4.05)
ax.plot([12.4,12.4,2.0],[3.4,2.75,2.75],c='#333',lw=.7); 
bx=[(.2,"Centralized ML\nLR, LinSVM, RF, ET",'#DCEBFA'),(4.0,"Federated learning\nFedAvg / FedProx-LR\nfederated forests",'#DDF3EA'),(7.8,"Quantum ML (simulated)\nVQC, QNN (4 qubits)",'#F6E3EE')]
for x,t,c in bx: box(ax,x,1.0,3.4,1.3,t,c,fs=6); arr(ax,x+1.7,2.75,x+1.7,2.3)
box(ax,11.7,1.0,2.4,1.3,"Locked test:\nbal. acc., AUC, MCC,\nCIs, Wilcoxon",'#F3F3F3',fs=5.8)
for x in (3.6,7.4,11.2): arr(ax,x,1.65,x+.4,1.65) if x==11.2 else None
ax.text(7.1,.35,"Test subjects are never used for fitting, preprocessing, or tuning",ha='center',fontsize=6,color=C['bad'])
save('fig1_overall_architecture')
# Fig2 preprocess
fig,ax=canvas(7.1,1.5,14.2,2.6)
chain(ax,["Raw CSV\n5,875×22","Drop IDs and\nUPDRS scores\n(label only)","Median impute\n(dev only)","StandardScaler\n(dev only)","Feature set\nd = 19","Model-specific\nprojection\n(PCA-4 for QML)"],.6,.3,1.9,1.4,.4,['#F3F3F3','#FFE0D6','#EAF2FB','#EAF2FB','#EAF2FB','#F6E3EE'])
save('fig2_preprocessing')
# Fig3 partition
fig,ax=plt.subplots(figsize=(3.5,2.0)); n=pd.read_csv(os.path.join(R,"per_subject.csv"))
df=load()
cnt=df.groupby('subject#').size()
order=part['subjects']['dev']+part['subjects']['val']+part['subjects']['test']
colors=[C['cen']]*26+['#E69F00']*7+[C['bad']]*9
ax.bar(range(42),[cnt[s] for s in order],color=colors,width=.8); ax.set_xlabel("Subjects (grouped by partition, seed 0)"); ax.set_ylabel("Recordings")
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in (C['cen'],'#E69F00',C['bad'])],labels=[f"dev: 26 subj., {cnt[part['subjects']['dev']].sum()} rec.",f"val: 7 subj., {cnt[part['subjects']['val']].sum()} rec.",f"test: 9 subj., {cnt[part['subjects']['test']].sum()} rec."],loc='lower right',frameon=False,ncol=1,fontsize=5.5,bbox_to_anchor=(1,1.0))
save('fig3_partition')
# Fig4 centralized
fig,ax=canvas(3.5,2.7,7,5.4)
for i,(t,c) in enumerate((("Dev subjects (26)",'#EAF2FB'),("Pipeline: impute → scale\n→ classifier",'#EAF2FB'),("GroupKFold(5) by subject\ngrid search (bal. acc)",'#FFF2CC'),("Refit best on dev;\nreport val & locked test",'#DCEBFA'))):
    box(ax,1,4.1-i*1.35,5,1.0,t,c)
    if i: arr(ax,3.5,4.1-(i-1)*1.35,3.5,4.1-i*1.35+1.0)
save('fig4_centralized_workflow')
# Fig5 federated
fig,ax=canvas(3.5,3.0,7,6.0)
box(ax,1.6,4.7,3.8,1.0,"Server: aggregate\nweights / ensemble",'#DDF3EA',fs=6)
xs=[.2,1.9,3.6,5.3]; lbl=["Client 1","Client 2","Client 3","Client K"]
for x,l in zip(xs,lbl):
    box(ax,x,1.5,1.5,1.0,l+"\n(1 subject)",'#EAF2FB',fs=5.6)
    cx=x+.75; arr(ax,cx-.2,4.7,cx-.2,2.5); arr(ax,cx+.2,2.5,cx+.2,4.7)
ax.text(3.5,3.55,"...",ha='center',fontsize=9)
ax.text(.1,.95,"down: global model w(t)      up: local update / local forest",fontsize=5.4)
ax.text(.1,.5,"LR: weighted FedAvg / FedProx;  forests: one-shot weighted soft vote",fontsize=5.4)
ax.text(3.5,.05,"Raw recordings never leave a client",ha='center',fontsize=5.8,color=C['bad'])
save('fig5_federated_architecture')
# Fig6/7 quantum architecture block diagrams
fig,ax=canvas(7.1,1.3,14.2,2.4)
chain(ax,["x ∈ R¹⁹","scale + PCA\n→ R⁴, to [0,π]","H, RZ(xᵢ)\nZZ ring\n(feature map)","L=3 × [RY,RZ\n+ CNOT ring]","⟨Z₀⟩","p = (1−⟨Z₀⟩)/2\nBCE loss"],.5,.3,2.0,1.5,.35,['#F3F3F3','#EAF2FB','#F6E3EE','#F6E3EE','#FFF2CC','#EAF2FB'])
save('fig6_vqc_architecture')
fig,ax=canvas(7.1,1.3,14.2,2.4)
chain(ax,["x ∈ R¹⁹","scale + PCA\n→ R⁴, to [0,π]","L=3 layers:\nRY(wx+c), RZ,\nCNOT ring","⟨Z₀..Z₃⟩","Linear head\nσ(vᵀ⟨Z⟩+b)","BCE loss\n(Adam)"],.5,.3,2.0,1.5,.35,['#F3F3F3','#EAF2FB','#F6E3EE','#FFF2CC','#EAF2FB','#EAF2FB'])
save('fig7_qnn_architecture')
# Fig8 circuits from PennyLane (same code as run_quantum)
NQ=4;L=3;dev=qml.device("default.qubit",wires=NQ)
def vq(x,th):
    for i in range(NQ): qml.Hadamard(i); qml.RZ(x[i],i)
    for i in range(NQ): j=(i+1)%NQ; qml.CNOT([i,j]); qml.RZ(x[i]*x[j],j); qml.CNOT([i,j])
    for l in range(L):
        for i in range(NQ): qml.RY(th[l,i,0],i); qml.RZ(th[l,i,1],i)
        for i in range(NQ): qml.CNOT([i,(i+1)%NQ])
    return qml.expval(qml.PauliZ(0))
def qn(x,w,c,th):
    for l in range(L):
        for i in range(NQ): qml.RY(w[l,i]*x[i]+c[l,i],i); qml.RZ(th[l,i],i)
        for i in range(NQ): qml.CNOT([i,(i+1)%NQ])
    return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]
for nm,fn,args in (('fig8a_vqc_circuit',vq,(np.ones(4)*.5,np.ones((3,4,2))*.1)),('fig8b_qnn_circuit',qn,(np.ones(4)*.5,np.ones((3,4)),np.ones((3,4))*.1,np.ones((3,4))*.1))):
    f,_=qml.draw_mpl(qml.QNode(fn,dev),decimals=None,style='black_white')(*args); f.set_size_inches(7.1,1.8); f.savefig(F+nm+'.pdf',bbox_inches='tight'); f.savefig(F+nm+'.png',dpi=200,bbox_inches='tight'); plt.close(f)
# ----- results
order=['LR','SVM','RF','ET','FedAvg-LR','FedProx-LR (mu=0.1)','FedForest-RF','FedForest-ET','VQC','QNN']
lab={'FedProx-LR (mu=0.1)':'FedProx-LR','FedForest-RF':'Fed-RF','FedForest-ET':'Fed-ET'}
L_=lambda m:lab.get(m,m)
fig,ax=plt.subplots(1,2,figsize=(7.1,2.3))
for a,k,t in((ax[0],'bacc','Balanced accuracy'),(ax[1],'auc','ROC-AUC')):
    v=[S.loc[m,k] for m in order]; e=[S.loc[m,k+'_ci'] for m in order]
    a.bar(range(len(order)),v,yerr=e,color=[col(m) for m in order],capsize=2,width=.7,error_kw=dict(lw=.6)); a.axhline(.5,ls='--',c=C['gray'],lw=.7); a.set_ylim(.3,.8)
    a.set_xticks(range(len(order))); a.set_xticklabels([L_(m) for m in order],rotation=40,ha='right'); a.set_ylabel(t+" (mean ± 95% CI, 10 splits)")
ax[0].legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in (C['cen'],C['fed'],C['q'])],labels=['Centralized','Federated','Quantum (sim.)'],frameon=False,loc='upper right')
plt.tight_layout(); save('fig9_accuracy_auc_comparison')
# ROC + PR seed0
sel=['LR','ET','FedAvg-LR','FedForest-ET','VQC','QNN']
fig,ax=plt.subplots(1,2,figsize=(7.1,2.2))
for m in sel:
    y,s,_=preds[m]; fp,tp,_=roc_curve(y,s); ax[0].plot(fp,tp,label=L_(m),c=None); p,r,_=precision_recall_curve(y,s); ax[1].plot(r,p)
ax[0].plot([0,1],[0,1],'k:',lw=.7); ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate'); ax[0].set_title('ROC (locked test, partition 0)')
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_title('Precision–recall (locked test, partition 0)')
ax[0].legend(frameon=False,loc='lower right'); plt.tight_layout(); save('fig10_roc_pr_curves')
# confusion matrices
fig,ax=plt.subplots(1,6,figsize=(7.1,1.55))
for a,m in zip(ax,sel):
    y,s,_=preds[m]; th=0 if m=='SVM' else .5; cm=confusion_matrix(y,(s>th).astype(int)); a.imshow(cm,cmap='Blues'); a.set_title(L_(m),fontsize=6.5)
    for i in range(2):
        for j in range(2): a.text(j,i,cm[i,j],ha='center',va='center',fontsize=6,color='w' if (cm[i,j]-cm.min())/(cm.max()-cm.min()+1e-9)>.5 else 'k')
    a.set_xticks([0,1]);a.set_yticks([0,1]);a.set_xlabel('Pred.',fontsize=6); a.spines[:].set_visible(True)
ax[0].set_ylabel('True')
plt.tight_layout(); save('fig11_confusion_matrices')
# FL convergence
h=pd.concat([pd.read_csv(f) for f in glob.glob(R+"fl_hist_seed*.csv")]); fig,ax=plt.subplots(1,3,figsize=(7.1,2.1))
for (fr,mu),g in h.groupby(['frac','mu']):
    nm={(1.0,0.0):'FedAvg (K=26)',(0.5,0.0):'FedAvg (50% clients)',(1.0,0.1):'FedProx μ=0.1'}[(fr,mu)]; m=g.groupby('round').mean(numeric_only=True)
    ax[0].plot(m.index,m.train_loss,label=nm); ax[1].plot(m.index,m.val_auc); ax[2].plot(m.index,m.test_auc)
ax[0].set_ylabel('Train log-loss'); ax[1].set_ylabel('Validation AUC'); ax[2].set_ylabel('Locked-test AUC')
for a in ax: a.set_xlabel('Communication round')
ax[0].legend(frameon=False); plt.tight_layout(); save('fig12_federated_convergence')
# communication analysis
P=pd.read_csv(R+"participation.csv"); fig,ax=plt.subplots(1,2,figsize=(7.1,2.1))
for m,g in P.groupby('model'): ax[0].plot(g.n_clients,g.test_auc,marker='o',ms=3,label=L_(m))
ax[0].axhline(.5,ls='--',c=C['gray'],lw=.7); ax[0].set_xlabel('Participating clients'); ax[0].set_ylabel('Locked-test AUC'); ax[0].legend(frameon=False); ax[0].set_ylim(.4,.65)
Cst=pd.read_csv(R+"cost.csv").set_index('model'); mm=['FedAvg-LR (50% clients)','FedAvg-LR','FedForest-RF','FedForest-ET']
ax[1].bar(range(4),[Cst.loc[m,'comm_bytes']/1e3 for m in mm],color=C['fed']); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['FedAvg\n50%','FedAvg\n100%','Fed-RF','Fed-ET']); ax[1].set_ylabel('Total uplink+downlink (kB)')
plt.tight_layout(); save('fig13_communication_analysis')
# cost
fig,ax=plt.subplots(1,2,figsize=(7.1,2.1)); cm_=['LR','SVM','RF','ET','VQC','QNN']
ax[0].bar(range(6),[Cst.loc[m,'fit_s'] for m in cm_],color=[col(m) for m in cm_]); ax[0].set_yscale('log'); ax[0].set_xticks(range(6)); ax[0].set_xticklabels(cm_); ax[0].set_ylabel('Training wall-clock (s, log)')
ax[1].bar(range(6),[Cst.loc[m,'infer_s']*1e3 for m in cm_],color=[col(m) for m in cm_]); ax[1].set_yscale('log'); ax[1].set_xticks(range(6)); ax[1].set_xticklabels(cm_); ax[1].set_ylabel('Test inference (ms, log)')
plt.tight_layout(); save('fig14_computational_cost')
# per-class (seed 0)
fig,ax=plt.subplots(figsize=(3.5,2.2)); w=.12
for i,m in enumerate(sel):
    y,s,_=preds[m]; p,r,f,_=precision_recall_fscore_support(y,(s>.5).astype(int),zero_division=0)
    ax.bar(np.arange(2)+i*w,f,w,label=L_(m))
ax.set_xticks(np.arange(2)+2.5*w); ax.set_xticklabels(['Class 0 (milder)','Class 1 (more severe)']); ax.set_ylabel('F1 (locked test, partition 0)'); ax.legend(frameon=False,ncol=3,fontsize=5.5)
save('fig15_per_class_f1')
# per-subject
ps=pd.read_csv(R+"per_subject.csv"); g=ps[ps.model.isin(['ET','LR','FedForest-ET'])]
fig,ax=plt.subplots(figsize=(3.5,2.2)); mean_s=g.groupby(['model','subject']).acc.mean().reset_index()
data=[mean_s[mean_s.model==m].acc for m in ['ET','LR','FedForest-ET']]; ax.boxplot(data,tick_labels=['ET','LR','Fed-ET'],widths=.5); 
for i,d in enumerate(data): ax.scatter(np.random.RandomState(0).normal(i+1,.04,len(d)),d,s=4,alpha=.5,c=C['gray'])
ax.axhline(.5,ls='--',c=C['gray'],lw=.7); ax.set_ylabel('Accuracy per held-out subject\n(mean over splits)'); save('fig16_subject_level')
# leakage ablation
A=pd.concat([pd.read_csv(R+"classical_fl_runs.csv"),pd.read_csv(R+"rowsplit_runs.csv")]); A['variant']=A['variant'].fillna('clean'); A=A[A.split=='test']
vs=[('clean','Subject-wise,\nclean features'),('record_wise_split','Record-wise split,\nclean features\n(patient leakage)'),('with_total_UPDRS','+ total_UPDRS\n(target proxy)'),('with_motor_UPDRS','+ motor_UPDRS\n(target leakage)')]
fig,ax=plt.subplots(figsize=(3.5,2.5)); 
for i,m in enumerate(['LR','RF']):
    v=[A[(A.variant==k)&(A.model==m)].auc.mean() for k,_ in vs]; e=[A[(A.variant==k)&(A.model==m)].auc.std(ddof=1) for k,_ in vs]
    ax.bar(np.arange(4)+i*.38,v,.38,yerr=e,capsize=2,label=m,color=[C['cen'],C['fed']][i],error_kw=dict(lw=.6))
ax.set_xticks(np.arange(4)+.19); ax.set_xticklabels([l for _,l in vs],fontsize=5); ax.axhline(.5,ls='--',c=C['gray'],lw=.7); ax.set_ylabel('Locked-test ROC-AUC (mean ± SD, 10 splits)'); ax.legend(frameon=False,loc='upper left'); ax.set_ylim(.4,1.05)
save('fig17_leakage_ablation')
# train/val/test
Z=pd.concat([pd.read_csv(R+"classical_fl_runs.csv"),pd.read_csv(R+"quantum_runs.csv")]); Z=Z[Z.variant.isna()] if 'variant' in Z else Z
fig,ax=plt.subplots(figsize=(3.5,2.3)); mo=['LR','RF','ET','FedAvg-LR','FedForest-ET','VQC','QNN']
for i,sp in enumerate(['train','val','test']):
    ax.bar(np.arange(len(mo))+i*.27,[Z[(Z.model==m)&(Z.split==sp)].bacc.mean() for m in mo],.27,label=sp)
ax.set_xticks(np.arange(len(mo))+.27); ax.set_xticklabels([L_(m) for m in mo],rotation=35,ha='right'); ax.axhline(.5,ls='--',c=C['gray'],lw=.7); ax.set_ylabel('Balanced accuracy (mean, 10 splits)'); ax.legend(frameon=False,ncol=3)
save('fig18_train_val_test')
# quantum training curves
q=pd.read_csv(R+"quantum_curves.csv").groupby(['model','epoch']).mean(numeric_only=True).reset_index(); fig,ax=plt.subplots(1,2,figsize=(7.1,2.0))
for m,g in q.groupby('model'): ax[0].plot(g.epoch,g.train_loss,label=m+' train'); ax[0].plot(g.epoch,g.val_loss,'--',label=m+' val'); ax[1].plot(g.epoch,g.train_auc,label=m+' train'); ax[1].plot(g.epoch,g.val_auc,'--',label=m+' val')
ax[0].set_ylabel('BCE loss'); ax[1].set_ylabel('ROC-AUC'); [a.set_xlabel('Epoch') for a in ax]; ax[0].legend(frameon=False,ncol=2,fontsize=5.5); plt.tight_layout(); save('fig19_quantum_training')
print("ok")

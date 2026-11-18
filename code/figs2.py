import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR
import glob,matplotlib; matplotlib.use("Agg")
import numpy as np,pandas as pd,matplotlib.pyplot as plt
R=os.path.join(BASE_DIR,"results")+os.sep
F=os.path.join(BASE_DIR,"figures")+os.sep
os.makedirs(F,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'legend.fontsize':6,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
C=dict(cen='#0072B2',fed='#009E73',q='#CC79A7',bad='#D55E00',gray='#7f7f7f')
h=pd.concat([pd.read_csv(f) for f in glob.glob(R+"fl_hist_seed*.csv")]); P=pd.read_csv(R+"participation.csv"); Cst=pd.read_csv(R+"cost.csv").set_index('model')
fig,ax=plt.subplots(1,4,figsize=(7.1,1.6))
for (fr,mu),g in h.groupby(['frac','mu']):
    nm={(1.0,0.0):'FedAvg (K=26)',(0.5,0.0):'FedAvg (50% clients)',(1.0,0.1):'FedProx'}[(fr,mu)]; m=g.groupby('round').mean(numeric_only=True)
    ax[0].plot(m.index,m.train_loss,label=nm); ax[1].plot(m.index,m.test_auc)
ax[0].set_xlabel('Round');ax[0].set_ylabel('Training log-loss');ax[0].legend(frameon=False,fontsize=5.3); ax[0].set_title('(a)',loc='left')
ax[1].set_xlabel('Round');ax[1].set_ylabel('Test ROC-AUC'); ax[1].set_title('(b)',loc='left')
for m,g in P.groupby('model'): ax[2].plot(g.n_clients,g.test_auc,marker='o',ms=3,label={'FedForest-ET':'Fed-ET','FedForest-RF':'Fed-RF'}[m])
ax[2].axhline(.5,ls='--',c=C['gray'],lw=.7); ax[2].set_ylim(.4,.65); ax[2].set_xlabel('Contributing clients'); ax[2].set_ylabel('Test ROC-AUC'); ax[2].legend(frameon=False,fontsize=5.5); ax[2].set_title('(c)',loc='left')
mm=['FedAvg-LR (50% clients)','FedAvg-LR','FedForest-RF','FedForest-ET']; ax[3].bar(range(4),[Cst.loc[m,'comm_bytes']/1e3 for m in mm],color=C['fed'])
ax[3].set_xticks(range(4)); ax[3].set_xticklabels(['FedAvg\n50%','FedAvg\n100%','Fed-RF','Fed-ET'],fontsize=5.5); ax[3].set_ylabel('Total traffic (kB)'); ax[3].set_title('(d)',loc='left')
plt.tight_layout(); plt.savefig(F+'figF_federated.pdf'); plt.savefig(F+'figF_federated.png',dpi=200); plt.close()
q=pd.read_csv(R+"quantum_curves.csv").groupby(['model','epoch']).mean(numeric_only=True).reset_index()
fig,ax=plt.subplots(1,3,figsize=(7.1,1.6)); cm_=['LR','SVM','RF','ET','VQC','QNN']; col=[C['cen']]*4+[C['q']]*2
ax[0].bar(range(6),[Cst.loc[m,'fit_s'] for m in cm_],color=col); ax[0].set_yscale('log'); ax[0].set_xticks(range(6)); ax[0].set_xticklabels(cm_,fontsize=6); ax[0].set_ylabel('Training time (s, log)'); ax[0].set_title('(a)',loc='left')
ax[1].bar(range(6),[Cst.loc[m,'infer_s']*1e3 for m in cm_],color=col); ax[1].set_yscale('log'); ax[1].set_xticks(range(6)); ax[1].set_xticklabels(cm_,fontsize=6); ax[1].set_ylabel('Test inference (ms, log)'); ax[1].set_title('(b)',loc='left')
for m,g in q.groupby('model'): ax[2].plot(g.epoch,g.train_loss,label=m+' train'); ax[2].plot(g.epoch,g.val_loss,'--',label=m+' val')
ax[2].set_xlabel('Epoch'); ax[2].set_ylabel('BCE loss'); ax[2].legend(frameon=False,ncol=2,fontsize=5.3); ax[2].set_title('(c)',loc='left')
plt.tight_layout(); plt.savefig(F+'figQ_cost_training.pdf'); plt.savefig(F+'figQ_cost_training.png',dpi=200); plt.close()
print('ok')

import pickle
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
preds,hp=pickle.load(open(R+"preds_seed0.pkl","rb")); preds.update(pickle.load(open(R+"preds_quantum_seed0.pkl","rb")))
fig,ax=plt.subplots(1,5,figsize=(7.1,1.65),gridspec_kw=dict(width_ratios=[1.25,1.25,.8,.8,.8]))
lab={'FedForest-ET':'Fed-ET'}
for m in ['LR','ET','FedAvg-LR','FedForest-ET','VQC','QNN']:
    y,sc,_=preds[m]; fp,tp,_=roc_curve(y,sc); ax[0].plot(fp,tp,label=lab.get(m,m),lw=1); p_,r_,_=precision_recall_curve(y,sc); ax[1].plot(r_,p_,lw=1)
ax[0].plot([0,1],[0,1],'k:',lw=.6); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR'); ax[0].set_title('(a) ROC',loc='left'); ax[0].legend(frameon=False,fontsize=4.8,loc='lower right')
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_title('(b) PR',loc='left')
for a_,m,t in zip(ax[2:],['ET','FedAvg-LR','VQC'],['(c) ET','(d) FedAvg-LR','(e) VQC']):
    y,sc,_=preds[m]; cm=confusion_matrix(y,(sc>.5).astype(int)); a_.imshow(cm,cmap='Blues'); a_.set_title(t,loc='left',fontsize=6.5)
    for i in range(2):
        for j in range(2): a_.text(j,i,cm[i,j],ha='center',va='center',fontsize=6,color='w' if (cm[i,j]-cm.min())/(cm.max()-cm.min()+1e-9)>.5 else 'k')
    a_.set_xticks([0,1]);a_.set_yticks([0,1]);a_.set_xlabel('Predicted',fontsize=6); a_.spines[:].set_visible(True)
ax[2].set_ylabel('True')
plt.tight_layout(); plt.savefig(F+'figR_roc_pr_cm.pdf'); plt.savefig(F+'figR_roc_pr_cm.png',dpi=200); plt.close()

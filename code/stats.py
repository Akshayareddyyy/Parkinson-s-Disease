import sys,os,json,pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR
import numpy as np,pandas as pd; from scipy import stats
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
R=os.path.join(BASE_DIR,"results")+os.sep
a=pd.concat([pd.read_csv(R+"classical_fl_runs.csv"),pd.read_csv(R+"quantum_runs.csv"),pd.read_csv(R+"rowsplit_runs.csv")])
a['variant']=a['variant'].fillna('clean')
M=['acc','bacc','prec','rec','f1m','f1w','auc','prauc','mcc']
def ci(x):
    x=np.asarray(x,float);m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return m,h
out=[]
clean=a[(a.variant=='clean')&(a.split=='test')]
for mod,g in clean.groupby('model'):
    r=dict(model=mod)
    for k in M: m,h=ci(g[k]); r[k]=m; r[k+'_ci']=h
    out.append(r)
T=pd.DataFrame(out); T.to_csv(R+"summary_test.csv",index=False)
# other summaries
for v in ['with_motor_UPDRS','with_total_UPDRS','record_wise_split']:
    g=a[(a.variant==v)&(a.split=='test')]; g.groupby('model')[M].mean().to_csv(R+f"summary_{v}.csv")
for sp in ['train','val']:
    g=a[(a.variant=='clean')&(a.split==sp)]; g.groupby('model')[M].mean().to_csv(R+f"summary_{sp}.csv")
cost=a[(a.variant=='clean')&(a.split=='test')].groupby('model')[['fit_s','infer_s','comm_bytes','n_params','rounds']].mean(); cost.to_csv(R+"cost.csv")
# significance: one-sample t vs chance AUC=.5 and paired Wilcoxon vs ET (best centralized)
pv=clean.pivot(index='seed',columns='model',values='auc'); res=[]
for mod in pv.columns:
    t,p=stats.ttest_1samp(pv[mod],0.5); res.append(dict(model=mod,test='AUC vs 0.5 (t)',stat=t,p=p))
base='ET'
for mod in pv.columns:
    if mod==base: continue
    d=pv[base]-pv[mod]
    w,p=stats.wilcoxon(d) if (d!=0).any() else (0,1.0); res.append(dict(model=mod,test=f'Wilcoxon {base} vs model (AUC)',stat=w,p=p,mean_diff=d.mean()))
R_=pd.DataFrame(res)
def holm(p):
    o=np.argsort(p);m=len(p);adj=np.empty(m);run=0
    for i,j in enumerate(o): run=max(run,(m-i)*p[j]);adj[j]=min(1,run)
    return adj
for t_ in R_.test.unique():
    ix=R_.test==t_; R_.loc[ix,'p_holm']=holm(R_.loc[ix,'p'].values)
R_.to_csv(R+"significance.csv",index=False)
# leakage: paired difference clean vs leaky RF
# subject-cluster bootstrap CI on seed-0 test AUC
preds,hp=pickle.load(open(R+"preds_seed0.pkl","rb")); pq=pickle.load(open(R+"preds_quantum_seed0.pkl","rb")); preds.update(pq)
rng=np.random.RandomState(0); bs=[]
for mod,(y,s,sid) in preds.items():
    subs=np.unique(sid); v=[]
    for _ in range(1000):
        pick=rng.choice(subs,len(subs)); ix=np.concatenate([np.where(sid==k)[0] for k in pick])
        if len(set(y[ix]))>1: v.append(roc_auc_score(y[ix],s[ix]))
    bs.append(dict(model=mod,auc=roc_auc_score(y,s),lo=np.percentile(v,2.5),hi=np.percentile(v,97.5)))
pd.DataFrame(bs).to_csv(R+"bootstrap_seed0.csv",index=False)
print(T[['model','acc','acc_ci','bacc','auc','auc_ci','mcc']].round(3).to_string()); print(R_.round(4).to_string()); print(pd.DataFrame(bs).round(3)); print(cost.round(3))
# part. analysis
p=a[a.split=='participation'].groupby(['model','n_clients'])[['val_auc','test_auc']].mean(); p.to_csv(R+"participation.csv"); print(p.round(3))

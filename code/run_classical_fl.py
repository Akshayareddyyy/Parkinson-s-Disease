import sys,os,time,json,pickle,warnings
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
NSEED=10; R=os.path.join(BASE_DIR,"results")+os.sep
os.makedirs(R,exist_ok=True)
df=load()
def zs(a): return (a-a.mean())/(a.std()+1e-12)
def mk(name,seed):
    if name=='LR': return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),{'logisticregression__C':[0.01,0.1,1,10]}
    if name=='SVM': return make_pipeline(StandardScaler(),LinearSVC(max_iter=20000)),{'linearsvc__C':[0.001,0.01,0.1,1]}
    c=RandomForestClassifier if name=='RF' else ExtraTreesClassifier
    return c(n_estimators=200,random_state=seed,n_jobs=-1),{'max_depth':[4,8,None],'min_samples_leaf':[1,10,40]}
def score(m,X):
    return m.predict_proba(X)[:,1] if hasattr(m,'predict_proba') else m.decision_function(X)
rows=[];preds={};hp={};persubj=[]
for seed in range(NSEED):
    D,V,T,thr=make_sets(df,seed)
    XD,XV,XT=D[FEATS].values,V[FEATS].values,T[FEATS].values
    for name in ['LR','SVM','RF','ET']:
        est,grid=mk(name,seed)
        gs=GridSearchCV(est,grid,cv=GroupKFold(5),scoring='balanced_accuracy',n_jobs=1).fit(XD,D.y,groups=D['subject#'])
        t=time.time(); m=gs.best_estimator_.fit(XD,D.y); ft=time.time()-t
        t=time.time(); sT=score(m,XT); it=time.time()-t
        hp[(seed,name)]={k:str(v) for k,v in gs.best_params_.items()}
        for sp,X,Y in (('train',XD,D.y),('val',XV,V.y),('test',XT,T.y)):
            s=score(m,X); rows.append(dict(seed=seed,model=name,split=sp,fit_s=ft,infer_s=it,**metrics(Y.values,(s>(0.5 if hasattr(m,'predict_proba') else 0)).astype(int),s)))
        if seed==0:
            preds[name]=(T.y.values,sT,T['subject#'].values)
        for sid,g in T.groupby('subject#'):
            p=(sT[g.index]>(0.5 if hasattr(m,'predict_proba') else 0)).astype(int)
            persubj.append(dict(seed=seed,model=name,subject=sid,n=len(g),pos_rate=g.y.mean(),acc=(p==g.y.values).mean()))
    # ---------------- leakage ablation (primary partition only + all seeds, RF & LR) ------
    for tag,cols in (('with_motor_UPDRS',LEAKY),('with_total_UPDRS',FEATS+['total_UPDRS'])):
        for name,m in (('LR',make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),
                       ('RF',RandomForestClassifier(200,random_state=seed,n_jobs=-1))):
            m.fit(D[cols].values,D.y); s=m.predict_proba(T[cols].values)[:,1]
            rows.append(dict(seed=seed,model=name,split='test',variant=tag,**metrics(T.y.values,(s>.5).astype(int),s)))
    # ---------------- FEDERATED ----------------
    sc=StandardScaler().fit(XD); ZD,ZV,ZT=sc.transform(XD),sc.transform(XV),sc.transform(XT)   # global scaler from dev (stat sharing; see paper)
    clients=[(g.index.values) for _,g in D.groupby('subject#')]; K=len(clients); d=ZD.shape[1]
    sig=lambda z:1/(1+np.exp(-np.clip(z,-40,40)))
    def fedavg(frac,rounds=40,E_=5,lr=0.1,mu=0.0,seed_=seed):
        rng=np.random.RandomState(seed_);w=np.zeros(d);b=0.;hist=[]
        for r in range(1,rounds+1):
            sel=rng.choice(K,max(1,int(round(frac*K))),replace=False);ws=[];bs=[];ns=[]
            for k in sel:
                i=clients[k];X=ZD[i];y=D.y.values[i];lw,lb=w.copy(),b
                for _ in range(E_):
                    e=sig(X@lw+lb)-y; lw-=lr*(X.T@e/len(i)+mu*(lw-w)); lb-=lr*e.mean()
                ws.append(lw);bs.append(lb);ns.append(len(i))
            a=np.array(ns)/sum(ns); w=sum(ai*wi for ai,wi in zip(a,ws)); b=sum(ai*bi for ai,bi in zip(a,bs))
            sV=sig(ZV@w+b); sT_=sig(ZT@w+b); ll=-np.mean(D.y.values*np.log(sig(ZD@w+b)+1e-9)+(1-D.y.values)*np.log(1-sig(ZD@w+b)+1e-9))
            hist.append(dict(seed=seed,round=r,frac=frac,mu=mu,train_loss=ll,val_auc=roc_auc_score(V.y,sV),val_acc=((sV>.5)==V.y).mean(),
                 test_auc=roc_auc_score(T.y,sT_),comm_bytes=2*len(sel)*(d+1)*8*r))
        return w,b,hist
    allh=[]
    for frac,mu in ((1.0,0.0),(0.5,0.0),(1.0,0.1)):
        w,b,h=fedavg(frac,mu=mu); allh+=h
        sT_=sig(ZT@w+b);sV_=sig(ZV@w+b);sD_=sig(ZD@w+b)
        nm='FedAvg-LR' if (frac==1 and mu==0) else ('FedAvg-LR (50% clients)' if mu==0 else 'FedProx-LR (mu=0.1)')
        for sp,Z,Y,s in (('train',ZD,D.y,sD_),('val',ZV,V.y,sV_),('test',ZT,T.y,sT_)):
            rows.append(dict(seed=seed,model=nm,split=sp,rounds=40,comm_bytes=h[-1]['comm_bytes'],**metrics(Y.values,(s>.5).astype(int),s)))
        if seed==0 and nm=='FedAvg-LR': preds[nm]=(T.y.values,sT_,T['subject#'].values)
    pd.DataFrame(allh).to_csv(R+f"fl_hist_seed{seed}.csv",index=False)
    # federated tree ensembles: one-shot, each client trains local forest, server forms weighted soft-vote ensemble
    for name,cls in (('RF',RandomForestClassifier),('ET',ExtraTreesClassifier)):
        depth=eval(hp[(seed,name)]['max_depth']); leaf=int(hp[(seed,name)]['min_samples_leaf'])
        loc=[];ns=[];t=time.time()
        for k in clients:
            i=k
            if len(set(D.y.values[i]))<1: continue
            loc.append(cls(n_estimators=20,max_depth=depth,min_samples_leaf=min(leaf,max(1,len(i)//10)),random_state=seed,n_jobs=-1).fit(XD[i],D.y.values[i])); ns.append(len(i))
        ft=time.time()-t; a=np.array(ns)/sum(ns)
        def ens(X,sel=None):
            idx=range(len(loc)) if sel is None else sel; ww=np.array([ns[j] for j in idx],float); ww/=ww.sum(); out=0
            for wj,j in zip(ww,idx):
                pr=loc[j].predict_proba(X); out=out+wj*(pr[:,list(loc[j].classes_).index(1)] if 1 in loc[j].classes_ else 0*pr[:,0])
            return out
        sz=sum(len(pickle.dumps(m)) for m in loc)
        for sp,X,Y in (('train',XD,D.y),('val',XV,V.y),('test',XT,T.y)):
            s=ens(X); rows.append(dict(seed=seed,model=f'FedForest-{name}',split=sp,fit_s=ft,comm_bytes=sz,rounds=1,**metrics(Y.values,(s>.5).astype(int),s)))
        if seed==0: preds[f'FedForest-{name}']=(T.y.values,ens(XT),T['subject#'].values)
        # participation analysis: random m clients
        rng=np.random.RandomState(seed)
        for m_ in (2,5,10,15,20,len(loc)):
            sel=rng.choice(len(loc),m_,replace=False); s=ens(XV,sel); sT_=ens(XT,sel)
            rows.append(dict(seed=seed,model=f'FedForest-{name}',split='participation',n_clients=m_,val_auc=roc_auc_score(V.y,s),test_auc=roc_auc_score(T.y,sT_),val_bacc=balanced_accuracy_score(V.y,(s>.5).astype(int))))
        for sid,g in T.groupby('subject#'):
            p=(ens(XT)[g.index]>.5).astype(int)
            persubj.append(dict(seed=seed,model=f'FedForest-{name}',subject=sid,n=len(g),pos_rate=g.y.mean(),acc=(p==g.y.values).mean()))
    if seed==0:
        cs=[dict(subject=int(g['subject#'].iloc[0]),n=len(g),pos_rate=float(g.y.mean())) for _,g in D.groupby('subject#')]
        json.dump(cs,open(R+"client_stats_seed0.json","w"))
        json.dump(dict(subjects={k:[int(x) for x in v] for k,v in zip(('dev','val','test'),partition(df,0))},thr=float(thr)),open(R+"partition_seed0.json","w"))
    print("seed",seed,"done",flush=True)
pd.DataFrame(rows).to_csv(R+"classical_fl_runs.csv",index=False); pd.DataFrame(persubj).to_csv(R+"per_subject.csv",index=False)
pickle.dump((preds,hp),open(R+"preds_seed0.pkl","wb"))

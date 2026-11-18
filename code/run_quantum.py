import sys,os,time,json,warnings
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import pennylane as qml; from pennylane import numpy as pnp
from common import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
NQ=4; L=3; R=os.path.join(BASE_DIR,"results")+os.sep
os.makedirs(R,exist_ok=True)
# Defaults aligned with final published paper: 10 seeds, full development set (NTRAIN=0), 40 epochs
NSEED=int(sys.argv[1]) if len(sys.argv)>1 else 10
NTRAIN=int(sys.argv[2]) if len(sys.argv)>2 else 0  # 0 indicates all development records (3,517-3,697)
EPOCHS=int(sys.argv[3]) if len(sys.argv)>3 else 40
print(f"Quantum run config: NSEED={NSEED}, NTRAIN={'Full Dev Set (no subsampling)' if NTRAIN==0 else NTRAIN}, EPOCHS={EPOCHS}")
dev=qml.device("default.qubit",wires=NQ)
@qml.qnode(dev,interface="autograd",diff_method="backprop")
def vqc(x,th):                      # x:(B,4)  th:(L,4,2)
    for i in range(NQ): qml.Hadamard(i); qml.RZ(x[:,i],i)             # angle map
    for i in range(NQ):                                               # ZZ entangling map (ring)
        j=(i+1)%NQ; qml.CNOT([i,j]); qml.RZ(x[:,i]*x[:,j],j); qml.CNOT([i,j])
    for l in range(L):
        for i in range(NQ): qml.RY(th[l,i,0],i); qml.RZ(th[l,i,1],i)
        for i in range(NQ): qml.CNOT([i,(i+1)%NQ])
    return qml.expval(qml.PauliZ(0))
@qml.qnode(dev,interface="autograd",diff_method="backprop")
def qnn(x,w,c,th):                  # data re-uploading: w,c,th:(L,4)
    for l in range(L):
        for i in range(NQ): qml.RY(w[l,i]*x[:,i]+c[l,i],i); qml.RZ(th[l,i],i)
        for i in range(NQ): qml.CNOT([i,(i+1)%NQ])
    return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]
sig=lambda z:1/(1+pnp.exp(-z))
def bce(p,y): p=pnp.clip(p,1e-6,1-1e-6); return -pnp.mean(y*pnp.log(p)+(1-y)*pnp.log(1-p))
def fwd(kind,P,X):
    if kind=='VQC': return (1-vqc(X,P[0]))/2                 # P(y=1)=(1-<Z0>)/2
    ex=pnp.stack(qnn(X,P[0],P[1],P[2]),axis=1); return sig(ex@P[3]+P[4])
rows=[];curves=[];preds={}
df=load()
for seed in range(NSEED):
    D,V,T,thr=make_sets(df,seed)
    rng=np.random.RandomState(seed); idx=np.arange(len(D)) if NTRAIN==0 else np.concatenate([rng.choice(np.where(D.y==c)[0],NTRAIN//2,replace=False) for c in (0,1)])
    mon=rng.choice(len(idx),min(600,len(idx)),replace=False)   # random monitoring subset (both classes)
    Dq=D.iloc[idx]; sc=StandardScaler().fit(Dq[FEATS]); pca=PCA(NQ,random_state=seed).fit(sc.transform(Dq[FEATS]))
    pr=lambda A:pca.transform(sc.transform(A[FEATS])); mm=MinMaxScaler((0,np.pi)).fit(pr(Dq))
    f=lambda A:np.clip(mm.transform(pr(A)),0,np.pi)
    XD,XV,XT=f(Dq),f(V),f(T); yD=Dq.y.values.astype(float)
    for kind in ('VQC','QNN'):
        r=np.random.RandomState(seed+100)
        if kind=='VQC': P=[pnp.array(r.uniform(0,2*np.pi,(L,NQ,2)),requires_grad=True)]
        else: P=[pnp.array(np.ones((L,NQ)),requires_grad=True),pnp.array(r.uniform(0,2*np.pi,(L,NQ)),requires_grad=True),
                 pnp.array(r.uniform(0,2*np.pi,(L,NQ)),requires_grad=True),pnp.array(r.normal(0,.5,NQ),requires_grad=True),pnp.array(0.,requires_grad=True)]
        npar=sum(int(np.prod(p.shape)) for p in P); opt=qml.AdamOptimizer(0.05); t0=time.time()
        for ep in range(1,EPOCHS+1):
            perm=r.permutation(len(XD))
            for b in range(0,len(perm),128):
                j=perm[b:b+128]; P=opt.step(lambda *q:bce(fwd(kind,list(q),XD[j]),yD[j]),*P) if kind=='QNN' else [opt.step(lambda q:bce(fwd(kind,[q],XD[j]),yD[j]),P[0])]
            if ep%2==0 or ep==1:
                pv=np.array(fwd(kind,P,XV)); pt=np.array(fwd(kind,P,XD[mon]))
                curves.append(dict(seed=seed,model=kind,epoch=ep,train_loss=float(bce(fwd(kind,P,XD[mon]),yD[mon])),
                    val_loss=float(bce(pv,V.y.values.astype(float))),val_auc=roc_auc_score(V.y,pv),train_auc=roc_auc_score(yD[mon],pt)))
        tt=time.time()-t0; t1=time.time(); sT=np.array(fwd(kind,P,XT)); it=time.time()-t1
        for sp,X,Y in (('train',XD,yD),('val',XV,V.y.values),('test',XT,T.y.values)):
            s=np.array(fwd(kind,P,X)); rows.append(dict(seed=seed,model=kind,split=sp,fit_s=tt,infer_s=it,n_params=npar,n_train=len(XD),**metrics(Y.astype(int),(s>.5).astype(int),s)))
        if seed==0: preds[kind]=(T.y.values,sT,T['subject#'].values)
        print(seed,kind,"test auc %.3f"%roc_auc_score(T.y,sT),"time %.0fs"%tt,flush=True)
pd.DataFrame(rows).to_csv(R+"quantum_runs.csv",index=False); pd.DataFrame(curves).to_csv(R+"quantum_curves.csv",index=False)
import pickle; pickle.dump(preds,open(R+"preds_quantum_seed0.pkl","wb"))

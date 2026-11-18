import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
R=os.path.join(BASE_DIR,"results")+os.sep
os.makedirs(R,exist_ok=True)
df=load(); rows=[]
for seed in range(10):
    tr,te=train_test_split(df,test_size=0.25,random_state=seed)   # random record-wise split (NOT subject-wise)
    thr=tr.motor_UPDRS.median(); ytr=(tr.motor_UPDRS>=thr).astype(int); yte=(te.motor_UPDRS>=thr).astype(int)
    for n,m in (('LR',make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),('RF',RandomForestClassifier(200,random_state=seed,n_jobs=-1))):
        m.fit(tr[FEATS],ytr); s=m.predict_proba(te[FEATS])[:,1]
        rows.append(dict(seed=seed,model=n,variant='record_wise_split',split='test',**metrics(yte.values,(s>.5).astype(int),s)))
pd.DataFrame(rows).to_csv(os.path.join(R,"rowsplit_runs.csv"),index=False)
print(pd.DataFrame(rows).groupby('model')[['acc','auc','mcc']].mean().round(3))

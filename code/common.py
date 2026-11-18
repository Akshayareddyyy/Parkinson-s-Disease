import os, numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,recall_score,
    f1_score,roc_auc_score,average_precision_score,matthews_corrcoef)

# Set environment variables for Windows portability to prevent non-writable ~/.matplotlib warnings and loky subprocess hangs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.tmp_mpl'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

DATA = os.path.join(BASE_DIR, "data", "parkinsons_updrs.csv")
VOICE=['Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)',
 'Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']
FEATS=['age','sex','test_time']+VOICE          # motor_UPDRS / total_UPDRS excluded
LEAKY=FEATS+['motor_UPDRS']
def load(): return pd.read_csv(DATA)
def partition(df,seed,n_dev=26,n_val=7,n_test=9):
    """Subject-wise 26/7/9 partition; random subjects, no subject in two sets."""
    rng=np.random.RandomState(seed); s=rng.permutation(np.sort(df['subject#'].unique()))
    return s[:n_dev],s[n_dev:n_dev+n_val],s[n_dev+n_val:]
def make_sets(df,seed):
    dv,va,te=partition(df,seed)
    D,V,T=[df[df['subject#'].isin(x)].reset_index(drop=True) for x in (dv,va,te)]
    thr=D['motor_UPDRS'].median()            # development-only threshold
    for x in (D,V,T): x['y']=(x['motor_UPDRS']>=thr).astype(int)
    return D,V,T,thr
def metrics(y,p,s):
    """p: hard predictions, s: continuous score for positive class"""
    out=dict(acc=accuracy_score(y,p),bacc=balanced_accuracy_score(y,p),
      prec=precision_score(y,p,zero_division=0),rec=recall_score(y,p,zero_division=0),
      f1m=f1_score(y,p,average='macro'),f1w=f1_score(y,p,average='weighted'),
      mcc=matthews_corrcoef(y,p))
    out['auc']=roc_auc_score(y,s) if len(set(y))>1 else np.nan
    out['prauc']=average_precision_score(y,s) if len(set(y))>1 else np.nan
    return out

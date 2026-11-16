import numpy as np,lightgbm as lgb,json
from sklearn.metrics import roc_auc_score
X=np.load('X.npy',mmap_mode='r'); Y=np.load('Y.npy'); F=np.load('fold.npy'); K=np.load('keep.npy')
def feats(a):
    d=np.diff(a,axis=1); f=np.abs(np.fft.rfft(a,axis=1))
    bands=[f[:,lo:hi].sum(1) for lo,hi in [(1,10),(10,30),(30,100),(100,300)]]
    q=np.percentile(a,[5,25,50,75,95],axis=1)
    return np.concatenate([a.mean(1),a.std(1),a.min(1),a.max(1),d.std(1),np.abs(d).max(1)]+bands+list(q),axis=1)
idx=np.where(K)[0]; Fe=np.concatenate([feats(np.asarray(X[idx[i:i+1000]])) for i in range(0,len(idx),1000)])
fo=F[idx]; Yk=Y[idx]; tr=fo<=8; te=fo==10; va=fo==9
cls=['NORM','MI','STTC','CD','HYP']; aucs=[]
for k in range(5):
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,verbose=-1,n_jobs=2).fit(Fe[tr],Yk[tr,k])
    aucs.append(roc_auc_score(Yk[te,k],m.predict_proba(Fe[te])[:,1]))
print({c:round(a,4) for c,a in zip(cls,aucs)},'macro',round(float(np.mean(aucs)),4))

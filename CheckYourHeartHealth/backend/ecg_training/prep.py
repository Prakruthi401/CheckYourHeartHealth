import numpy as np,pandas as pd,ast
db=pd.read_csv('ptbxl_database.csv',index_col='ecg_id')
sc=pd.read_csv('scp.csv',index_col=0); sc=sc[sc.diagnostic==1.0]
cls=['NORM','MI','STTC','CD','HYP']
def agg(d):
    s=set()
    for k in ast.literal_eval(d):
        if k in sc.index: s.add(sc.loc[k,'diagnostic_class'])
    return s
S=db.scp_codes.apply(agg)
Y=np.array([[c in s for c in cls] for s in S],dtype=np.float32)
keep=Y.sum(1)>0
fn=db.filename_lr.values
X=np.lib.format.open_memmap('X.npy',mode='w+',dtype=np.float32,shape=(len(db),1000,12))
for i,f in enumerate(fn):
    X[i]=np.fromfile('dat/'+f.split('/')[-1]+'.dat',dtype='<i2').reshape(1000,12)/1000.0
X.flush()
np.save('Y.npy',Y); np.save('fold.npy',db.strat_fold.values); np.save('keep.npy',keep)
print(len(db),keep.sum(),Y[keep].mean(0),pd.Series(db.strat_fold[keep]).value_counts().sort_index().tolist())

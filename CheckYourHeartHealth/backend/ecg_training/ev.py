import torch,numpy as np,json
from cnn import *
m=Net(); m.load_state_dict(torch.load('best.pt')); 
_,_,a=evalm(m,te); P,yt,_=evalm(m,te)
from sklearn.metrics import f1_score
cls=['NORM','MI','STTC','CD','HYP']
print({c:round(x,4) for c,x in zip(cls,a)},'macro',round(float(np.mean(a)),4))
json.dump({'test_records':int(len(te)),'per_class_auc':{c:round(x,4) for c,x in zip(cls,a)},'macro_auc':round(float(np.mean(a)),4)},open('cnn_metrics.json','w'),indent=1)

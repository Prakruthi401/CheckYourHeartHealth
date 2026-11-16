import numpy as np,torch,torch.nn as nn,time,os,sys,json
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
X=np.load('X.npy',mmap_mode='r'); Y=np.load('Y.npy'); F=np.load('fold.npy'); K=np.load('keep.npy')
tr=np.where(K&(F<=8))[0]; va=np.where(K&(F==9))[0]; te=np.where(K&(F==10))[0]
mu=np.load('mu.npy') if os.path.exists('mu.npy') else None
if mu is None:
    s=np.asarray(X[tr[:3000]]); mu=np.stack([s.mean((0,1)),s.std((0,1))]); np.save('mu.npy',mu)
def get(idx):
    return torch.from_numpy(((np.asarray(X[np.sort(idx)])-mu[0])/mu[1]).astype(np.float32)).permute(0,2,1), torch.from_numpy(Y[np.sort(idx)])
class Blk(nn.Module):
    def __init__(s,i,o,st):
        super().__init__(); s.c1=nn.Conv1d(i,o,7,st,3,bias=False); s.b1=nn.BatchNorm1d(o); s.c2=nn.Conv1d(o,o,7,1,3,bias=False); s.b2=nn.BatchNorm1d(o)
        s.sk=nn.Sequential(nn.Conv1d(i,o,1,st,bias=False),nn.BatchNorm1d(o))
    def forward(s,x):
        y=torch.relu(s.b1(s.c1(x))); y=s.b2(s.c2(y)); return torch.relu(y+s.sk(x))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(12,32,15,2,7,bias=False),nn.BatchNorm1d(32),nn.ReLU(),Blk(32,32,1),Blk(32,64,2),Blk(64,96,2),Blk(96,128,2),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,5))
    def forward(s,x): return s.f(x)
def evalm(m,idx):
    m.eval(); P=[]
    with torch.no_grad():
        for i in range(0,len(idx),512):
            x,_=get(idx[i:i+512]); P.append(torch.sigmoid(m(x)).numpy())
    P=np.concatenate(P); yt=Y[np.sort(idx)]
    return P,yt,[roc_auc_score(yt[:,k],P[:,k]) for k in range(5)]
if __name__=='__main__':
    EP=int(sys.argv[1]); budget=float(sys.argv[2])
    m=Net(); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-2)
    st={'ep':0,'best':0}
    if os.path.exists('ck.pt'):
        c=torch.load('ck.pt',weights_only=False); m.load_state_dict(c['m']); opt.load_state_dict(c['o']); st=c['st']
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=EP*(len(tr)//64+1),last_epoch=-1) if st['ep']==0 else None
    if sched is None:
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=EP*(len(tr)//64+1)); sched.load_state_dict(c['s'])
    t0=time.time(); lossf=nn.BCEWithLogitsLoss()
    while st['ep']<EP:
        m.train(); perm=np.random.RandomState(st['ep']).permutation(tr); st.setdefault('pos',0)
        while st['pos']<len(perm):
            b=perm[st['pos']:st['pos']+64]; x,y=get(b)
            sh=np.random.randint(-50,50); x=torch.roll(x,sh,2)
            opt.zero_grad(); l=lossf(m(x),y); l.backward(); opt.step(); sched.step(); st['pos']+=64
            if time.time()-t0>budget:
                torch.save({'m':m.state_dict(),'o':opt.state_dict(),'st':st,'s':sched.state_dict()},'ck.pt'); print('paused ep',st['ep'],'pos',st['pos'],'loss',l.item()); sys.exit()
        _,_,a=evalm(m,va); mean=np.mean(a); print('epoch',st['ep'],'val macro AUC',round(mean,4))
        if mean>st['best']: st['best']=mean; torch.save(m.state_dict(),'best.pt')
        st['ep']+=1; st['pos']=0
        torch.save({'m':m.state_dict(),'o':opt.state_dict(),'st':st,'s':sched.state_dict()},'ck.pt')
        if time.time()-t0>budget: print('paused after epoch'); sys.exit()
    print('done')

import sys, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, time
sys.path.insert(0,'/home/claude/work'); from vec import score, DT
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
STR=10
D={nm:pickle.load(open(nm+'N.pkl','rb')) for nm in ['tr','va','t200']}
def seqs(d,cols):
    X=d['X'][cols].values; out=[]
    idx=pd.Series(np.arange(len(d['sid']))).groupby(d['sid'],sort=False).agg(['first','last'])
    for s,(a,b) in idx.iterrows():
        ii=np.arange(a,b+1)[STR-1::STR]            # крок 50 мс; значення на кроці i бачить лише минуле
        if len(ii)==0: continue
        m=(d['t'][ii]>1.0)&(d['t'][ii]<=d['tD'][ii])&(~d['drop'][ii])
        out.append((s,ii,X[ii],d['y'][ii],m.astype('float32')))
    return out
def pad(S,mu,sd):
    L=max(len(x[1]) for x in S); n=len(S); k=S[0][2].shape[1]
    X=np.zeros((n,L,k),'float32'); Y=np.zeros((n,L),'float32'); M=np.zeros((n,L),'float32')
    for i,(s,ii,x,y,m) in enumerate(S): X[i,:len(ii)]=(x-mu)/sd; Y[i,:len(ii)]=y; M[i,:len(ii)]=m
    return torch.tensor(X),torch.tensor(Y),torch.tensor(M)
class Net(nn.Module):
    def __init__(s,k,h):
        super().__init__(); s.g=nn.GRU(k,h,batch_first=True); s.o=nn.Linear(h,1)
    def forward(s,x): return s.o(s.g(x)[0]).squeeze(-1)
def full_p(d,S,logit,use_lm=True):
    p=np.zeros(len(d['sid'])); ends=pd.Series(np.arange(len(d['sid']))).groupby(d['sid'],sort=False).last()
    for i,(s,ii,x,y,m) in enumerate(S):
        v=1/(1+np.exp(-logit[i,:len(ii)])); v=np.where(d['t'][ii]>1.0,v,0)
        a=ii[0]; end=ends[s]+1                      # не виходимо за межі свого розряду
        seg=np.repeat(v,STR)[:end-a]; p[a:a+len(seg)]=seg          # тримаємо до наступного оновлення
    return np.maximum(p,d['plm']) if use_lm else p
def run(name,cols,h=24,epochs=40,lr=3e-3,wd=1e-4,seed=0):
    torch.manual_seed(seed)
    St=seqs(D['tr'],cols); mu=np.concatenate([x[2] for x in St]).mean(0); sd=np.concatenate([x[2] for x in St]).std(0)+1e-6
    T={nm:(seqs(D[nm],cols)) for nm in D}; P={nm:pad(T[nm],mu,sd) for nm in D}
    net=Net(len(cols),h); opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    Xt,Yt,Mt=P['tr']; n=len(Xt); best=(-1,None,None)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,64):
            j=perm[i:i+64]; lo=net(Xt[j]); loss=(nn.functional.binary_cross_entropy_with_logits(lo,Yt[j],reduction='none')*Mt[j]).sum()/Mt[j].sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        sched.step()
        if ep%4==3 or ep==epochs-1:
            net.eval()
            with torch.no_grad(): lv=net(P['va'][0]).numpy()
            v=score(full_p(D['va'],T['va'],lv),D['va']['F'],D['va']['sh'],verbose=False)[:,1:].mean()
            if v>best[0]: best=(v,ep,{k:w.clone() for k,w in net.state_dict().items()})
            print(f'  {name} ep{ep+1} loss {loss.item():.4f} val {v:.3f}',flush=True)
    net.load_state_dict(best[2]); net.eval(); res={}
    with torch.no_grad():
        for nm in ['va','t200']:
            lo=net(P[nm][0]).numpy(); p=full_p(D[nm],T[nm],lo); tab=score(p,D[nm]['F'],D[nm]['sh'],verbose=False); res[nm]=(tab,p)
    print(f'{name:28s} best ep {best[1]+1}: VAL {res["va"][0][:,1:].mean():.3f}  TEAM200 {res["t200"][0][:,1:].mean():.3f}',flush=True)
    return net,mu,sd,res
ALL=list(D['tr']['X'].columns)
RAWISH=[c for c in ALL if c not in ('risk','lam','trem')]
if __name__=='__main__':
    for nm in ['va','t200']:
        tab=score(D[nm]['p_phys'],D[nm]['F'],D[nm]['sh'],verbose=False); print('physics v2',nm,'%.3f'%tab[:,1:].mean())
    out={}
    out['hybrid']=run('GRU hybrid (all feats)',ALL)
    out['nophys']=run('GRU without formula',RAWISH)
    pickle.dump({k:(v[0].state_dict(),v[1],v[2],{n:(t,) for n,(t,p) in v[3].items()}) for k,v in out.items()},open('nn_out.pkl','wb'))
    for k,v in out.items(): np.save(f'p_{k}_va.npy',v[3]['va'][1]); np.save(f'p_{k}_t200.npy',v[3]['t200'][1])

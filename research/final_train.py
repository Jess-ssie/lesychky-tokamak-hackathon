"""Крок 2: навчання моделей на seed 105+106, добір епохи на seed 102, тест на 101 / 202 / наборі команди (200)."""
import sys, pickle, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0,'/home/claude/work'); from shotstats import *
torch.set_num_threads(2); STR=10
TRAIN=['s105','s106']; SEL='s102'; TESTS=['s101','va','t200']
ALL=None
def seqs(d,cols):
    X=d['X'][cols].values; out=[]
    idx=pd.Series(np.arange(len(d['sid']))).groupby(d['sid'],sort=False).agg(['first','last'])
    for s,(a,b) in idx.iterrows():
        ii=np.arange(a,b+1)[STR-1::STR]
        if len(ii)==0: continue
        t=d['t'][ii]; m=(t>1.0)&(t<=d['tD'][ii])&(~d['drop'][ii]); r=d['sh'].loc[s]
        out.append(dict(s=s,ii=ii,end=b,x=X[ii].copy(),y=d['y'][ii].copy(),m=m.astype('float32'),t=t.copy(),
                        dis=bool(r.disrupted),bad=bool(r.disrupted and (r.spontaneous or np.isnan(r.t_tq_s))),tq=float(r.t_tq_s) if r.disrupted else np.nan))
    return out
def pad(S,mu,sd,ri):
    L=max(len(q['ii']) for q in S); n=len(S); k=S[0]['x'].shape[1]
    X=np.zeros((n,L,k),'float32'); Y=np.zeros((n,L),'float32'); M=np.zeros((n,L),'float32'); B=np.full((n,L),-9.0,'float32')
    W=np.zeros((5,n,L),'float32'); lab=np.zeros(n,'float32'); use=np.ones(n,'float32')
    for i,q in enumerate(S):
        l=len(q['ii']); X[i,:l]=(q['x']-mu)/sd; Y[i,:l]=q['y']; M[i,:l]=q['m']
        p=np.clip(1-np.exp(-q['x'][:,ri]),1e-4,1-1e-4); B[i,:l]=np.log(p/(1-p))
        if q['bad']: use[i]=0
        elif q['dis']:
            lab[i]=1
            for k_,Ld in enumerate([0.1,0.3,0.5,1.0,2.0]): W[k_,i,:l]=(q['t']>1.0)&(q['t']<=q['tq']-Ld)
        else: W[:,i,:l]=(q['t']>1.0)
    return [torch.tensor(a) for a in (X,Y,M,B,W,lab,use)]
class Net(nn.Module):
    def __init__(s,k,h):
        super().__init__(); s.g=nn.GRU(k,h,batch_first=True); s.o=nn.Linear(h,1)
    def forward(s,x): return s.o(s.g(x)[0]).squeeze(-1)
def full_p(d,S,logit):
    p=np.zeros(len(d['sid']))
    for i,q in enumerate(S):
        v=1/(1+np.exp(-logit[i,:len(q['ii'])])); v=np.where(q['t']>1.0,v,0); a=q['ii'][0]
        seg=np.repeat(v,STR)[:q['end']+1-a]; p[a:a+len(seg)]=seg
    return np.maximum(p,d['plm'])
def train(name,cols,mode,h,epochs,lr,wd,seed,EV,TR):
    """mode: 'bce' (покрокова мітка) або 'resid' (поправка до логіта фізичної моделі, втрата по розряду)"""
    torch.manual_seed(seed); np.random.seed(seed); ci=[ALL.index(c) for c in cols]; ri=ALL.index('risk')
    cat=np.concatenate([q['x'][:,ci] for q in TR]); mu=cat.mean(0); sd=cat.std(0)+1e-6; del cat
    sub=lambda S:[dict(q,x=q['x']) for q in S]
    def P(S):
        S2=[dict(q) for q in S]; out=pad(S2,np.zeros(len(ALL)),np.ones(len(ALL)),ri)   # B з сирого risk
        X=out[0][:,:,ci]; X=(X-torch.tensor(mu,dtype=torch.float32))/torch.tensor(sd,dtype=torch.float32)
        lens=torch.tensor([len(q['ii']) for q in S]); mask=(torch.arange(X.shape[1])[None,:]<lens[:,None]).float()[:,:,None]
        return [X*mask]+out[1:]
    Xt,Yt,Mt,Bt,Wt,labt,uset=P(TR); PE={nm:P(EV[nm][1]) for nm in EV}
    net=Net(len(cols),h); resid=(mode=='resid')
    if resid: nn.init.zeros_(net.o.weight); nn.init.zeros_(net.o.bias)
    alpha=nn.Parameter(torch.tensor(1.0)); beta=nn.Parameter(torch.tensor(0.0))
    opt=torch.optim.AdamW(list(net.parameters())+([alpha,beta] if resid else []),lr=lr,weight_decay=wd); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    def logits(X,B): return (alpha*B+beta if resid else 0)+net(X)
    def evalset(nm):
        net.eval()
        with torch.no_grad(): lo=logits(PE[nm][0],PE[nm][3]).numpy()
        d,S=EV[nm]; return per_shot(full_p(d,S,lo),d)
    n=len(TR); best=(-1,-1,None); tau=0.3; t0=time.time()
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,64):
            j=perm[i:i+64]; lo=logits(Xt[j],Bt[j])
            if mode=='bce': loss=(nn.functional.binary_cross_entropy_with_logits(lo,Yt[j],reduction='none')*Mt[j]).sum()/Mt[j].sum()
            else:
                loss=0
                for k in range(5):
                    w=Wt[k][j]; has=(w.sum(1)>0).float()*uset[j]
                    pooled=tau*torch.logsumexp(lo/tau+torch.log(w+1e-30),dim=1)
                    loss=loss+(nn.functional.binary_cross_entropy_with_logits(pooled.clamp(-30,30),labt[j],reduction='none')*has).sum()/has.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        sched.step()
        if ep%3==2 or ep==epochs-1:
            v=table(*evalset(SEL)).mean()
            if v>best[0]: best=(v,ep,({k:w.clone() for k,w in net.state_dict().items()},alpha.item(),beta.item()))
            print(f'   {name} ep{ep+1} sel {v:.3f} ({time.time()-t0:.0f}s)',flush=True)
    net.load_state_dict(best[2][0])
    with torch.no_grad(): alpha.fill_(best[2][1]); beta.fill_(best[2][2])
    res={nm:evalset(nm) for nm in EV}
    print(f'{name:32s} best ep {best[1]+1}: '+'  '.join(f'{nm} {table(*res[nm]).mean():.3f}' for nm in EV),flush=True)
    return res,dict(state={k:v.numpy().tolist() for k,v in net.state_dict().items()},mu=mu.tolist(),sd=sd.tolist(),cols=cols,alpha=alpha.item(),beta=beta.item(),mode=mode,h=h)
if __name__=='__main__':
    TR=[]; EV={}
    for nm in TRAIN:
        d=pickle.load(open(nm+'N.pkl','rb')); ALL=list(d['X'].columns); TR+=seqs(d,ALL); del d
    for nm in [SEL]+TESTS:
        d=pickle.load(open(nm+'N.pkl','rb')); S=seqs(d,ALL); d.pop('X'); EV[nm]=(d,S)
    print('train shots',len(TR),'| eval',{nm:len(EV[nm][1]) for nm in EV},flush=True)
    RAWISH=[c for c in ALL if c not in ('risk','lam','trem')]
    R={}; M={}
    R['Фізична формула']={nm:per_shot(EV[nm][0]['p_phys'],EV[nm][0]) for nm in EV}
    print('physics: '+'  '.join(f'{nm} {table(*R["Фізична формула"][nm]).mean():.3f}' for nm in EV),flush=True)
    pickle.dump(R,open('final_R.pkl','wb'))
    for name,cols,mode,h,ep,lr,wd in [('GRU-поправка до формули',ALL,'resid',16,30,2e-3,1e-3),('GRU без формули',RAWISH,'bce',32,36,3e-3,1e-4),('GRU з ознаками формули',ALL,'bce',32,36,3e-3,1e-4)]:
        R[name],M[name]=train(name,cols,mode,h,ep,lr,wd,0,EV,TR)
        pickle.dump(R,open('final_R.pkl','wb')); pickle.dump(M,open('final_M.pkl','wb'))
    print('TRAINDONE',flush=True)

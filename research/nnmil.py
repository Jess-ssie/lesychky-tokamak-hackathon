import sys; sys.path.insert(0,'/home/claude/work')
from nntrain import *
LEADS_T=[0.1,0.3,0.5,1.0,2.0]
def run_mil(name,cols,h=16,epochs=30,lr=2e-3,wd=1e-3,seed=0,tau=0.3,resid=True):
    torch.manual_seed(seed)
    St=seqs(D['tr'],cols); mu=np.concatenate([x[2] for x in St]).mean(0); sd=np.concatenate([x[2] for x in St]).std(0)+1e-6
    T={nm:seqs(D[nm],cols) for nm in D}; P={nm:pad(T[nm],mu,sd) for nm in D}
    def base_logit(nm):   # логіт фізичної моделі (без locked mode), на тій самій сітці
        d=D[nm]; S=T[nm]; L=P[nm][0].shape[1]; B=np.full((len(S),L),-9.0,'float32')
        r=d['X']['risk'].values
        for i,(s,ii,x,y,m) in enumerate(S):
            p=np.clip(1-np.exp(-r[ii]),1e-4,1-1e-4); B[i,:len(ii)]=np.log(p/(1-p))
        return torch.tensor(B)
    Bl={nm:base_logit(nm) for nm in D}
    def masks(nm):
        d=D[nm]; S=T[nm]; L=P[nm][0].shape[1]; sh=d['sh']
        W=np.zeros((len(LEADS_T),len(S),L),'float32'); lab=np.zeros(len(S),'float32'); use=np.ones(len(S),'float32')
        for i,(s,ii,x,y,m) in enumerate(S):
            r=sh.loc[s]; t=d['t'][ii]
            if r.disrupted and (r.spontaneous or np.isnan(r.t_tq_s)): use[i]=0; continue
            if r.disrupted:
                lab[i]=1
                for k,Ld in enumerate(LEADS_T): W[k,i,:len(ii)]=(t>1.0)&(t<=r.t_tq_s-Ld)
            else:
                for k in range(len(LEADS_T)): W[k,i,:len(ii)]=(t>1.0)
        return torch.tensor(W),torch.tensor(lab),torch.tensor(use)
    Wt,labt,uset=masks('tr')
    net=Net(len(cols),h); 
    if resid: nn.init.zeros_(net.o.weight); nn.init.zeros_(net.o.bias)
    alpha=nn.Parameter(torch.tensor(1.0)); beta=nn.Parameter(torch.tensor(0.0))
    opt=torch.optim.AdamW(list(net.parameters())+[alpha,beta],lr=lr,weight_decay=wd); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    def logits(nm,idx=None):
        X=P[nm][0] if idx is None else P[nm][0][idx]; B=Bl[nm] if idx is None else Bl[nm][idx]
        return (alpha*B+beta if resid else 0)+net(X)
    def evalp(nm):
        net.eval()
        with torch.no_grad(): lo=logits(nm).numpy()
        return full_p(D[nm],T[nm],lo)
    v0=score(evalp('va'),D['va']['F'],D['va']['sh'],verbose=False)[:,1:].mean(); print(f'  {name} init val {v0:.3f}',flush=True)
    n=len(St); best=(v0,-1,{k:w.clone() for k,w in net.state_dict().items()},alpha.item(),beta.item())
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n); tot=0
        for i in range(0,n,64):
            j=perm[i:i+64]; lo=logits('tr',j); loss=0
            for k in range(len(LEADS_T)):
                w=Wt[k][j]; has=(w.sum(1)>0).float()*uset[j]
                pooled=tau*torch.logsumexp(lo/tau+torch.log(w+1e-30),dim=1)
                l=nn.functional.binary_cross_entropy_with_logits(pooled.clamp(-30,30),labt[j],reduction='none')
                loss=loss+(l*has).sum()/has.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); tot+=loss.item()
        sched.step()
        if ep%3==2 or ep==epochs-1:
            v=score(evalp('va'),D['va']['F'],D['va']['sh'],verbose=False)[:,1:].mean()
            if v>best[0]: best=(v,ep,{k:w.clone() for k,w in net.state_dict().items()},alpha.item(),beta.item())
            print(f'  {name} ep{ep+1} loss {tot:.3f} val {v:.3f} alpha {alpha.item():.2f}',flush=True)
    net.load_state_dict(best[2]); 
    with torch.no_grad(): alpha.fill_(best[3]); beta.fill_(best[4])
    res={nm:score(evalp(nm),D[nm]['F'],D[nm]['sh'],verbose=False) for nm in ['va','t200']}
    print(f'{name:30s} best ep {best[1]+1}: VAL {res["va"][:,1:].mean():.3f}  TEAM200 {res["t200"][:,1:].mean():.3f}',flush=True)
    return net,mu,sd,best,res
if __name__=='__main__':
    r1=run_mil('MIL residual on physics',ALL)
    r2=run_mil('MIL GRU alone (no formula)',RAWISH,resid=False,h=24,lr=3e-3)
    pickle.dump({'resid':(r1[0].state_dict(),r1[1],r1[2],r1[3][3],r1[3][4],r1[4]),'alone':(r2[0].state_dict(),r2[1],r2[2],r2[4])},open('mil_out.pkl','wb'))

import numpy as np, pandas as pd, pickle
DT=0.005
sig=lambda x:1/(1+np.exp(-np.clip(x,-40,40)))
P0=dict(k_ne=0.9621,k_rad=1.0552,q_a=2.2555,q_b=0.1515)
def lam_fn(fgw,bN,q,fr):
    rg=sig((fgw-1)/.08); rb=sig((bN/2.8-.85)/.06); rq=sig((2.2/q-.9)/.05); rr=sig((fr-.8)/.08)
    rl=5e-4*(1+5*rg+3*rb+2*rq)
    return np.maximum(1-(1-rg**3)*(1-rb**3)*(1-rq**3)*(1-rr**3)*(1-rl),0.005+np.minimum(rl,0.01))
def gema(x,sid,tau):
    return pd.Series(x).groupby(sid,sort=False).transform(lambda s:s.ewm(alpha=DT/tau,adjust=False).mean()).values
def gdiff(x,sid):
    d=pd.Series(x).groupby(sid,sort=False).diff().fillna(0).values; return d
def base(F,P=P0):
    sid=F['sid']; t=F['t']; ip=F['ip']; ipc=np.maximum(ip,.05)
    G={}
    G['q']=P['q_a']*F['btip']*(1+P['q_b']*F['v']**2)
    G['fgw']=np.minimum(P['k_ne']*F['fgw_lag'],1.3*F['fgw_ch']+0.02*np.pi*.9025/ipc)
    G['frad']=P['k_rad']*F['frad_m']
    pk=pd.Series(ip).groupby(sid,sort=False).cummax().values
    G['gate']=(ip>0.24)&(ip>=0.9*pk)
    G['lam']=np.where(G['gate'],lam_fn(G['fgw'],F['bN'],G['q'],G['frad']),0.0)
    dip=gdiff(ip,sid)/DT; G['dip']=dip
    flat=(ip>0.24)&(dip<=1e-7)&(t>0.02)
    tf=pd.Series(np.where(flat,t-DT,np.nan)).groupby(sid,sort=False).transform(lambda s:s.ffill().where(s.ffill().notna()).groupby(lambda i:0).transform('first') if False else s.ffill())
    # first flat time per shot, causal: nan before it happens
    first=pd.Series(np.where(flat,t-DT,np.nan)).groupby(sid,sort=False).transform(lambda s:s.where(s.notna().cumsum()==1).ffill())
    G['dur']=(first/0.15).values
    trem=np.where(np.isnan(G['dur']),1.0,np.maximum(0.82*G['dur']-t,0))
    rd=G['gate']&(dip<-1e-6)&(ip<pk)
    trem=np.where(rd,np.minimum(trem,(ip-0.9*pk)/np.maximum(-dip,1e-9)),trem)
    G['trem']=np.clip(trem,0.05,5.0)
    lm=gema(F['saddle'],sid,0.02); G['plm']=np.where(ip>0.24,sig((lm-0.011)/0.001),0)
    return G
LEADS=[0.03,0.05,0.1,0.2,0.3,0.5,1.0,2.0]; FPRS=[0.0,0.02,0.05,0.1,0.15,0.2,0.3]
def score(p,F,sh,verbose=True,leads=LEADS):
    sid=F['sid']; t=F['t']
    idx=pd.Series(np.arange(len(sid))).groupby(sid,sort=False).agg(['first','last'])
    safe=[];dis=[]
    for s,(a,b) in idx.iterrows():
        r=sh.loc[s]
        if not r.disrupted: safe.append(p[a:b+1].max())
        elif not r.spontaneous and not r.wall_contact and not np.isnan(r.t_tq_s):
            pp=np.maximum.accumulate(p[a:b+1]); tt=t[a:b+1]
            dis.append([pp[max(np.searchsorted(tt,r.t_tq_s-L+1e-9,side='right')-1,0)] if tt[0]<=r.t_tq_s-L else -1 for L in leads])
    safe=np.sort(safe)[::-1]; dis=np.array(dis)
    tab=np.zeros((len(leads),len(FPRS)))
    for j,f in enumerate(FPRS):
        k=int(np.floor(f*len(safe)+1e-9)); thr=safe[k]
        tab[:,j]=(dis>thr).mean(0)
    if verbose:
        print(pd.DataFrame(tab,index=[f'lead {l}' for l in leads],columns=[f'FPR {f}' for f in FPRS]).round(2).to_string()); print('volume=%.3f  (safe %d, dis %d)'%(tab.mean(),len(safe),len(dis)))
    return tab
if __name__=='__main__':
    for nm in ['trF.pkl','vaF.pkl']:
        F,sh=pickle.load(open(nm,'rb')); G=base(F)
        for P_,lab in [(dict(k_ne=0.9883,k_rad=0.999,q_a=1.7238,q_b=0.38),'old weights'),(P0,'new weights')]:
            G=base(F,P_); p=np.maximum(1-np.exp(-G['lam']*G['trem']),G['plm']); print(nm,lab); score(p,F,sh)
        pickle.dump(G,open(nm.replace('F.','G.'),'wb'),protocol=4)

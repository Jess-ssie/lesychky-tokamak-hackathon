"""Оцінка по розрядах + бутстреп по розрядах."""
import numpy as np, pandas as pd
LEADS=[0.03,0.05,0.1,0.2,0.3,0.5,1.0,2.0]; FPRS=[0.02,0.05,0.1,0.15,0.2,0.3]
def per_shot(p,d):
    sid=d['sid']; t=d['t']; sh=d['sh']
    idx=pd.Series(np.arange(len(sid))).groupby(sid,sort=False).agg(['first','last'])
    safe=[];dis=[]
    for s,(a,b) in idx.iterrows():
        r=sh.loc[s]
        if not r.disrupted: safe.append(p[a:b+1].max())
        elif not r.spontaneous and not r.wall_contact and not np.isnan(r.t_tq_s):
            pp=np.maximum.accumulate(p[a:b+1]); tt=t[a:b+1]
            dis.append([pp[max(np.searchsorted(tt,r.t_tq_s-L+1e-9,side='right')-1,0)] if tt[0]<=r.t_tq_s-L else -1 for L in LEADS])
    return np.array(safe),np.array(dis)
def table(safe,dis):
    s=np.sort(safe)[::-1]; T=np.zeros((len(LEADS),len(FPRS)))
    for j,f in enumerate(FPRS): T[:,j]=(dis>s[int(np.floor(f*len(s)+1e-9))]).mean(0)
    return T
def boot(stats_list,n=400,seed=0):
    """stats_list: [(safe,dis),...] для кількох моделей на ТОМУ САМОМУ наборі; спільний ресемплінг розрядів"""
    rng=np.random.default_rng(seed); ns=len(stats_list[0][0]); nd=len(stats_list[0][1]); out=[]
    for _ in range(n):
        i=rng.integers(0,ns,ns); j=rng.integers(0,nd,nd); out.append([table(s[i],d[j]).mean() for s,d in stats_list])
    return np.array(out)

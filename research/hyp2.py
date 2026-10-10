import sys,pickle,json,numpy as np,pandas as pd
sys.argv=['x']; sys.path.insert(0,'/home/claude/work'); from v3 import *
P=json.load(open('P_new.json')); rng=np.random.default_rng(0)
rows=[]   # по одному рядку на розряд
for nm in ['s102','s105','s106']:
    F,sh=load(nm); G=base(F,P); sid=F['sid']; t=F['t']; ip=F['ip']
    lm=gema(F['saddle'],sid,0.02); det=(lm>0.013)&(ip>0.24)
    pk=pd.Series(ip).groupby(sid,sort=False).cummax().values
    rg=sig((G['fgw']-1)/.08)**3; rb=sig((F['bN']/2.8-.85)/.06)**3; rq=sig((2.2/G['q']-.9)/.05)**3; rr=sig((G['frad']-.8)/.08)**3
    lam_raw=lam_fn(G['fgw'],F['bN'],G['q'],G['frad'])
    gpk=pd.Series(F['gne']).groupby(sid,sort=False).cummax().values
    idx=pd.Series(np.arange(len(sid))).groupby(sid,sort=False).agg(['first','last'])
    for s,(a,b) in idx.iterrows():
        r=sh.loc[s]; tt=t[a:b+1]; dur=G['dur'][b] if np.isfinite(G['dur'][b]) else np.nan
        d=dict(set=nm,sid=s,sc=r.scenario,dis=bool(r.disrupted),sp=bool(r.spontaneous),dur=dur)
        if r.disrupted:
            j=a+min(np.searchsorted(tt,r.t_trigger_s-1e-6),b-a); jm=max(j-1,a)   # останній крок ДО жереба
            dd=det[a:b+1]; k=np.argmax(dd) if dd.any() else -1
            d.update(tD=r.t_trigger_s,tQ=r.t_tq_s,det_t=tt[k] if k>=0 else np.nan,ramp=bool(ip[j]<0.9*pk[j]),
                     lam=lam_raw[jm],dom=['gw','beta','q','rad'][int(np.argmax([rg[jm],rb[jm],rq[jm],rr[jm]]))],domv=max(rg[jm],rb[jm],rq[jm],rr[jm]),bN=F['bN'][jm])
        if np.isfinite(dur):
            i=a+np.searchsorted(tt,0.74*dur)
            if i<=b and not (r.disrupted and r.t_trigger_s<=t[i]): d.update(alive=True,gpk=gpk[i])
        rows.append(d)
R=pd.DataFrame(rows); R.to_pickle('hyp2_R.pkl'); D=R[R.dis]; NS=D[~D.sp]
def ci(f,df,B=1000):
    v=[f(df.iloc[rng.integers(0,len(df),len(df))]) for _ in range(B)]; return np.percentile(v,2.5),np.percentile(v,97.5)
print('розрядів',len(R),'зривів',len(D),'неспонтанних',len(NS),'спонтанних',int(D.sp.sum()))
# O1
dt=D.dropna(subset=['det_t']); f1=lambda x:(x.det_t<x.tD-1e-6).mean(); f2=lambda x:((x.tQ-x.det_t)<=0.1+1e-9).mean()
x=dt.dropna(subset=['tQ'])
print('O1 спрацювань',len(dt),'з',len(D),'| до жереба: %.4f'%f1(dt),ci(f1,dt),'| випередження<=0.1: %.4f'%f2(x),ci(f2,x),'| медіана %.3f макс %.3f'%((x.tQ-x.det_t).median(),(x.tQ-x.det_t).max()))
print('   ранні спрацювання:',dt[dt.det_t<dt.tD-1e-6][['set','sid','sc','det_t','tD']].head(8).to_dict('records'))
# O2
print('O2 зривів на спаді: %d з %d'%(D.ramp.sum(),len(D)))
# O3
print('O3 домінантна межа (неспонтанні):',NS.dom.value_counts().to_dict(),'| частка бета %.4f'%(NS.dom=='beta').mean(),'| макс betaN у момент жереба %.2f'%D.bN.max())
# O4
S=D[D.sp]; g1=lambda x:(x.lam<0.02).mean(); g2=lambda x:(x.lam>0.05).mean()
print('O4 спонтанних',len(S),'| lam<0.02: %.3f'%g1(S),ci(g1,S),'| lam>0.05: %.3f'%g2(S),ci(g2,S)); print('   квантилі lam:',S.lam.quantile([.5,.9,.95,.99,1]).round(3).tolist())
print('   (для порівняння) неспонтанні lam<0.02: %.3f, медіана %.2f'%(g1(NS),NS.lam.median()))
# O6
key={'density':'gw','low_q':'q','neon':'rad'}
C=NS[NS.sc.apply(lambda s:any(k in s.split('+') for k in key))].copy()
C['ok']=[d in {key[k] for k in s.split('+') if k in key} for d,s in zip(C.dom,C.sc)]
h=lambda x:x.ok.mean(); print('O6 збіг причини: %.3f'%h(C),ci(h,C),'n',len(C)); print(pd.crosstab(C.sc.str.split('+').str[0],C.dom))
print('   розбіжності за сценарієм:',C[~C.ok].groupby('sc').size().sort_values(ascending=False).head(8).to_dict(),'| медіана внеску домінанти в розбіжних %.2f'%C[~C.ok].domv.median())
# N1
A=R[(R.alive==True)&~R.sp].copy(); A['win']=A.dis&((A.tD/A.dur)>=0.765)&((A.tD/A.dur)<=0.83); A['hi']=A.gpk>0.7
p1=A[A.hi].win.mean(); p0=A[~A.hi].win.mean()
def dff(x): return x[x.hi].win.mean()-x[~x.hi].win.mean()
def rat(x): return x[x.hi].win.mean()/max(x[~x.hi].win.mean(),1e-9)
print('N1 живих на 0.74: %d | неон>0.7: n=%d частка %.3f | решта: n=%d частка %.3f | різниця %.3f'%(len(A),A.hi.sum(),p1,(~A.hi).sum(),p0,p1-p0),ci(dff,A),'| відношення %.1f'%(p1/p0),ci(rat,A))

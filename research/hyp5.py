import sys,pickle,json,numpy as np,pandas as pd
sys.argv=['x']; sys.path.insert(0,'/home/claude/work'); from v3 import *
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold; from sklearn.metrics import roc_auc_score
P=json.load(open('P_new.json')); rows=[]
for nm in ['s102','s105','s106']:
    F,sh=load(nm); G=base(F,P); raw=pd.read_pickle(nm+'.pkl'); da=raw.dalpha_div_inner.groupby(raw.shot_id,sort=False).ffill().bfill().values.astype(float)
    sid=F['sid']; t=F['t']; gpk=pd.Series(F['gne']).groupby(sid,sort=False).cummax().values
    idx=pd.Series(np.arange(len(sid))).groupby(sid,sort=False).agg(['first','last'])
    for s,(a,b) in idx.iterrows():
        r=sh.loc[s]; dur=G['dur'][b]
        if not np.isfinite(dur) or (r.disrupted and r.spontaneous): continue
        i=a+np.searchsorted(t[a:b+1],0.74*dur)
        if i>b or (r.disrupted and r.t_trigger_s<=t[i]): continue
        d=dict(set=nm,sid=s,y=int(r.disrupted and 0.765<=r.t_trigger_s/dur<=0.83),gpk=gpk[i],gne=F['gne'][i],bol=F['bol'][i],paux=F['paux'][i],ne=F['nelag'][i],ip=F['ip'][i])
        for w,tag in [(200,'1'),(100,'05')]:
            x=da[max(a,i-w):i+1]; ref=da[max(a,i-400):i+1]; med=np.median(ref)+1e-9; rel=x/med
            up=lambda thr:int(((rel[1:]>thr)&(rel[:-1]<=thr)).sum())
            pk=rel[rel>1.8]
            d.update({f'n18_{tag}':up(1.8),f'n4_{tag}':up(4.0),f'amp_{tag}':pk.mean() if len(pk) else 1.0,f'base_{tag}':np.median(x)})
        rows.append(d)
R=pd.DataFrame(rows); R.to_pickle('hyp5_R.pkl')
BASE=['gpk','gne','bol','paux','ne','ip']; DA=[c for c in R.columns if c.startswith(('n18','n4','amp','base'))]
def oof(df,cols):
    X=df[cols].values; y=df.y.values; p=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000)).fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    return p
rng=np.random.default_rng(0)
for name,df in [('ОСНОВНА: неон > 0.7',R[R.gpk>0.7].reset_index(drop=True)),('ДОДАТКОВА: усі',R.reset_index(drop=True))]:
    y=df.y.values; pb=oof(df,BASE); pe=oof(df,BASE+DA); pd_=oof(df,DA)
    a0=roc_auc_score(y,pb); a1=roc_auc_score(y,pe); dd=[]
    for _ in range(1000):
        i=rng.integers(0,len(y),len(y))
        if y[i].min()==y[i].max(): continue
        dd.append(roc_auc_score(y[i],pe[i])-roc_auc_score(y[i],pb[i]))
    print(f'{name}: n={len(y)} подій={y.sum()} ({y.mean()*100:.1f}%) | AUC база {a0:.3f} | база+D_alpha {a1:.3f} | лише D_alpha {roc_auc_score(y,pd_):.3f} | Δ={a1-a0:+.3f} [{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}]')
    print('   середні ознаки D_alpha (подія / без події):',{c:(round(df[df.y==1][c].mean(),2),round(df[df.y==0][c].mean(),2)) for c in ['n18_1','n4_1','amp_1','base_1']})

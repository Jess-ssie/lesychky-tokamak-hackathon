"""Крок 1: перепідгонка констант фізичної моделі на навчальних seed і розрахунок ознак для всіх наборів."""
import sys, pickle, json, numpy as np, pandas as pd
sys.argv=['x']; sys.path.insert(0,'/home/claude/work')
from v3 import *
from scipy.optimize import minimize
TRAIN=['s105','s106']; ALLSETS=['s105','s106','s102','s101','va','t200']
def mle_arrays(nm):
    F,sh=pickle.load(open(nm+'F.pkl','rb')); m=F['live']&(F['t']>1.0)&((np.arange(len(F['t']))%8==0)|F['ev'])
    return {k:F[k][m] for k in ['fgw_lag','bN','btip','v','frad_m','ev']}
def refit():
    A=[mle_arrays(nm) for nm in TRAIN]; X={k:np.concatenate([a[k] for a in A]) for k in A[0]}; ev=X['ev']; wt=np.where(ev,1.0,8.0)
    def nll(p):
        l=lam_fn(p[0]*X['fgw_lag'],X['bN'],p[2]*X['btip']*(1+p[3]*X['v']**2),p[1]*X['frad_m']); return (l*wt).sum()*0.005-np.log(l[ev]).sum()
    p0=[P0['k_ne'],P0['k_rad'],P0['q_a'],P0['q_b']]
    o=minimize(nll,p0,method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-3,maxiter=600))
    print('MLE: events',int(ev.sum()),'nll old %.2f -> new %.2f'%(nll(p0),o.fun),'p',np.round(o.x,4),flush=True)
    return dict(k_ne=float(o.x[0]),k_rad=float(o.x[1]),q_a=float(o.x[2]),q_b=float(o.x[3]))
def feat(nm,P):
    F,sh=load(nm); G=base(F,P); sid=F['sid']; t=F['t']; ip=F['ip']; ipc=np.maximum(ip,.05); ngw=ipc/(np.pi*.9025)
    def slope(x,w=0.2):
        s=pd.Series(x); n=int(w/DT); return ((s-s.groupby(sid,sort=False).shift(n))/w).fillna(0).values
    p_nolm=model3(F,P=P,w_sl=.35,neon=(0.4,0.7,0.07),lm_thr=9.0)
    lm=gema(F['saddle'],sid,0.02); plm=np.where(ip>0.24,sig((lm-0.013)/0.001),0)
    dur=np.where(np.isnan(G['dur']),20,G['dur'])
    X=dict(risk=-np.log(np.clip(1-p_nolm,1e-4,1)), lam=G['lam'], trem=G['trem'], fgw=G['fgw'], q=np.clip(G['q'],0,8), frad=np.clip(G['frad'],0,4), bN=F['bN'],
      tfrac=t/dur, gate=G['gate'].astype(float), fgw_set=np.clip(F['ne_set']/ngw,0,3), dne=np.clip(slope(F['ne_set'])/ngw,-2,2), dip=np.clip(slope(ip)/ipc,-2,2), btip=np.clip(F['btip'],0,3), v=np.clip(F['v'],0,3),
      paux=F['paux']/20, dpaux=np.clip(slope(F['paux'])/20,-3,3), poh=F['poh']/5, bol=F['bol']/10, gne=F['gne'], gne_pk=pd.Series(F['gne']).groupby(sid,sort=False).cummax().values,
      nelag=F['nelag'], ip=ip/3)
    X=pd.DataFrame(X).astype('float32')
    tD=F['tD']; disr=pd.Series(sid).map(sh.disrupted).values; sp=pd.Series(sid).map(sh.spontaneous).values
    return dict(X=X,sid=sid,t=t,tD=tD,y=(disr&~sp).astype('float32'),drop=(disr&sp),p_phys=np.maximum(p_nolm,plm),plm=plm,F={'sid':sid,'t':t},sh=sh)
if __name__=='__main__':
    P=refit(); json.dump(P,open('P_new.json','w'))
    for nm in ALLSETS:
        for tag,PP in [('new',P)]+([('old',P0)] if nm not in TRAIN else []):
            d=feat(nm,PP)
            if tag=='new': pickle.dump(d,open(nm+'N.pkl','wb'),protocol=4)
            else: np.save(nm+'_pold.npy',d['p_phys'])
            print(nm,tag,d['X'].shape,'vol(no FPR0) %.3f'%score(d['p_phys'],d['F'],d['sh'],verbose=False)[:,1:].mean(),flush=True)
            del d

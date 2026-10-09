import sys; sys.argv=['x']
from v3 import *
def feat(nm,shots_csv=None):
    F,sh=load(nm); G=base(F); sid=F['sid']; t=F['t']; ip=F['ip']; ipc=np.maximum(ip,.05); ngw=ipc/(np.pi*.9025)
    def slope(x,w=0.2):
        s=pd.Series(x); n=int(w/DT); return ((s-s.groupby(sid,sort=False).shift(n))/w).fillna(0).values
    p_phys=model3(F,w_sl=.35,neon=(0.4,0.7,0.07))
    p_nolm=model3(F,w_sl=.35,neon=(0.4,0.7,0.07),lm_thr=9.0)
    lm=gema(F['saddle'],sid,0.02); plm=np.where(ip>0.24,sig((lm-0.013)/0.001),0)
    dur=np.where(np.isnan(G['dur']),20,G['dur'])
    X=dict(risk=-np.log(np.clip(1-p_nolm,1e-4,1)), lam=G['lam'], trem=G['trem'], fgw=G['fgw'], q=np.clip(G['q'],0,8), frad=np.clip(G['frad'],0,4), bN=F['bN'],
      tfrac=t/dur, gate=G['gate'].astype(float), fgw_set=np.clip(F['ne_set']/ngw,0,3), dne=np.clip(slope(F['ne_set'])/ngw,-2,2), dip=np.clip(slope(ip)/ipc,-2,2), btip=np.clip(F['btip'],0,3), v=np.clip(F['v'],0,3),
      paux=F['paux']/20, dpaux=np.clip(slope(F['paux'])/20,-3,3), poh=F['poh']/5, bol=F['bol']/10, gne=F['gne'], gne_pk=pd.Series(F['gne']).groupby(sid,sort=False).cummax().values,
      nelag=F['nelag'], ip=ip/3, wdia=F['bN']*0+pd.Series(F['bol']*0).values)
    X.pop('wdia')
    X=pd.DataFrame(X).astype('float32')
    tD=F['tD']; disr=pd.Series(sid).map(sh.disrupted).values; sp=pd.Series(sid).map(sh.spontaneous).values
    return dict(X=X,sid=sid,t=t,tD=tD,y=(disr&~sp).astype('float32'),drop=(disr&sp),p_phys=p_phys,plm=plm,F={k:F[k] for k in ['sid','t']},sh=sh)
if __name__=='__main__':
    import pickle
    for nm in ['tr','va','t200']:
        d=feat(nm); pickle.dump(d,open(nm+'N.pkl','wb'),protocol=4); print(nm,d['X'].shape,flush=True)
        if nm!='tr': print(nm,'physics v2'); score(d['p_phys'],d['F'],d['sh'])

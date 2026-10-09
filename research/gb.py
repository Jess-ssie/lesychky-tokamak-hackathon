import numpy as np, pandas as pd, pickle, sys
from vec import *
from sklearn.ensemble import HistGradientBoostingClassifier
def feats(F,G):
    sid=F['sid']; t=F['t']; ip=F['ip']; ipc=np.maximum(ip,.05); ngw=ipc/(np.pi*.9025)
    X={}
    X['lam']=G['lam']; X['trem']=G['trem']; X['fgw']=G['fgw']; X['q']=G['q']; X['frad']=G['frad']; X['bN']=F['bN']
    X['tfrac']=t/np.where(np.isnan(G['dur']),20,G['dur'])
    neset=None
    X['fgw_set']=F['nelag']*0  # placeholder replaced below
    return X
def build(nm):
    F,sh=pickle.load(open(nm+'F.pkl','rb')); G=pickle.load(open(nm+'G.pkl','rb'))
    raw=pd.read_pickle(nm+'.pkl'); ff=raw.groupby('shot_id',sort=False).ffill().bfill()
    sid=F['sid']; t=F['t']; ip=F['ip']; ipc=np.maximum(ip,.05); ngw=ipc/(np.pi*.9025)
    def slope(x,w=0.2):
        s=pd.Series(x); n=int(w/DT); return ((s-s.groupby(sid,sort=False).shift(n))/w).fillna(0).values
    ne_set=ff.prog_ne_1e20.values.astype(float)
    X=dict(lam=G['lam'],trem=G['trem'],fgw=G['fgw'],q=G['q'],frad=G['frad'],bN=F['bN'],
      tfrac=t/np.where(np.isnan(G['dur']),20,G['dur']), gate=G['gate'].astype(float),
      fgw_set=ne_set/ngw, dne=slope(ne_set)/ngw, dip=slope(ip)/ipc, btip=F['btip'], v=F['v'],
      paux=F['paux'], dpaux=slope(F['paux']), poh=F['poh'], bol=F['bol'], bol_nw=F['bol']-0.11*F['paux'],
      gne=F['gne'], dgne=slope(F['gne']), gd2=ff.gas_valve_d2_cmd.values.astype(float), rga20=gema(ff.vac_rga_mass20.values.astype(float)*1e6,sid,0.3),
      nelag=F['nelag'], ip=ip, cum=pd.Series(G['lam']*DT).groupby(sid,sort=False).cumsum().values)
    X=pd.DataFrame(X)
    tD=F['tD']; disr=pd.Series(sid).map(sh.disrupted).values; sp=pd.Series(sid).map(sh.spontaneous).values
    y=(disr&~sp).astype(int)
    keep=(t>1.0)&(t<=tD)&~(disr&sp)
    return F,G,sh,X,y,keep
if __name__=='__main__':
    Ft,Gt,sht,Xt,yt,kt=build('tr'); Fv,Gv,shv,Xv,yv,kv=build('va')
    pickle.dump((Xt,yt,kt,Xv,yv,kv),open('gbdata.pkl','wb'),protocol=4)
    sub=kt&(np.arange(len(kt))%10==0)
    for name,cols in [('all',list(Xt.columns)),('no-setpoint-forecast',['lam','trem','fgw','q','frad','bN','tfrac','gate','cum'])]:
        m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=200,l2_regularization=1.0,random_state=0)
        m.fit(Xt.loc[sub,cols],yt[sub])
        for nm,F,sh,X in [('train',Ft,sht,Xt),('VAL',Fv,shv,Xv)]:
            p=m.predict_proba(X[cols])[:,1]; p=np.where(F['t']>1.0,p,0)
            tab=score(p,F,sh,verbose=(nm=='VAL')); print(name,nm,'vol=%.3f'%tab.mean())
        if name=='all':
            from sklearn.inspection import permutation_importance
            s2=kv&(np.arange(len(kv))%40==0); r=permutation_importance(m,Xv.loc[s2,cols],yv[s2],n_repeats=3,random_state=0,scoring='roc_auc')
            print(pd.Series(r.importances_mean,index=cols).sort_values(ascending=False).round(4).to_string())
            pickle.dump(m,open('gb_all.pkl','wb'))

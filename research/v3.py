import sys; sys.argv=['x']
from v2 import *
def model3(F,P=P0,neon=(0.5,0.7,0.07),w_set=1.0,w_sl=0.5,h=0.5,lm_thr=0.013,tmin=1.0,hmax=3.0,sl_w=0.2,tw=0.775,neon_mode='add'):
    G=base(F,P); sid=F['sid']; t=F['t']; ip=F['ip']; ipc=np.maximum(ip,.05)
    live=G['gate']&(t>tmin); lam=np.where(live,G['lam'],0); trem=np.minimum(G['trem'],hmax); L=lam*trem
    n=int(sl_w/DT); sh_=lambda x:pd.Series(x).groupby(sid,sort=False).shift(n).values
    s_n=np.nan_to_num((F['ne_set']-sh_(F['ne_set']))/sl_w); s_i=np.nan_to_num((ip-sh_(ip))/sl_w)
    tfr=t/np.where(np.isnan(G['dur']),1e9,G['dur'])
    s_i=np.where(tfr>0.16,np.maximum(s_i,0),0)
    if w_set>0:   # плазма дійде до ПОТОЧНОЇ уставки (без екстраполяції нахилу)
        ne_f=F['ne_set']-(F['ne_set']-F['nelag'])*np.exp(-h/0.3)
        lf=lam_fn(P['k_ne']*ne_f/(ipc/(np.pi*.9025)),F['bN'],G['q'],G['frad'])
        L=np.maximum(L,np.where(live,w_set*lf*np.maximum(trem-h,0),0))
    if w_sl>0:
        ne_f=F['ne_set']+s_n*h-0.3*s_n-(F['ne_set']-0.3*s_n-F['nelag'])*np.exp(-h/0.3); ip_f=ipc+s_i*h
        lf=lam_fn(P['k_ne']*ne_f/(ip_f/(np.pi*.9025)),F['bN'],G['q']*ipc/ip_f,G['frad'])
        L=np.maximum(L,np.where(live,w_sl*lf*np.maximum(trem-h,0),0))
    if neon:
        pk=pd.Series(F['gne']).groupby(sid,sort=False).cummax().values
        Pw=neon[0]*sig((pk-neon[1])/neon[2]); Ln=np.where(live&(tfr<tw),-np.log(1-Pw),0)
        L=L+Ln if neon_mode=='add' else np.maximum(L,Ln)
    lm=gema(F['saddle'],sid,0.02); plm=np.where(ip>0.24,sig((lm-lm_thr)/0.001),0)
    return np.maximum(1-np.exp(-L),plm)
if __name__=='__main__':
    D={nm:load(nm) for nm in ['tr','va']}
    def ev(name,**kw):
        out=[]
        for nm in ['tr','va']:
            F,sh=D[nm]; T=score(model3(F,**kw),F,sh,verbose=False); out.append(T)
        c=lambda T:' '.join('%.2f'%x for x in T[2:,1:].mean(0))   # leads>=0.1, по колонках FPR 2..30%
        print(f'{name:30s} tr {out[0][:,1:].mean():.3f} [{c(out[0])}] | VAL {out[1][:,1:].mean():.3f} [{c(out[1])}]',flush=True)
    ev('no forecast, no neon',w_set=0,w_sl=0,neon=None)
    ev('neon only',w_set=0,w_sl=0)
    ev('set1 only',w_sl=0,neon=None)
    ev('set1 + neon',w_sl=0)
    for ws in [0.2,0.35,0.5,0.75,1.0]: ev(f'set1 + slope {ws} + neon',w_sl=ws)
    ev('set1 + slope .35 + neon max',w_sl=.35,neon_mode='max')
    ev('set1 + slope .35 + neon .4',w_sl=.35,neon=(0.4,0.7,0.07))
    ev('set1 + slope .35 + neon, h=1',w_sl=.35,h=1.0)
    ev('set1 + slope .35 + neon, hmax2',w_sl=.35,hmax=2.0)
    ev('set1 + slope .35 + neon, hmax4',w_sl=.35,hmax=4.0)

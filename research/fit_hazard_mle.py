import pandas as pd, numpy as np
from scipy.optimize import minimize
df=pd.read_pickle('m.pkl'); sh=pd.read_csv('shots.csv').set_index('shot_id')
from scipy.signal import lfilter
def lp(x,tau,dt=0.005):
    a=dt/tau; x=np.asarray(x,float)
    y,_=lfilter([a],[1,-(1-a)],x,zi=[(1-a)*x[0]]); return y
S=[]
for sid,d in df.groupby('shot_id'):
    r=sh.loc[sid]; d=d.ffill().bfill()
    ip=d.prog_ip_MA.values; bt=d.prog_bt_T.values; ipc=np.maximum(ip,.05)
    ne=lp(d.prog_ne_1e20.values,0.3)
    fgw=ne/(ipc/(np.pi*0.95**2))
    nem=d[['ifr_chord_core','ifr_chord_mid_in','ifr_chord_mid_out']].values@np.array([1,1/.863,1/.859])/3
    W=d.mag_diamag_MJ.values
    bN=100*(2/3*W*1e6/100)/(bt**2/(2*4e-7*np.pi))*0.95*bt/ipc
    paux=d.prog_p_nbi_MW.values+d.prog_p_ich_MW.values+d.prog_p_ech_MW.values
    poh=np.clip(d.mag_vloop_V.rolling(40,min_periods=1).median().values*ip,0,5)
    bol=d[[f'bolo_0{i}' for i in range(1,9)]].sum(axis=1).rolling(9,min_periods=1).median().values
    ipfrac=ip/np.maximum.accumulate(ip)
    t=d.t.values
    tD=r.t_trigger_s if r.disrupted else 1e9
    live=(ip>0.24)&(ipfrac>=0.9)&(t<=tD+1e-6)
    ev=np.zeros(len(t),bool)
    if r.disrupted: ev[np.argmin(np.abs(t-tD))]=True
    S.append(dict(sid=sid,t=t,fgw=fgw,nem=nem/ (ipc/(np.pi*0.95**2)),bN=bN,paux=paux,poh=poh,bol=bol,btip=bt/ipc,v=d.coil_pf_vert_kA.rolling(9,min_periods=1).median().values/ipc,
      shp=d.coil_pf_shape_kA.rolling(9,min_periods=1).median().values,live=live,ev=ev,ipfrac=ipfrac,ip=ip))
import pickle; pickle.dump(S,open('S.pkl','wb'))
sig=lambda x:1/(1+np.exp(-np.clip(x,-40,40)))
def lam(s,p):
    kne,krad,A,B=p[:4]
    fg=lp(s['fgw']*kne,0.1); 
    pin=s['paux']+s['poh']
    fr=lp(np.where(pin>1,krad*s['bol']/np.maximum(pin,1),0),0.1)
    q=A*s['btip']*(1+B*s['v']**2)
    rg=sig((fg-1)/.08); rb=sig((lp(s['bN'],.1)/2.8-.85)/.06); rq=sig((2.2/q-.9)/.05); rr=sig((fr-.8)/.08)
    rlm=5e-4*(1+5*rg+3*rb+2*rq)
    l=1-(1-rg**3)*(1-rb**3)*(1-rq**3)*(1-rr**3)*(1-rlm)
    return np.maximum(l,0.005+np.minimum(rlm,0.01)),(rg,rb,rq,rr,fg,fr,q)
def nll(p):
    tot=0
    for s in S:
        l,_=lam(s,p); m=s['live']
        tot+=np.sum(l[m])*0.005-np.sum(np.log(l[s['ev']&m]))
    return tot
best=None
for p0 in [[1,.25,1.56,.55],[1,.3,2,.3],[1,.2,1.2,1.0],[1,.35,2.5,.2]]:
    o=minimize(nll,p0,method='Nelder-Mead',options=dict(xatol=1e-3,fatol=1e-3,maxiter=600))
    print(p0,o.x.round(4),round(o.fun,3))
    if best is None or o.fun<best.fun: best=o
np.save('p.npy',best.x)
p=best.x
print('at events:')
for s in S:
    if s['ev'].any():
        l,(rg,rb,rq,rr,fg,fr,q)=lam(s,p); i=np.argmax(s['ev'])
        r=sh.loc[s['sid']]
        print(s['sid'],r.scenario,'sp' if r.spontaneous else '  ',f'lam={l[i]:.3f} fgw={fg[i]:.2f} q={q[i]:.2f} frad={fr[i]:.2f} bN={s["bN"][i]:.2f} live={s["live"][i]} intL={np.sum(l[s["live"]])*.005:.2f}')
print('safe:')
for s in S:
    if not s['ev'].any():
        l,(rg,rb,rq,rr,fg,fr,q)=lam(s,p); m=s['live']
        print(s['sid'],sh.loc[s['sid']].scenario,f'maxlam={l[m].max():.3f} intL={np.sum(l[m])*.005:.2f} fgw={fg[m].max():.2f} qmin={q[m][200:].min():.2f} frad={fr[m].max():.2f}')

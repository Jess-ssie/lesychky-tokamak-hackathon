import sys,json,pickle,numpy as np,pandas as pd
sys.argv=['x']; sys.path.insert(0,'/home/claude/work'); from v3 import *; from shotstats import *
P=json.load(open('P_new.json')); sets=['s101','va','t200','s102']; D={nm:load(nm) for nm in sets}; n={'s101':500,'va':600,'t200':200,'s102':1000}
iL=LEADS.index(0.3)
print('neon_p | M(сітка) | TPR@0.3с при 2% 5% 10% 15% | серед неонових зривів спіймано @5% | хибних серед безпечних неонових @5%')
for npb in [0.0,0.2,0.35,0.4,0.5,0.6,0.75,0.9]:
    Ss=[];Ds=[];Ms=[];tags=[]
    for nm in sets:
        F,sh=D[nm]; p=model3(F,P=P,w_sl=.35,neon=(npb,0.7,0.07) if npb>0 else None); d=dict(sid=F['sid'],t=F['t'],sh=sh); s_,d_=per_shot(p,d); Ss.append(s_); Ds.append(d_[:,iL]); Ms.append(n[nm]*table(s_,d_).mean())
        ok=sh[(sh.disrupted)&(~sh.spontaneous)&(~sh.t_tq_s.isna())]; tags.append(ok.scenario.str.contains('neon').values); 
        safe_sc=sh[~sh.disrupted].scenario.str.contains('neon').values
        if nm==sets[0]: SN=[safe_sc]
        else: SN.append(safe_sc)
    S=np.concatenate(Ss); Dd=np.concatenate(Ds); T=np.concatenate(tags); SNc=np.concatenate(SN); o=np.sort(S)[::-1]
    thr=lambda f:o[int(np.floor(f*len(S)+1e-9))]
    print(f'{npb:5.2f} | {sum(Ms)/sum(n.values()):.3f} | '+' '.join(f'{(Dd>thr(f)).mean()*100:5.1f}' for f in [.02,.05,.10,.15])+f' | {(Dd[T]>thr(.05)).mean()*100:5.1f}% з {T.sum()} | {(S[SNc]>thr(.05)).mean()*100:5.1f}% з {SNc.sum()}',flush=True)

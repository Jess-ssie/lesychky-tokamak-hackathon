import json,sys,numpy as np,pandas as pd
LEADS=[0.03,0.05,0.1,0.2,0.3,0.5,1.0,2.0]; FPRS=[0.0,0.02,0.05,0.1,0.15,0.2,0.3]
def tab(path):
    d=json.load(open(path)); safe=[];dis=[]
    for s in d['shots']:
        st=np.array(s['staircase']) if s['staircase'] else np.zeros((1,2))
        if not s['disrupted']: safe.append(st[:,1].max())
        elif s['verdict']!='excluded' and s['t_tq'] is not None:
            dis.append([st[st[:,0]<=s['t_tq']-L+1e-9,1].max() if (st[:,0]<=s['t_tq']-L+1e-9).any() else -1 for L in LEADS])
    safe=np.sort(safe)[::-1]; dis=np.array(dis); T=np.zeros((len(LEADS),len(FPRS)))
    for j,f in enumerate(FPRS):
        thr=safe[int(np.floor(f*len(safe)+1e-9))]; T[:,j]=(dis>thr).mean(0)
    return pd.DataFrame(T,index=[f'lead {l}' for l in LEADS],columns=[f'FPR {f}' for f in FPRS]),len(safe),len(dis),safe
for p in sys.argv[1:]:
    T,a,b,safe=tab(p); print(p,'safe',a,'dis',b); print(T.round(2).to_string()); print('volume %.3f | no FPR0: %.3f'%(T.values.mean(),T.values[:,1:].mean())); print('top safe p:',np.round(safe[:14],3))

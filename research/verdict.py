import sys,pickle,numpy as np,pandas as pd; sys.path.insert(0,'/home/claude/work'); from shotstats import *
R=pickle.load(open('final_R.pkl','rb')); TESTS=['s101','va','t200']; n={'s101':500,'va':600,'t200':200}
old={}
for nm in ['va','t200','s101']:
    d=pickle.load(open(nm+'N.pkl','rb')); d.pop('X'); old[nm]=per_shot(np.load(nm+'_pold.npy'),d)
R['Фізична, старі константи']=old
def pooled(model,sets): return sum(n[s]*table(*R[model][s]).mean() for s in sets)/sum(n[s] for s in sets)
def diff_ci(a,b,sets,B=400):
    tot=np.zeros(B)
    for s in sets:
        bb=boot([R[a][s],R[b][s]],n=B,seed=hash(s)%1000); tot+=n[s]*(bb[:,0]-bb[:,1])
    tot/=sum(n[s] for s in sets); return pooled(a,sets)-pooled(b,sets),np.percentile(tot,2.5),np.percentile(tot,97.5)
print('M по наборах:'); 
for m in R: print(f'  {m:30s}'+'  '.join(f'{s} {table(*R[m][s]).mean():.3f}' for s in TESTS)+f'  | pooled {pooled(m,TESTS):.3f}')
for s in TESTS: b=boot([R['Фізична формула'][s]]); print('  sd фізичної на',s,'%.3f'%b.std())
P='Фізична формула'
for h,a,b,sets in [('H1 GRU без формули − фізична','GRU без формули',P,TESTS),('H2 поправка − фізична','GRU-поправка до формули',P,TESTS),('H3 нові − старі константи',P,'Фізична, старі константи',['va','t200']),('H4 GRU з формулою − GRU без','GRU з ознаками формули','GRU без формули',TESTS)]:
    d,lo,hi=diff_ci(a,b,sets); print(f'{h:34s} Δ={d:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]')
T=table(*R[P]['s101']); print(pd.DataFrame(T,index=LEADS,columns=FPRS).round(2))

import sys, time, json, importlib.util, numpy as np, pandas as pd
def load_model(path):
    spec=importlib.util.spec_from_file_location('m',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def run(Model, df, sh, keep=None, **kw):
    cols=list(df.columns[2:]); out={}; st={}
    mdl=Model(**kw); tt=[]
    for sid,d in df.groupby('shot_id'):
        r=sh.loc[sid]; mdl.start_shot(dict(shot_id=int(sid),dt=0.005,duration=float(d.t.iloc[-1]) if not r.disrupted else None))
        V=d[cols].values; T=d.t.values; p=np.empty(len(T)); S=[]
        for i in range(len(T)):
            t0=time.perf_counter()
            p[i]=mdl.predict(float(T[i]),dict(zip(cols,V[i].tolist())))
            tt.append(time.perf_counter()-t0)
            if keep: S.append([mdl.state.get(k,np.nan) for k in keep])
        out[sid]=(T,p); st[sid]=np.array(S)
    return out,st,np.array(tt)
LEADS=[0.03,0.05,0.1,0.2,0.3,0.5,1.0,2.0]; FPRS=[0.0,0.02,0.05,0.1,0.15,0.2,0.3]
def score(out, sh, verbose=True):
    safe=[s for s in out if not sh.loc[s].disrupted]; dis=[s for s in out if sh.loc[s].disrupted and not sh.loc[s].spontaneous]
    smax=np.sort([out[s][1].max() for s in safe])[::-1]
    tab=np.zeros((len(LEADS),len(FPRS)))
    for j,f in enumerate(FPRS):
        k=int(np.floor(f*len(safe)+1e-9)); thr=smax[k] if k<len(smax) else -1   # alarms iff p>thr -> exactly k false alarms
        for i,L in enumerate(LEADS):
            c=0
            for s in dis:
                T,p=out[s]; tq=sh.loc[s].t_tq_s
                c+= np.any(p[T<=tq-L+1e-9]>thr)
            tab[i,j]=c/len(dis)
    if verbose:
        print(pd.DataFrame(tab,index=[f'lead {l}' for l in LEADS],columns=[f'FPR {f}' for f in FPRS]).round(2).to_string())
        print('volume(mean TPR)=%.3f'%tab.mean())
    return tab
if __name__=='__main__':
    df=pd.read_pickle('m.pkl'); sh=pd.read_csv('shots.csv').set_index('shot_id')
    M=load_model(sys.argv[1] if len(sys.argv)>1 else 'sub/model.py').Model
    out,st,tt=run(M,df,sh)
    print('step ms p50 %.3f p99 %.3f max %.3f'%(np.percentile(tt,50)*1e3,np.percentile(tt,99)*1e3,tt.max()*1e3))
    score(out,sh)
    import pickle; pickle.dump(out,open('out.pkl','wb'))

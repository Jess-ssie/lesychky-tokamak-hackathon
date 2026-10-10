import sys,pickle,numpy as np,pandas as pd,time
sys.path.insert(0,'/home/claude/work'); import harness; from shotstats import *
df=pd.read_csv('/mnt/user-data/uploads/measured_200.csv'); sh=pd.read_csv('shots_200.csv').set_index('shot_id')
M=harness.load_model('sub2/model.py').Model; cols=list(df.columns[2:]); V=df[cols].values; T=df.t.values; sid=df.shot_id.values
bounds=pd.Series(np.arange(len(sid))).groupby(sid,sort=False).agg(['first','last'])
d=dict(sid=sid,t=T,sh=sh)
def run(dead=()):
    di=[cols.index(c) for c in dead]; mdl=M(); p=np.empty(len(T))
    for s,(a,b) in bounds.iterrows():
        mdl.start_shot(dict(shot_id=int(s),dt=0.005))
        for i in range(a,b+1):
            row=V[i].tolist()
            for j in di: row[j]=float('nan')
            p[i]=mdl.predict(float(T[i]),dict(zip(cols,row)))
    return per_shot(p,d)
USED=['prog_ip_MA','prog_bt_T','prog_ne_1e20','prog_p_nbi_MW','prog_p_ich_MW','prog_p_ech_MW','ifr_chord_core','ifr_chord_mid_in','ifr_chord_mid_out','mag_diamag_MJ','coil_pf_vert_kA','mag_vloop_V','gas_valve_ne_cmd','mag_saddle_lm']+['bolo_%02d'%i for i in range(1,9)]
base=run(); m0=table(*base).mean(); print('base M %.3f'%m0,flush=True); out={'base':base}
for c in USED:
    r=run([c]); out[c]=r; b=boot([r,base],n=300); dd=b[:,0]-b[:,1]
    print(f'{c:20s} M {table(*r).mean():.3f}  Δ {table(*r).mean()-m0:+.3f}  [{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}]',flush=True)
pickle.dump(out,open('dead_out.pkl','wb')); print('DEADDONE')

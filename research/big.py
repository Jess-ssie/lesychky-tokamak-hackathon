import pandas as pd, numpy as np, pickle, sys
from scipy.optimize import minimize
DT=0.005; A=0.95; MU0=4e-7*np.pi
BOLO=["bolo_%02d"%i for i in range(1,9)]
def build(pkl, shots_csv):
    df=pd.read_pickle(pkl); sh=pd.read_csv(shots_csv).set_index('shot_id')
    g=df.groupby('shot_id',sort=False)
    ff=g.ffill().bfill(); ff['shot_id']=df.shot_id
    def ema(col,tau): return col.groupby(ff.shot_id,sort=False).transform(lambda s:s.ewm(alpha=DT/tau,adjust=False).mean()).values.astype('float64')
    ip=ff.prog_ip_MA.values.astype('float64'); bt=ff.prog_bt_T.values.astype('float64'); ipc=np.maximum(ip,.05); t=ff.t.values.astype('float64')
    ngw=ipc/(np.pi*A*A)
    F={}
    F['sid']=ff.shot_id.values; F['t']=t; F['ip']=ip; F['btip']=bt/ipc
    nelag=ema(ff.prog_ne_1e20,0.3); F['nelag']=nelag
    F['fgw_lag']=ema(pd.Series(nelag/ngw,index=ff.index),0.1)
    ch=(ff.ifr_chord_core+ff.ifr_chord_mid_in/.863+ff.ifr_chord_mid_out/.859)/3/2.73
    F['fgw_ch']=ema(pd.Series(ema(ch,0.05)/ngw,index=ff.index),0.1)
    beta=100*(2/3*ff.mag_diamag_MJ.values*1e6/100)/(bt**2/(2*MU0))*A*bt/ipc
    F['bN']=ema(pd.Series(beta,index=ff.index),0.1)
    paux=(ff.prog_p_nbi_MW+ff.prog_p_ich_MW+ff.prog_p_ech_MW).values.astype('float64')
    poh=np.clip(ema(ff.mag_vloop_V,0.1)*ip,0,5); pin=paux+poh
    bol=ff[BOLO].sum(axis=1).values.astype('float64')
    F['paux']=paux; F['poh']=poh; F['pin']=pin; F['bol']=bol
    F['frad_m']=ema(pd.Series(np.where(pin>1,bol/np.maximum(pin,1),0),index=ff.index),0.1)
    F['v']=ema(ff.coil_pf_vert_kA/ipc,0.05); F['s']=ema(ff.coil_pf_shape_kA,0.05); F['r']=ema(ff.coil_pf_radial_kA/ipc,0.05)
    for c in ['ne_bar_1e20','p_rad_MW','p_input_MW','p_ohmic_MW','w_th_MJ','in_hmode','impurity_fraction','li','xpoint_z_m','xpoint_r_m','axis_z_m','te0_keV','tau_e_s']:
        if c in ff: F[c]=ff[c].values.astype('float64')
    if 'ne_bar_1e20' in ff:
        F['fgw_true']=ema(pd.Series(F['ne_bar_1e20']/ngw,index=ff.index),0.1)
        pi_=F['p_input_MW']; F['frad_true']=ema(pd.Series(np.where(pi_>1,F['p_rad_MW']/np.maximum(pi_,1),0),index=ff.index),0.1)
        bt_=100*(2/3*F['w_th_MJ']*1e6/100)/(bt**2/(2*MU0))*A*bt/ipc; F['bN_true']=ema(pd.Series(bt_,index=ff.index),0.1)
    F['gne']=ff.gas_valve_ne_cmd.values.astype('float64'); F['saddle']=ff.mag_saddle_lm.values.astype('float64')
    pk=pd.Series(ip).groupby(F['sid'],sort=False).cummax().values
    tD=pd.Series(F['sid']).map(sh.t_trigger_s).values; tD=np.where(np.isnan(tD),1e9,tD)
    tQ=pd.Series(F['sid']).map(sh.t_tq_s).values
    F['ipfrac']=ip/np.maximum(pk,1e-9)
    F['live']=(ip>0.24)&(ip>=0.9*pk)&(t<=tD+1e-6)
    F['ev']=np.abs(t-tD)<0.0026
    F['tD']=tD; F['tQ']=tQ
    return F,sh
if __name__=='__main__':
    F,sh=build(sys.argv[1],sys.argv[2]); pickle.dump((F,sh),open(sys.argv[3],'wb'),protocol=4); print({k:v.shape for k,v in list(F.items())[:3]}, 'events',F['ev'].sum(),'live ev',(F['ev']&F['live']).sum())

import pandas as pd, numpy as np, sys, pickle, time
src,dst=sys.argv[1],sys.argv[2]
BOLO=["bolo_%02d"%i for i in range(1,9)]
use=['shot_id','t','prog_ip_MA','prog_bt_T','prog_ne_1e20','prog_p_nbi_MW','prog_p_ich_MW','prog_p_ech_MW','mag_diamag_MJ','mag_vloop_V','coil_pf_vert_kA','coil_pf_shape_kA','coil_pf_radial_kA','ifr_chord_core','ifr_chord_mid_in','ifr_chord_mid_out','gas_valve_ne_cmd','gas_valve_d2_cmd','mag_saddle_lm','dalpha_div_inner','ece_01','vac_rga_mass20','gas_flow_ne_meas','mag_flux_loop_in','mag_flux_loop_top','mag_flux_loop_out','mag_flux_loop_bot']+BOLO
truth=['ne_bar_1e20','p_rad_MW','p_input_MW','p_ohmic_MW','w_th_MJ','te0_keV','in_hmode','impurity_fraction','axis_r_m','axis_z_m','xpoint_r_m','xpoint_z_m','li','tau_e_s','neon_puff','d2_puff']
hdr=pd.read_csv(src,nrows=0).columns
cols=use+[c for c in truth if c in hdr]
t0=time.time()
df=pd.read_csv(src,usecols=cols,dtype={c:'float32' for c in cols if c not in('shot_id','in_hmode')})
print('loaded',df.shape,round(time.time()-t0))
df.to_pickle(dst)

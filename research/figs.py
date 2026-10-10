import sys,json,pickle,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/home/claude/work'); from shotstats import *
BG='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUT='#898781'; GRID='#e1e0d9'; AX='#c3c2b7'; C1='#2a78d6'; C2='#eb6834'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.facecolor':BG,'figure.facecolor':BG,'axes.edgecolor':AX,'axes.labelcolor':INK2,
 'xtick.color':MUT,'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.8,'axes.axisbelow':True,'text.color':INK})
from matplotlib.ticker import FuncFormatter
comma=FuncFormatter(lambda v,p:('%g'%round(v,2)).replace('.',',').replace('-','−'))
def fmt(v): return '0,00' if abs(v)<0.005 else f'{v:+.2f}'.replace('.',',').replace('-','−')
def fin(fig,ax,title,sub,name):
    fig.text(0.02,0.975,title,fontsize=14,fontweight='bold',va='top',color=INK); fig.text(0.02,0.915,sub,fontsize=10,va='top',color=INK2)
    fig.subplots_adjust(top=0.80); fig.savefig(f'figs/{name}.png',dpi=160); plt.close(fig)
def rj(path):
    d=json.load(open(path)); safe=[];dis=[]
    for s in d['shots']:
        st=np.array(s['staircase']) if s['staircase'] else np.zeros((1,2))
        if not s['disrupted']: safe.append(st[:,1].max())
        elif s['verdict']!='excluded' and s['t_tq'] is not None: dis.append([st[st[:,0]<=s['t_tq']-L+1e-9,1].max() if (st[:,0]<=s['t_tq']-L+1e-9).any() else -1 for L in LEADS])
    return np.array(safe),np.array(dis)
R=pickle.load(open('final_R.pkl','rb')); n={'s101':500,'va':600,'t200':200}
# 1. ours vs baseline, official run, budget 10%
ours=rj('/tmp/claude-0/r4/run.json'); base=rj('/tmp/claude-0/rb/run.json'); To=table(*ours); Tb=table(*base); j=FPRS.index(0.1)
fig,ax=plt.subplots(figsize=(8,4.8)); x=np.arange(len(LEADS))
ax.plot(x,To[:,j]*100,color=C1,lw=2,marker='o',ms=7,mec=BG,mew=2); ax.plot(x,Tb[:,j]*100,color=C2,lw=2,marker='o',ms=7,mec=BG,mew=2)
ax.text(x[-1]+.12,To[-1,j]*100,'Наша модель',color=INK,va='center',fontsize=10); ax.text(x[-1]+.12,Tb[-1,j]*100,'Бейзлайн\nорганізаторів',color=INK,va='center',fontsize=10)
for k in [2,4,6]: ax.text(x[k],To[k,j]*100+5,f'{To[k,j]*100:.0f} %',ha='center',fontsize=9,color=INK2); ax.text(x[k],Tb[k,j]*100-9,f'{Tb[k,j]*100:.0f} %',ha='center',fontsize=9,color=INK2)
ax.set_xticks(x); ax.set_xticklabels([str(l).replace('.',',') for l in LEADS]); ax.set_ylim(0,108); ax.set_xlim(-.3,len(LEADS)+.6); ax.set_xlabel('Lead time, с'); ax.set_ylabel('Спіймано зривів, %'); ax.grid(axis='x',visible=False)
fin(fig,ax,'Наша модель ловить удвічі більше зривів, ніж бейзлайн','Офіційний Test run у toksym: 200 розрядів, seed 1, бюджет хибних тривог 10 %','01_vs_baseline')
# 2. heatmap surface pooled 1300
T=sum(n[s]*table(*R['Фізична формула'][s]) for s in n)/1300
fig,ax=plt.subplots(figsize=(8,5.2)); from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('b',['#cde2fb','#86b6ef','#2a78d6','#0d366b']); im=ax.imshow(T*100,cmap=cm,vmin=0,vmax=100,aspect='auto')
for i in range(T.shape[0]):
    for k in range(T.shape[1]): ax.text(k,i,f'{T[i,k]*100:.0f}',ha='center',va='center',fontsize=10,color='white' if T[i,k]>0.5 else INK)
ax.set_xticks(range(len(FPRS))); ax.set_xticklabels([f'{int(f*100)} %' for f in FPRS]); ax.set_yticks(range(len(LEADS))); ax.set_yticklabels([str(l).replace('.',',')+' с' for l in LEADS])
ax.set_xlabel('Бюджет хибних тривог'); ax.set_ylabel('Lead time'); ax.grid(False); ax.set_xticks(np.arange(-.5,len(FPRS)),minor=True); ax.set_yticks(np.arange(-.5,len(LEADS)),minor=True); ax.grid(which='minor',color=BG,lw=2); ax.tick_params(which='minor',length=0)
for s in ax.spines.values(): s.set_visible(False)
fin(fig,ax,'Поверхня TPR: частка спійманих зривів, %','Фізична модель на 1300 тестових розрядах (seed 101, 202 і набір команди); середнє по сітці 69 %','02_tpr_surface')
# 3. model comparison with bootstrap CI
names=['Фізична формула','GRU-поправка до формули','GRU з ознаками формули','GRU без формули']; lab=['Фізична формула (у здачі)','GRU-поправка до формули','GRU з ознаками формули','GRU без формули']
vals=[];lo=[];hi=[]
for m in names:
    v=sum(n[s]*table(*R[m][s]).mean() for s in n)/1300; bb=sum(n[s]*boot([R[m][s]],n=400,seed=1)[:,0] for s in n)/1300; vals.append(v); lo.append(np.percentile(bb,2.5)); hi.append(np.percentile(bb,97.5))
fig,ax=plt.subplots(figsize=(8,4.2)); y=np.arange(len(names))[::-1]
ax.barh(y,vals,height=0.5,color=C1); ax.errorbar(vals,y,xerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',ecolor=INK2,elinewidth=1.2,capsize=3)
for yy,v,h in zip(y,vals,hi): ax.text(h+0.008,yy,f'{v:.3f}'.replace('.',','),va='center',fontsize=10)
ax.set_yticks(y); ax.set_yticklabels(lab,color=INK); ax.xaxis.set_major_formatter(comma); ax.set_xlim(0,0.85); ax.set_xlabel('Середній TPR по сітці Lead time × бюджет'); ax.grid(axis='y',visible=False); fig.subplots_adjust(left=0.36)
fin(fig,ax,'Нейромережі не обіграли фізичну формулу','1300 тестових розрядів, навчання на 4000; відрізки — 95 % інтервал (бутстреп по розрядах)','03_models')
# 4. causes
H=pd.read_pickle('hyp2_R.pkl'); D=H[H.dis]; NS=D[~D.sp]; c=NS.dom.value_counts()
labs=['Межа Грінвальда\n(густина)','Кінк-межа\n(низький q95)','Радіаційний\nколапс','Межа Тройона\n(бета)']; v=[c.get('gw',0),c.get('q',0),c.get('rad',0),c.get('beta',0)]
fig,ax=plt.subplots(figsize=(8,4.4)); ax.bar(range(4),v,width=0.55,color=C1)
for i,x_ in enumerate(v): ax.text(i,x_+12,f'{x_}  ({x_/sum(v)*100:.0f} %)',ha='center',fontsize=10)
ax.set_xticks(range(4)); ax.set_xticklabels(labs,color=INK); ax.set_ylabel('Зривів'); ax.set_ylim(0,640); ax.grid(axis='x',visible=False)
fin(fig,ax,'Причини зривів: три межі, жодного бета-зриву','1345 неспонтанних зривів на 5000 розрядів; домінантна межа в момент жереба за оцінкою моделі','04_causes')
# 5. lambda at trigger
fig,ax=plt.subplots(figsize=(8,4.4)); bins=np.logspace(np.log10(0.004),0,26)
ax.hist(D[D.sp].lam.clip(0.004,1),bins=bins,color=C2,alpha=0.9,label=f'Спонтанні ({int(D.sp.sum())})',edgecolor=BG,lw=1); ax.hist(NS.lam.clip(0.004,1),bins=bins,color=C1,alpha=0.9,label=f'Неспонтанні ({len(NS)})',edgecolor=BG,lw=1)
ax.set_xscale('log'); ax.set_xlabel('Оцінка інтенсивності зриву λ в момент жереба, с⁻¹'); ax.set_ylabel('Зривів'); ax.legend(frameon=False,loc='upper center'); ax.grid(axis='x',visible=False)
ax.axvline(0.02,color=MUT,lw=1,ls='--'); ax.text(0.022,ax.get_ylim()[1]*0.62,'поріг 0,02',color=INK2,fontsize=9)
fin(fig,ax,'Спонтанні зриви лежать на підлозі ризику','98,7 % спонтанних мають λ < 0,02 с⁻¹; серед неспонтанних таких 0,5 %. Поділ модель не бачила','05_lambda_at_trigger')
# 6. neon
A=H[(H.alive==True)&~H.sp].copy(); A['win']=A.dis&((A.tD/A.dur)>=0.765)&((A.tD/A.dur)<=0.83); A['hi']=A.gpk>0.7
rng=np.random.default_rng(0); res=[]
for m in [True,False]:
    x_=A[A.hi==m].win.values; b=[x_[rng.integers(0,len(x_),len(x_))].mean() for _ in range(1000)]; res.append((x_.mean(),np.percentile(b,2.5),np.percentile(b,97.5),len(x_)))
fig,ax=plt.subplots(figsize=(8,4.6)); v=[r[0]*100 for r in res]
ax.bar([0,1],v,width=0.5,color=C1); ax.errorbar([0,1],v,yerr=[[v[i]-res[i][1]*100 for i in range(2)],[res[i][2]*100-v[i] for i in range(2)]],fmt='none',ecolor=INK2,elinewidth=1.2,capsize=4)
for i in range(2): ax.text(i,res[i][2]*100+1.5,f'{v[i]:.1f} %'.replace('.',','),ha='center',fontsize=11)
ax.set_xticks([0,1]); ax.set_xticklabels([f'Сильний напуск неону\n(команда клапана > 0,7; {res[0][3]} розрядів)',f'Решта\n({res[1][3]} розрядів)'],color=INK); ax.set_ylabel('Зрив у вікні після вимкнення нагріву, %'); ax.set_ylim(0,46); ax.grid(axis='x',visible=False)
fin(fig,ax,'Неон робить вікно після вимкнення нагріву небезпечним','Розряди, що дожили до 0,74 тривалості; відрізки — 95 % інтервал','06_neon_window')
# 7. dead channels before/after
d0=pickle.load(open('dead_out.pkl','rb')); d3=pickle.load(open('dead_out3.pkl','rb')); ch=['prog_ip_MA','coil_pf_vert_kA','mag_vloop_V','prog_ne_1e20']; m0=table(*d0['base']).mean(); m3=table(*d3['base']).mean()
a=[table(*d0[c]).mean()-m0 for c in ch]; b=[table(*d3[c]).mean()-m3 for c in ch]
fig,ax=plt.subplots(figsize=(8,4.4)); y=np.arange(len(ch))[::-1]; h=0.34
ax.barh(y+h/2+0.01,a,height=h,color=C2,label='До виправлення'); ax.barh(y-h/2-0.01,b,height=h,color=C1,label='Після виправлення')
for yy,va,vb in zip(y,a,b): ax.text(va-0.006,yy+h/2,fmt(va),ha='right',va='center',fontsize=9.5); ax.text(min(vb,0)-0.006,yy-h/2,fmt(vb),ha='right',va='center',fontsize=9.5)
ax.set_yticks(y); ax.set_yticklabels(ch,color=INK,family='DejaVu Sans Mono'); ax.xaxis.set_major_formatter(comma); ax.set_xlim(-0.40,0.02); ax.axvline(0,color=AX,lw=1); ax.set_xlabel('Зміна середнього TPR при відмові каналу на весь розряд'); ax.legend(frameon=False,loc='lower left'); ax.grid(axis='y',visible=False); fig.subplots_adjust(left=0.24)
fin(fig,ax,'Запасні оцінки закрили найбільші вразливості','Набір команди, 200 розрядів; базовий рівень без відмов 0,770 в обох версіях','07_dead_channels')
print('ok',To[4],Tb[4],vals,lo,hi)

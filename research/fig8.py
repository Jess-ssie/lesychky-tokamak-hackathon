import sys,pickle,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/home/claude/work'); from shotstats import LEADS
BG='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUT='#898781'; GRID='#e1e0d9'; AX='#c3c2b7'; C1='#2a78d6'; C2='#eb6834'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.facecolor':BG,'figure.facecolor':BG,'axes.edgecolor':AX,'axes.labelcolor':INK2,'xtick.color':MUT,'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.8,'axes.axisbelow':True,'text.color':INK})
s=pd.read_csv('run_shots.csv').set_index('shot_id'); m=pd.read_csv('/mnt/user-data/uploads/run_measured_5.csv',usecols=['shot_id','t','p'])
safe=[];dis=[]
for sid,g in m.groupby('shot_id',sort=False):
    r=s.loc[sid]; p=g.p.values; t=g.t.values
    if not r.disrupted: safe.append(p.max())
    elif r.verdict!='excluded': k=t<=r.t_tq_s-0.3+1e-9; dis.append(p[k].max() if k.any() else -1)
safe=np.sort(safe)[::-1]; dis=np.array(dis)
def curve(safe,dis,maxf=0.30):
    ks=np.arange(0,int(maxf*len(safe))+1); return ks/len(safe)*100,np.array([(dis>safe[k]).mean()*100 for k in ks]),np.array([(dis>safe[k]).sum() for k in ks])
x1,y1,c1=curve(safe,dis)
for b in [0.05,0.09,0.15]:
    k=int(np.floor(b*len(safe)+1e-9)); print('budget',b,'FA',k,'caught',c1[k],'thr %.3f'%safe[k])
R=pickle.load(open('final_R.pkl','rb'))['Фізична формула']; S=np.sort(np.concatenate([R[n][0] for n in ['s101','va','t200']]))[::-1]; D=np.concatenate([R[n][1][:,LEADS.index(0.3)] for n in ['s101','va','t200']])
x2,y2,_=curve(S,D); print('1300: 5%% %.0f 9%% %.0f 15%% %.0f'%(y2[int(.05*len(S))],y2[int(.09*len(S))],y2[int(.15*len(S))]),len(S),len(D))
fig,ax=plt.subplots(figsize=(8.4,5))
ax.step(x2,y2,where='post',color=C2,lw=2,label='1300 тестових розрядів'); ax.step(x1,y1,where='post',color=C1,lw=2,label='Test run команди (200 розрядів)')
OFF={0.05:44,0.09:47,0.15:58}
for b,dx,dy in [(0.05,-0.6,6),(0.09,0.5,-9),(0.15,-0.8,5)]:
    k=int(np.floor(b*len(safe)+1e-9)); yo=OFF[b]/len(dis)*100; ax.plot(x1[k],yo,'o',ms=9,color=C1,mec=BG,mew=2,zorder=5)
    ax.text(x1[k]+dx,yo+dy,f'бюджет {int(b*100)} %\n{OFF[b]} з {len(dis)} ({yo:.0f} %)'.replace('.',','),fontsize=9.5,color=INK,ha='right' if dx<0 else 'left',va='center')
ax.legend(frameon=False,loc='lower right',fontsize=10)
ax.axvspan(0,5,color='#2a78d6',alpha=0.06,lw=0); ax.text(2.5,97,'обрана зона\n≤ 5 %',ha='center',va='top',fontsize=9,color=INK2)
ax.set_xlim(0,30.5); ax.set_ylim(0,104); ax.set_xlabel('Хибні тривоги серед безпечних розрядів, %'); ax.set_ylabel('Спіймано зривів, %'); ax.set_xticks([0,5,10,15,20,25,30])
fig.text(0.02,0.975,'Ціна кожної додаткової хибної тривоги при Lead time 0,3 с',fontsize=14,fontweight='bold',va='top'); fig.text(0.02,0.915,'Крива на 200 розрядах східчаста через малу вибірку; на 1300 розрядах зростання плавне',fontsize=10,va='top',color=INK2)
fig.subplots_adjust(top=0.82); fig.savefig('figs/08_budget_tradeoff.png',dpi=160)

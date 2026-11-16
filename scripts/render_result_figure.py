"""Render descriptive research figures in a separate reporting environment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/real/figures'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=OUTPUT/'FIGURE_INPUTS.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':12,'svg.hashsalt':'structural-break-frozen-dev'})
    fig,axes=plt.subplots(3,2,figsize=(16,15))
    fig.subplots_adjust(top=.9,bottom=.18,left=.155,right=.96,hspace=.5,wspace=.3)
    fig.suptitle('Structural Break — frozen DEV results',fontsize=22,fontweight='bold',y=.968)
    fig.text(.5,.937,'8,000 DEV series | 4,014,405 online points | 5 fixed folds | 249 completed candidates',ha='center',fontsize=11,color='#465261')
    colors={'PRIMARY_DEV_CANDIDATE':'#126b96','BACKUP_ROBUST_CANDIDATE':'#d77518','BACKUP_FAST_CANDIDATE':'#288457'}
    labels={'PRIMARY_DEV_CANDIDATE':'Primary: AR4 LGB + AR8 HistGB',
        'BACKUP_ROBUST_CANDIDATE':'Robust: AR8 ARCH1 HistGB','BACKUP_FAST_CANDIDATE':'Fast: AR4 LGB (50 trees)'}
    ax=axes[0,0];rows=data['statistical_ladder'];values=[r['mean'] for r in rows]
    names=['B0 constant','B1 official EWMA','B2 mean shift','B3 CUSUM','B4 scale','B5 tails','B6 dependence','B7 combined','C2 AR1 residual\nscale LR']
    y=np.arange(len(rows));ax.hlines(y,.5,values,color='#adb7c3',lw=2);ax.scatter(values,y,color=['#4d6074']*8+['#126b96'],s=45,zorder=3)
    for index,value in enumerate(values):ax.text(value+.001,index,f'{value:.4f}',va='center',fontsize=9)
    ax.set(yticks=y,yticklabels=names,xlim=(.498,.562),xlabel='Mean DEV TS-AUC',title='A. Statistical baseline ladder');ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    ax=axes[0,1]
    for role,row in data['selected'].items():ax.plot(range(5),row['fold_scores'],marker='o',color=colors[role],label=labels[role],lw=1.8)
    ax.set(xticks=range(5),xlabel='Frozen validation fold',ylabel='DEV TS-AUC',ylim=(.59,.638),title='B. The three frozen roles');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
    for ax,key,ticks,title in (
        (axes[1,0],'time_profile',['0-9','10-24','25-49','50-99','100-199','200-399','400+'],'C. Online-time discrimination'),
        (axes[1,1],'age_profile',['0-4','5-9','10-19','20-49','50+'],'D. Discrimination after the break'),
    ):
        for role,row in data['selected'].items():ax.plot(range(len(ticks)),[row[key][t] for t in ticks],marker='o',color=colors[role],lw=1.8)
        ax.plot(range(len(ticks)),[data['baseline'][key][t] for t in ticks],ls='--',color='#7d8792',label='Official EWMA')
        ax.axhline(.5,color='#bcc4ce',lw=1);ax.set(xticks=range(len(ticks)),xticklabels=ticks,ylabel='Mean fold TS-AUC',ylim=(.48,.65),title=title)
        ax.tick_params(axis='x',labelsize=9);ax.grid(alpha=.18);ax.legend(loc='upper left',fontsize=9)
    axes[1,0].set_xlabel('Online step bucket');axes[1,1].set_xlabel('Break age bucket; all negative controls retained')
    ax=axes[2,0];families=data['matched_ar4_panel']['families'];order=sorted(families,key=families.get,reverse=True)
    y=np.arange(4);values=[families[name] for name in order]
    ax.hlines(y,.5,values,color='#adb7c3',lw=2);ax.scatter(values,y,s=65,color='#126b96',zorder=3)
    for index,value in enumerate(values):ax.text(value+.002,index,f'{value:.4f}',va='center')
    full={'A':'A: location','B':'B: scale','C':'C: tail / distribution','D':'D: dependence'}
    ax.set(yticks=y,yticklabels=[full[x] for x in order],xlim=(.495,.625),xlabel='Mean DEV TS-AUC',title='E. Matched AR4 component ablation');ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    ax.text(0,-.31,'Same AR4 learner / sampling / calibration;\nthis is not an ablation of the final two-model average.',transform=ax.transAxes,fontsize=9,color='#465261')
    ax=axes[2,1];roles=list(data['selected']);y=np.arange(3);h=.32
    point=[data['latency'][r]['point_us'] for r in roles];total=[data['latency'][r]['including_initialization_estimate_us'] for r in roles]
    ax.barh(y-h/2,point,height=h,color='#126b96',label='Measured point-only')
    ax.barh(y+h/2,total,height=h,color='#b7d2df',edgecolor='#648b9e',hatch='//',label='With historical initialization (estimate)')
    for index,(a,b) in enumerate(zip(point,total)):
        ax.text(a+.2,index-h/2,f'{a:.2f}',va='center',fontsize=9);ax.text(b+.2,index+h/2,f'{b:.2f}',va='center',fontsize=9)
    ax.set(yticks=y,yticklabels=['Primary','Robust','Fast'],xlim=(0,max(total)*1.15),xlabel='Microseconds per point',title='F. Actual final-model CPU cost');ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.legend(fontsize=8,loc='lower right')
    ax.text(0,-.31,'Windows; point-only stream repeats real values.\nPrimary: 10M points; backups: 1M each; init: 128 histories.\nFile I/O, initial JIT and socket transport excluded.',transform=ax.transAxes,fontsize=9,color='#465261')
    fig.text(.155,.04,'Descriptive DEV comparisons after adaptive selection; no selection-adjusted confidence intervals.\nBreak-age TS-AUC is not detection delay or an alarm rate. LOCAL_FINAL_SEAL_OPENED = false. Reduced scores are omitted.',fontsize=10,color='#465261')
    paths=[]
    for suffix in ('png','pdf','svg'):
        path=OUTPUT/f'DEV_RESULTS_OVERVIEW.{suffix}'
        metadata={'Creator':'Structural Break report'} if suffix in ('pdf','svg') else {'Software':'Structural Break report'}
        if suffix=='pdf':metadata.update(CreationDate=None,ModDate=None)
        if suffix=='svg':metadata['Date']=None
        fig.savefig(path,dpi=180,metadata=metadata);paths.append(path)
    plt.close(fig)
    report={'status':'PASS','figure_inputs_sha256':digest(source),'renderer_sha256':digest(Path(__file__)),
        'matplotlib':importlib.metadata.version('matplotlib'),'numpy':importlib.metadata.version('numpy'),
        'outputs':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths],
        'scope':data['scope'],'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    (OUTPUT/'FIGURE_RENDER_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('DEV SUMMARY FIGURES RENDERED',len(paths))


if __name__=='__main__':main()

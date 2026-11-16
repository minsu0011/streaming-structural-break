"""Figures3..10 from fixed diagnostics; run in the existing figure environment."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
os.environ['MPLCONFIGDIR']=str(BASE/'mpl')


def main():
    from src.next9.boundary import install_workspace_boundary
    violations=install_workspace_boundary()
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from PIL import Image
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    data=BASE/'figure_data';manifest=read_json(data/'DONE.json')
    for name,sha in manifest['files_sha256'].items():assert digest(data/name)==sha,name
    candidates=manifest['candidates'];colors={0:'#d15a28',1:'#a1a7ad',2:'#2077b4',3:'#602090'}
    with np.load(data/'TRAJECTORIES.npz') as z:trajectory={k:z[k] for k in z.files}
    output=BASE/'FIGURES';output.mkdir(parents=True,exist_ok=True);files=[]
    def finish(fig,number,name):
        path=output/f'FIG{number}_{name}'
        fig.savefig(str(path)+'.png',dpi=160,bbox_inches='tight');fig.savefig(str(path)+'.pdf',bbox_inches='tight')
        plt.close(fig);files.extend([path.with_suffix('.png'),path.with_suffix('.pdf')])
    def nine(title):
        fig,axes=plt.subplots(3,3,figsize=(16,10),layout='constrained');fig.suptitle(title,fontsize=17)
        return fig,axes.ravel()
    with activity('Render eight final evidence figures and assemble the ten-figure PDF','FIGURE_RENDERING','B2_NO_REPLICATED_OPPOSITE_ORIENTATION'):
        fig,axes=nine(f'FIG3 | Actual held-out trajectories, series {manifest["trajectory_dataset_id"]}\nColor is pre-current DOWN / FLAT / UP, determined before F[t]')
        for ax,j in zip(axes,candidates):
            t=trajectory['time'];v=trajectory[f'F{j}_current'];state=trajectory[f'F{j}_S1_DIRECTION']
            ax.plot(t,v,color='#b0b0b0',lw=.7)
            for code in [0,1,2,3]:
                keep=state==code;ax.scatter(t[keep],v[keep],s=7,color=colors[code],label=['DOWN','FLAT','UP','UNKNOWN'][code])
            ax.set_title(f'F{j:03d}');ax.set_xlabel('Observed online time');ax.set_ylabel('H-normalized F')
        axes[0].legend(ncol=2,fontsize=8);finish(fig,3,'ACTUAL_PRECURRENT_TRAJECTORIES')
        bins=read_json(data/'COMMON_F_BINS.json')['records']
        fig,axes=nine('FIG4 | Same TRAIN-frozen current-F bins: previous UP vs DOWN\nFold0 held-out descriptive group-weighted break fractions; main inference is same-time TS-AUC')
        for ax,j in zip(axes,candidates):
            for state in [0,2]:
                rows=[r for r in bins if r['feature']==j and r.get('state')==state]
                x=[r['current_F_midpoint'] for r in rows];y=[np.nan if r['group_weighted_break_fraction'] is None else r['group_weighted_break_fraction'] for r in rows]
                ax.plot(x,y,'.-',color=colors[state],lw=1.2,label='DOWN' if state==0 else 'UP')
            ax.set_title(f'F{j:03d}');ax.set_xlabel('Same current F bin');ax.set_ylabel('Break fraction');ax.set_ylim(-.03,1.03)
            if not any(r.get('state') in [0,2] for r in bins if r['feature']==j):ax.text(.5,.5,'Insufficient TRAIN common range',transform=ax.transAxes,ha='center',fontsize=9)
        axes[0].legend();finish(fig,4,'MATCHED_CURRENT_F_ORIENTATION')
        transition=read_json(BASE/'SELF_TRANSITION_MODELS.json')['records']
        fig,axes=nine('FIG5 | TRAIN-chosen self laws: predicted transition probabilities\nAverage over held-out covariates and original folds; rows=previous, columns=next state')
        for ax,j in zip(axes,candidates):
            rows=[r for r in transition if r['feature']==j and r['mode']=='NORMAL'];matrices=[]
            for r in rows:
                matrix=r['matrices']['predicted_on_heldout_covariates'][r['chosen_variant']]
                matrices.append(np.array([[np.nan if v is None else v for v in row[:3]] for row in matrix[:3]]))
            matrix=np.nanmean(np.stack(matrices),axis=0)
            im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
            for a in range(3):
                for b in range(3):ax.text(b,a,f'{matrix[a,b]:.2f}',ha='center',va='center',color='white' if matrix[a,b]>.55 else 'black')
            ax.set_xticks(range(3),['DOWN','FLAT','UP']);ax.set_yticks(range(3),['DOWN','FLAT','UP']);ax.set_title(f'F{j:03d}')
        fig.colorbar(im,ax=axes.tolist(),shrink=.65,label='Predicted probability; UNKNOWN class omitted');finish(fig,5,'SELF_TRANSITION_MATRICES')
        j=candidates[0];t=trajectory['time'];path=trajectory[f'F{j}_path'];fig,axes=plt.subplots(4,1,figsize=(16,10),sharex=True,layout='constrained')
        fig.suptitle(f'FIG6 | F{j:03d} frozen self-transition state timeline | held-out series {manifest["trajectory_dataset_id"]}',fontsize=16)
        axes[0].plot(t,trajectory[f'F{j}_current'],color='#34546a');axes[0].set_ylabel('Current F')
        axes[1].step(t,trajectory[f'F{j}_S1_DIRECTION'],where='post',color='#2676a5');axes[1].set_yticks([0,1,2],['DOWN','FLAT','UP']);axes[1].set_ylabel('Pre-current state')
        axes[2].plot(t,path[:,3],label='Reversal probability');axes[2].plot(t,path[:,4],label='Continuation probability');axes[2].set_ylim(-.02,1.02);axes[2].legend(loc='upper right');axes[2].set_ylabel('Frozen-law probability')
        axes[3].plot(t,path[:,2],color='#9a4c80');axes[3].set_ylabel('Previous-issued surprise');axes[3].set_xlabel('Observed online time; no current-label parameter updates')
        finish(fig,6,'TRANSITION_STATE_TIMELINE')
        coefficients=read_json(data/'CONDITIONAL_COEFFICIENTS.json')['records'];matrix=np.zeros((len(candidates),5));active=np.zeros_like(matrix,dtype=bool);text={}
        for a,j in enumerate(candidates):
            for fold in range(5):
                rr=sorted([r for r in coefficients if r['feature']==j and r['fold']==fold],key=lambda r:r['pair_state_index'])
                active[a,fold]=any(r['active_conditional_branch'] for r in rr)
                matrix[a,fold]=rr[0]['conditional_increment_coefficient']-rr[1]['conditional_increment_coefficient']
                text[(a,fold)]=f'{rr[0]["conditional_increment_coefficient"]:.3g} / {rr[1]["conditional_increment_coefficient"]:.3g}' if active[a,fold] else 'absent'
        fig,ax=plt.subplots(figsize=(13,9),layout='constrained');limit=max(float(np.abs(matrix).max()),.01)
        im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        for (a,fold),label in text.items():ax.text(fold,a,label,ha='center',va='center',fontsize=10)
        ax.set_xticks(range(5),[f'Outer {k}' for k in range(5)]);ax.set_yticks(range(len(candidates)),[f'F{j:03d}' for j in candidates])
        ax.set_title('FIG7 | Joint M4 conditional increment coefficients by TRAIN-chosen state pair\nText: first / second coefficient. Absent branch is not an estimated zero effect.',fontsize=15)
        fig.colorbar(im,ax=ax,label='First minus second conditional coefficient; not total F effect');finish(fig,7,'CONDITIONAL_STATE_COEFFICIENTS')
        graph=read_json(BASE/'PREDICTIVE_FACTOR_GRAPH_NEXT9.json');cert=read_json(BASE/'VARIABLE_CERTIFICATION_NEXT9.json');certified={r['feature'] for r in cert['certified_rules']}
        fig,ax=plt.subplots(figsize=(14,10),layout='constrained');angles=np.linspace(0,2*np.pi,len(candidates),endpoint=False);pos={j:np.array([np.cos(a),np.sin(a)]) for j,a in zip(candidates,angles)}
        for j,(x,y) in pos.items():
            ax.scatter([x],[y],s=2100,color='#add8b8' if j in certified else '#dce3ea',edgecolor='#526779',zorder=3)
            ax.text(x,y,f'F{j:03d}',ha='center',va='center',zorder=4)
            arrow=FancyArrowPatch((x-.06,y+.11),(x+.06,y+.11),connectionstyle='arc3,rad=-2',arrowstyle='->',mutation_scale=12,color='#6d8594')
            ax.add_patch(arrow)
        for edge in graph['certified_edges']:
            a,b=pos[edge['parent']],pos[edge['destination']];ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=18,shrinkA=28,shrinkB=28,color='#21864e'))
        ax.set_xlim(-1.55,1.55);ax.set_ylim(-1.4,1.55);ax.axis('off')
        ax.set_title(f'FIG8 | Certified cross edges: {graph["cross_edge_count"]} | B-certified features: {len(certified)}\nSelf loops depict auxiliary self-law diagnostics (A), not causal discovery.',fontsize=17)
        if not graph['certified_edges']:ax.text(0,0,'ZERO CROSS EDGES\nNo threshold relaxation',ha='center',va='center',fontsize=19,color='#526779')
        finish(fig,8,'FROZEN_SELF_AND_CROSS_GRAPH')
        with (BASE/'MODEL_COMPARISON_NEXT9.csv').open(encoding='utf-8-sig',newline='') as f:models=list(csv.DictReader(f))
        selected=['M0','M2','M4','M5','M7'];mr=[next(r for r in models if r['mode']=='NORMAL' and r['method']==m) for m in selected]
        fig,axes=plt.subplots(1,2,figsize=(16,8),layout='constrained');x=np.arange(len(mr))
        for fold in range(5):axes[0].plot(x,[float(r[f'fold_{fold}_TS_AUC']) for r in mr],'.-',alpha=.4,label=f'Fold {fold}')
        axes[0].plot(x,[float(r['mean_TS_AUC']) for r in mr],'ko-',lw=2,label='Equal-fold mean');axes[0].legend();axes[0].set_ylabel('Exact TS-AUC')
        points=np.array([float(r['mean_delta_vs_BL80']) for r in mr]);low=np.array([float(r['ci_low']) for r in mr]);high=np.array([float(r['ci_high']) for r in mr])
        axes[1].vlines(x,low,high,color='#236c93');axes[1].scatter(x,points,color='#236c93');axes[1].axhline(0,color='black',lw=.8);axes[1].axhline(.0012,color='#36864c',ls='--',label='Promotion mean gate +.0012');axes[1].legend();axes[1].set_ylabel('Paired delta vs matched BL80; 95% group bootstrap CI')
        for ax in axes:ax.set_xticks(x,selected);ax.grid(axis='y',alpha=.15)
        fig.suptitle('FIG9 | Frozen model comparison on original excluded outer groups\nAdaptive RP development; performance gain is separate from direction certification',fontsize=17);finish(fig,9,'MATCHED_MODEL_COMPARISON')
        with (BASE/'PAIRED_UNCERTAINTY_NEXT9.csv').open(encoding='utf-8-sig',newline='') as f:uncertainty=list(csv.DictReader(f))
        labels=[('NORMAL',f'TIME_{k}') for k in range(7)]+[('NORMAL',f'H_LENGTH_{k}') for k in range(5)]+[('NORMAL',f'H_LAG1_{k}') for k in range(3)]+[('NORMAL',f'H_TAIL3_{k}') for k in range(3)]+[('EARLY','ALL')]
        fig,ax=plt.subplots(figsize=(14,11),layout='constrained')
        for i,(mode,name) in enumerate(labels):
            r=next(v for v in uncertainty if v['candidate']=='M7' and v['reference']=='M0' and v['mode']==mode and v['domain']==name)
            if r['point'] and r['ci_low'] and r['ci_high']:
                value,lo,hi=map(float,[r['point'],r['ci_low'],r['ci_high']]);color='#237595' if mode=='NORMAL' else '#c06024';ax.hlines(i,lo,hi,color=color);ax.scatter([value],[i],color=color)
            else:ax.text(0,i,'  insufficient support',va='center',fontsize=8,color='#666666')
        ax.axvline(0,color='black',lw=.8);ax.axvline(-.001,color='#bc5a39',ls='--',lw=1);ax.set_yticks(range(len(labels)),[f'{m}: {n}' for m,n in labels]);ax.invert_yaxis();ax.set_xlabel('M7 minus BL80 TS-AUC; paired whole-original-group 95% CI')
        ax.set_title('FIG10 | Fixed time / TRAIN-only H regimes and early-to-late transfer\nUNKNOWN H bins excluded from replication claims; all bins remain in the tables',fontsize=16);ax.grid(axis='x',alpha=.15);finish(fig,10,'TIME_H_AND_EARLY_LATE')
        pages=[output/'FIG1_153_MOVEMENT_ATLAS.png',output/'FIG2_PERSISTENCE_CENTER_OWN_V2.png']+[p for p in files if p.suffix=='.png']
        assert len(pages)==10
        images=[Image.open(p).convert('RGB') for p in pages]
        images[0].save(output/'NEXT9_TEN_FIGURES.pdf',save_all=True,append_images=images[1:],resolution=160.)
        for img in images:img.close()
        assert not violations
        atomic(output/'FINAL_FIGURES_DONE.json',dict(status='COMPLETE',timestamp=utc_now(),figure_count=10,
            preferred_pngs=[p.name for p in pages],files_sha256={p.name:digest(p) for p in [*files,output/'NEXT9_TEN_FIGURES.pdf']},
            visual_inspection_pending=True,combined_PDF='Ten original rendered scientific plots, rasterized at160dpi for dependency-free assembly'))


if __name__=='__main__':main()

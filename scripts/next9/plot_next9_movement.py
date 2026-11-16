"""Two fixed A-stage figures from fully frozen target-free evidence."""
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
os.environ['MPLCONFIGDIR']=str(BASE/'mpl')


def main():
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    with activity('Render frozen all153 movement atlas and Hcenter comparison','FIGURES','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
        freeze=read_json(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')
        source=BASE/'TARGET_FREE_MOVEMENT_SCREEN.csv'
        assert digest(source)==freeze['result_sha256'][source.name]
        with source.open(encoding='utf-8',newline='') as f: records=list(csv.DictReader(f))
        records=[r for r in records if not r['inner'] and r['mode']=='NORMAL']
        values=np.zeros((153,6))
        mse=np.zeros((153,3,3))
        for j in range(153):
            for k,h in enumerate([1,4,16]):
                for b,reference in enumerate(['persistence_contrast','H_center_contrast']):
                    r=[v for v in records if int(v['feature'])==j and int(v['horizon'])==h and v['reference']==reference]
                    assert len(r)==5
                    values[j,b*3+k]=np.mean([float(v['mean_gain'])/max(float(v['H_center_MSE']),1e-8) for v in r])
                    if b==0:
                        mse[j,k]=np.mean([[float(v[n]) for n in ['D0A_MSE','H_center_MSE','D1_MSE']] for v in r],axis=0)
        outer=[r for r in freeze['candidates'] if '_inner_' not in r['scope']]
        support={j:sum(j in r['normal_candidates'] for r in outer) for j in range(153)}
        selected=sorted(j for j,n in support.items() if n==5)
        folder=BASE/'FIGURES';folder.mkdir(parents=True,exist_ok=True)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
        transformed=np.sign(values)*np.log1p(np.abs(values))
        limit=max(.01,float(np.quantile(np.abs(transformed),.98)))
        fig,axes=plt.subplots(1,3,figsize=(16,17),layout='constrained')
        for block,ax in enumerate(axes):
            js=np.arange(block*51,(block+1)*51)
            im=ax.imshow(transformed[js],aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
            ax.set_yticks(np.arange(51),[f'F{j:03d}'+('  *' if support[j]==5 else '') for j in js],fontsize=7)
            ax.set_xticks(np.arange(6),['P h1','P h4','P h16','H h1','H h4','H h16'],rotation=45,ha='right')
            ax.axvline(2.5,color='black',lw=1)
            ax.set_title(f'Features {js[0]:03d}–{js[-1]:03d}')
        fig.colorbar(im,ax=axes,label='signed log1p(gain / H-center MSE); blue = D1 improvement',shrink=.45)
        fig.suptitle('NEXT9: all 153 target-free movement forecasts\nP = current-F persistence; H = H-normal median; * = selected in all five TRAIN procedures',fontsize=13)
        fig.supxlabel('Internal heldout evidence within original outer TRAIN; overlapping procedures are not independent confirmations.\nA-stage only: no break labels, no variable-orientation or BL80-gain claim. Color saturates at the 98th absolute percentile.',fontsize=9)
        files=[]
        for ext in ['png','pdf']:
            path=folder/('FIG1_153_MOVEMENT_ATLAS.'+ext);fig.savefig(path,dpi=170);files.append(path)
        plt.close(fig)
        fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
        colors=['#7a7a7a','#e89a31','#246eac']
        for ax,j in zip(axes.flat,selected):
            x=np.arange(3)
            for b,(name,color) in enumerate(zip(['D0A persistence','D0B H median','D1 own history'],colors)):
                ax.bar(x+(b-1)*.24,np.maximum(mse[j,:,b],1e-10),width=.23,label=name,color=color)
            ax.set_xticks(x,['h=1','h=4','h=16']);ax.set_yscale('log');ax.set_title(f'F{j:03d}');ax.set_ylabel('Group-mean MSE (log scale)')
            ax.grid(axis='y',alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle('The nine consistently selected movement candidates\nPersistence, H-normal center, and strict own-history forecasting',fontsize=14)
        fig.supxlabel('Equal mean of five TRAIN-procedure internal-validation summaries. H median is not the MSE-optimal constant.\nCandidate eligibility additionally requires early-to-late and multiplicity/support gates.',fontsize=9)
        for ext in ['png','pdf']:
            path=folder/('FIG2_PERSISTENCE_CENTER_OWN.'+ext);fig.savefig(path,dpi=170);files.append(path)
        plt.close(fig)
        atomic(folder/'MOVEMENT_FIGURES_DONE.json',dict(status='COMPLETE',timestamp=utc_now(),
            source_csv_sha256=digest(source),candidate_freeze_sha256=digest(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json'),
            figures_sha256={p.name:digest(p) for p in files},A_only=True,outer_break_quality_used=False))


if __name__=='__main__':main()

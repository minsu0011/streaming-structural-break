"""Readable FIG2 layout; preserve the first rendering and exact A data."""
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
os.environ['MPLCONFIGDIR']=str(BASE/'mpl')


if __name__=='__main__':
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with activity('Repair FIG2 legend overlap using explicit margins and point curves','FIGURES','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
        frozen=read_json(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')
        source=BASE/'TARGET_FREE_MOVEMENT_SCREEN.csv'
        assert digest(source)==frozen['result_sha256'][source.name]
        with source.open(encoding='utf-8',newline='') as f:records=list(csv.DictReader(f))
        records=[r for r in records if not r['inner'] and r['mode']=='NORMAL' and r['reference']=='H_center_contrast']
        parents=[r for r in frozen['candidates'] if '_inner_' not in r['scope']]
        selected=sorted(set.intersection(*(set(r['normal_candidates']) for r in parents)))
        fig,axes=plt.subplots(3,3,figsize=(14,12))
        fig.subplots_adjust(left=.08,right=.97,bottom=.15,top=.90,hspace=.36,wspace=.33)
        colors=['#777777','#db8a20','#216aa5'];labels=['D0A persistence','D0B H median','D1 own history']
        for ax,j in zip(axes.flat,selected):
            mse=[]
            for h in [1,4,16]:
                rows=[r for r in records if int(r['feature'])==j and int(r['horizon'])==h]
                assert len(rows)==5
                mse.append(np.mean([[float(r[n]) for n in ['D0A_MSE','H_center_MSE','D1_MSE']] for r in rows],axis=0))
            mse=np.asarray(mse)
            for k in range(3):ax.plot(np.arange(3),mse[:,k],marker=['s','^','o'][k],color=colors[k],label=labels[k],lw=1.8)
            ax.set_xticks(np.arange(3),['h=1','h=4','h=16']);ax.set_yscale('log');ax.set_title(f'F{j:03d}')
            ax.set_ylabel('Group-mean MSE (log scale)',fontsize=9);ax.grid(alpha=.2)
            ax.margins(x=.15,y=.15)
        fig.suptitle('The nine consistently selected movement candidates\nPersistence, H-normal center, and strict own-history forecasting',fontsize=14,y=.97)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False)
        fig.text(.5,.027,'Equal mean of five TRAIN-procedure internal-validation summaries; overlapping procedures are not independent confirmations.\nH median is not the MSE-optimal constant. Eligibility additionally requires early-to-late, multiplicity, and support gates.',ha='center',va='center',fontsize=9)
        files=[]
        for ext in ['png','pdf']:
            p=BASE/'FIGURES'/('FIG2_PERSISTENCE_CENTER_OWN_V2.'+ext);fig.savefig(p,dpi=170);files.append(p)
        plt.close(fig)
        atomic(BASE/'FIGURES/FIG2_V2_DONE.json',dict(status='COMPLETE',timestamp=utc_now(),
            previous_rendering_preserved=True,reason='Legend overlapped caption; revised spacing and point curves make log-scale comparisons explicit',
            numerical_source_unchanged_sha256=digest(source),files_sha256={p.name:digest(p) for p in files}))

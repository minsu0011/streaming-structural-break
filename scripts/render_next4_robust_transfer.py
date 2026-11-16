"""Plot existing original and alternate-split intervals without new statistics."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    assert args.candidate in {'D2_DISCRETE_SCALE_MIXTURE','B1_SPLIT_ENERGY_BRANCHES'}
    assert args.tag.replace('_','').isalnum()
    cid=args.candidate
    paths=[BASE/'forensic'/cid/'RESULTS.json',BASE/'bootstrap'/cid/'RESULTS.json',
        BASE/'robustness_forensic'/cid/'RESULTS.json',BASE/'robustness'/cid/'RESULTS.json']
    forensic,bootstrap,scales,overall=(read_json(p) for p in paths)
    splits=['ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT']
    scale=[forensic['paired_scale']]+[next(s for s in scales['summaries'] if s['split']==split and s['scope']=='SCALE') for split in splits]
    total=[bootstrap['paired_all_time']]+[next(s for s in overall['summaries'] if s['split']==split)['paired_all_time'] for split in splits]
    out=BASE/'figures'/args.tag;out.mkdir(parents=True,exist_ok=True)
    stem=out/(cid+'_TRANSFER');assert not stem.with_suffix('.png').exists()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True)
    labels=['Original PRIMARY','ALT 1','ALT 2','ALT 3','Held-out H regime']
    for ax,rows,title,color in zip(axes,[scale,total],['SCALE subset','Overall TS-AUC'],['#2166ac','#b35806']):
        for y,row in enumerate(rows):
            value=row['estimate_mean_delta'];lo=row['ci_2_5'];hi=row['ci_97_5']
            ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color=color,capsize=4,markersize=6)
            ax.annotate(f'{value:+.5f}',(value,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        ax.set_title(title);ax.set_xlabel('TS-AUC difference vs matched BL80')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis='x',alpha=.2);ax.margins(x=.16)
    axes[0].set_yticks(range(5),labels);axes[0].set_ylim(4.5,-.65)
    fig.suptitle(cid.replace('_',' ')+' | fixed split transfer',fontsize=15,y=.97)
    fig.subplots_adjust(left=.17,right=.98,bottom=.23,top=.84,wspace=.23)
    fig.text(.17,.10,'Points and paired 95% intervals use the original 2,000 whole-series draws.\nSCALE: 132 changed series + 4,056 null series. ALT/H reuse the same DEV; no independent holdout.',fontsize=10)
    for suffix in ('.png','.svg'):fig.savefig(stem.with_suffix(suffix),dpi=160,facecolor='white')
    plt.close(fig)
    atomic_json(stem.with_suffix('.json'),{'status':'PASS','created_utc':utc_now(),'candidate':cid,
        'new_statistics':False,'source_sha256':digest(Path(__file__)),
        'evidence_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'files_sha256':{stem.with_suffix(s).name:digest(stem.with_suffix(s)) for s in ('.png','.svg')}},immutable=True)
    print('NEXT4 TRANSFER FIGURE',cid,args.tag,flush=True)


if __name__=='__main__':main()

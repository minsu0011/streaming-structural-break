"""Standalone evidence figures using the existing pinned figure environment."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if not args.worker:
        python=ROOT/'.figure-venv/Scripts/python.exe'
        if not python.is_file():raise RuntimeError('Existing pinned figure environment required; no implicit installation')
        subprocess.run([str(python),str(Path(__file__).resolve()),'--snapshot',args.snapshot,'--worker'],cwd=ROOT,
            check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));return
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert matplotlib.__version__=='3.10.6' and np.__version__=='2.5.3'
    snapshot=(ROOT/args.snapshot).resolve()
    if not snapshot.is_relative_to(ROOT/'artifacts/next4/report_tables'):raise ValueError('Explicit NEXT4 table snapshot required')
    result=json.loads((snapshot/'RESULTS.json').read_text(encoding='utf-8'));assert result['status']=='PASS'
    evidence={}
    def rows(name):
        path=snapshot/name
        assert sha(path)==result['files_sha256'][name]
        evidence[path.relative_to(ROOT).as_posix()]=sha(path)
        with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    out=ROOT/'artifacts/next4/figures'/snapshot.name;out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.bbox':'tight','svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.svg'));plt.close(fig)
    primary=sorted(rows('PAIRED_BOOTSTRAP_NEXT4.csv'),key=lambda r:float(r['estimate_mean_delta']))
    fig,ax=plt.subplots(figsize=(9,6.2))
    for y,row in enumerate(primary):
        point=float(row['estimate_mean_delta'])*1000;lo=float(row['ci_2_5'])*1000;hi=float(row['ci_97_5'])*1000
        ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',markersize=4,capsize=2,
            color='#176d97' if row['family']=='B' else '#697582')
    ax.axvline(0,color='#303943',lw=.8);ax.axvline(1.3,color='#c17812',ls='--',lw=1,label='Original numeric gate +0.0013')
    ax.set_yticks(range(len(primary)),[r['candidate'] for r in primary]);ax.set_xlabel('Mean TS-AUC gain over BL80 (x 0.001)')
    ax.set_title('Original primary 5-fold gain and paired 95% interval\nAll 14 intervals include zero',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.18);ax.legend(loc='lower right',fontsize=8)
    fig.text(.01,.005,'2,000 paired whole-series draws. Conditional on fitted OOF; no adjustment for adaptive selection.',fontsize=8)
    save(fig,'PRIMARY_GAIN_INTERVALS')
    mechanism=[r for r in rows('MECHANISM_RESULTS_NEXT4.csv') if r['scope']=='mechanism_SCALE_DOMINANT']
    mechanism.sort(key=lambda r:float(r['estimate_mean_delta']))
    fig,ax=plt.subplots(figsize=(9,6.2))
    for y,row in enumerate(mechanism):
        point=float(row['estimate_mean_delta'])*1000;lo=float(row['ci_2_5'])*1000;hi=float(row['ci_97_5'])*1000
        ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',markersize=4,capsize=2,color='#176d97' if lo>0 else '#697582')
    ax.axvline(0,color='#303943',lw=.8);ax.axvline(2,color='#c17812',ls='--',lw=1,label='Original SCALE point gate +0.002')
    ax.set_yticks(range(len(mechanism)),[r['candidate'] for r in mechanism]);ax.set_xlabel('SCALE TS-AUC gain over BL80 (x 0.001)')
    ax.set_title('Original supported SCALE slice: 132 changed series\nPositive SCALE evidence alone does not authorize promotion',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.18);ax.legend(loc='lower right',fontsize=8)
    fig.text(.01,.005,'Same fixed mechanism definition and null comparators; conditional, unadjusted paired intervals.',fontsize=8)
    save(fig,'SCALE_GAIN_INTERVALS')
    credit=ROOT/'artifacts/next4/influence/B2_LOG_ENERGY_SEMITAIL/TIME_PAIR_CREDIT_SUMMARY.csv'
    parent=json.loads((credit.parent/'RESULTS.json').read_text());assert sha(credit)==parent['files_sha256'][credit.name]
    with credit.open(encoding='utf-8-sig',newline='') as stream:times=list(csv.DictReader(stream))
    evidence[credit.relative_to(ROOT).as_posix()]=sha(credit)
    fig,ax=plt.subplots(figsize=(7.6,4.3));values=np.array([float(r['contribution_to_mean_delta']) for r in times])*1000
    ax.bar([r['time_bin'] for r in times],values,color=['#b35248' if x<0 else '#176d97' for x in values])
    ax.margins(y=.22)
    ax.axhline(0,color='#303943',lw=.8);ax.set_xlabel('Online time bin');ax.set_ylabel('Contribution to total gain (x 0.001)')
    ax.set_title('B2 exact time contribution: late gain offsets early losses',loc='left',fontweight='bold')
    for i,row in enumerate(times):
        ax.annotate(f"{100*float(row['mean_pair_mass_fraction']):.1f}% pairs",(i,values[i]),
            xytext=(0,5 if values[i]>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
    ax.grid(axis='y',alpha=.18);fig.text(.01,.005,'Contributions use actual same-time positive x negative pair mass within each fold; sum equals primary gain.',fontsize=8)
    save(fig,'B2_TIME_PAIR_CREDIT')
    null=[r for r in rows('LONG_NULL_RESULTS_NEXT4.csv') if r['candidate']=='B2_LOG_ENERGY_SEMITAIL' and r['training_scope']=='all_dev']
    if null:
        fig,axes=plt.subplots(2,2,figsize=(8,6),sharex=True,sharey=True)
        for ax,regime in zip(axes.ravel(),('N0','N1','N2','N3')):
            for alpha,style in ((.01,'--'),(.05,'-')):
                group=sorted([r for r in null if r['regime']==regime and float(r['alpha'])==alpha],key=lambda r:int(r['horizon']))
                assert len(group)==3
                for column,label,color in [('eventual_alarm_rate','B2','#176d97'),('reference_eventual_alarm_rate','BL80','#697582')]:
                    ax.plot([int(r['horizon']) for r in group],[float(r[column]) for r in group],style,marker='o',markersize=3,color=color,label=f'{label}, alpha={alpha}')
            ax.set_title(regime);ax.set_xscale('log',base=2);ax.set_xticks([1024,4096,16384],[1024,4096,16384]);ax.set_ylim(-.02,1.02);ax.grid(alpha=.18)
        axes[0,0].legend(fontsize=7);fig.subplots_adjust(bottom=.17,hspace=.35)
        fig.supxlabel('Null observation horizon',y=.045);fig.supylabel('Eventual alarm fraction')
        fig.suptitle('B2 vs matched all-DEV BL80: fixed 1,024-point calibration',fontweight='bold')
        fig.text(.01,.005,'Independent calibration and null seeds; model-specific thresholds remain fixed. This is not an anytime error guarantee.',fontsize=8)
        save(fig,'B2_LONG_NULL_ALARMS')
    receipt={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
        'snapshot_result_sha256':sha(snapshot/'RESULTS.json'),'input_sha256':evidence,
        'figure_environment_lock_sha256':sha(ROOT/'requirements-next2-figures-lock.txt'),
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,
        'new_fits':0,'new_scores':0,'synthetic_complete':result['synthetic_complete'],'synthetic_pending':result['synthetic_pending'],
        'files_sha256':{p.name:sha(p) for p in out.iterdir()}}
    (out/'RESULTS.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('NEXT4 FIGURES',out,len(receipt['files_sha256']),flush=True)


if __name__=='__main__':main()

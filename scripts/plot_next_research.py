"""Export standalone research figures from small audited derived reports."""
from pathlib import Path
import argparse
import csv as csv_module
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',default='research_checkpoint_1')
    args = parser.parse_args()
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Invalid figure tag')
    out = ROOT/'artifacts/next/figures'/args.tag
    out.mkdir(parents=True,exist_ok=True)
    if (out/'MANIFEST.json').exists():
        print('Figure snapshot already exists; use a new explicit snapshot tag')
        return
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
    sources = set()
    def csv(relative):
        path = ROOT/relative
        sources.add(path)
        with path.open(encoding='utf-8', newline='') as handle:
            return list(csv_module.DictReader(handle))
    def record(relative):
        path = ROOT/relative
        sources.add(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def save(fig,name,note):
        fig.text(.015,.012,note,fontsize=8,color='#4b5563',ha='left',va='bottom',wrap=True)
        fig.tight_layout(rect=(0,.055,1,1))
        fig.savefig(out/(name+'.png'),dpi=190)
        fig.savefig(out/(name+'.svg'))
        plt.close(fig)
    colors = {'PRIMARY':'#697586','HIGHEST_MEAN':'#a07633','V007_VARIANCE_FORECAST':'#007f78','Q005_VARIANCE_NO_MEMORY':'#7956a3'}
    labels = {'PRIMARY':'Frozen primary','HIGHEST_MEAN':'Matched 51-feature baseline','V007_VARIANCE_FORECAST':'V007: variance pair','Q005_VARIANCE_NO_MEMORY':'Q005: no memory'}
    curve = csv('artifacts/next/learning_curve/SUMMARY.csv')
    fig,axes = plt.subplots(1,2,figsize=(10,4.5),sharex=True)
    for role in ('PRIMARY','HIGHEST_MEAN','V007_VARIANCE_FORECAST'):
        group = sorted((row for row in curve if row['candidate']==role), key=lambda row: float(row['training_fraction']))
        for ax,column,title in zip(axes,('mean','min'),('Mean of five held-out folds','Worst held-out fold')):
            ax.plot([100*float(row['training_fraction']) for row in group],[float(row[column]) for row in group],marker='o',color=colors[role],label=labels[role],lw=2)
            ax.set(title=title,xlabel='Outer-training groups retained (%)',ylabel='TS-AUC',xticks=[25,50,75,100])
            ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=9,loc='lower right')
    save(fig,'learning_curves','45 new nested training-size fits; the original validation series stay fixed. Full-size points reuse frozen OOF. Secondary diagnostic, not independent validation.')
    reference = record('artifacts/next/REFERENCE_RESTORATION.json')['references']['PRIMARY']['fold_scores']
    fig,ax = plt.subplots(figsize=(9,4.6))
    ax.plot(range(5),reference,'o-',color=colors['PRIMARY'],lw=2,label=labels['PRIMARY'])
    for eid in ('V007_VARIANCE_FORECAST','Q005_VARIANCE_NO_MEMORY'):
        report = record(f'artifacts/next/experiments/{eid}/RESULTS.json')
        ax.plot(range(5),report['full']['fold_scores'],'o-',color=colors[eid],lw=2,label=labels[eid])
    ax.set(title='Original five-fold DEV results',xlabel='Frozen validation fold',ylabel='TS-AUC',xticks=range(5))
    ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=9)
    save(fig,'original_fold_scores','Full DEV: 8,000 series and 4,014,405 online observations. This plot uses no seal or reduced evaluation.')
    scopes = ['MEAN_DOMINANT','SCALE_DOMINANT','DEPENDENCE_DOMINANT','TAIL_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS']
    short = ['Mean','Scale','Dependence','Tail','Mixed','Weak / ambiguous']
    fig,axes = plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,relative,title in zip(axes,
            ('artifacts/next/forensic/V007_VARIANCE_FORECAST/PAIRED_FRACTURES.csv','artifacts/next/matched_contrast/V007_VARIANCE_FORECAST/PAIRED_FRACTURES.csv'),
            ('V007 minus frozen primary','V007 minus matched 51-feature baseline')):
        frame = {row['scope']: row for row in csv(relative)}
        tick_labels = []
        for j,(scope,label) in enumerate(zip(scopes,short)):
            item = frame['mechanism_'+scope]
            point,lo,hi = (float(item[key]) for key in ('estimate_mean_delta','ci_2_5','ci_97_5'))
            ax.plot([lo,hi],[j,j],color='#007f78',lw=2)
            ax.scatter([point],[j],s=55,facecolor='#007f78' if item['supported'].lower()=='true' else 'white',edgecolor='#007f78',zorder=3)
            tick_labels.append(f"{label} (n={int(item['break_series']):,})")
        ax.axvline(0,color='#555',lw=1,ls='--')
        ax.set(title=title,xlabel='Mean TS-AUC difference',yticks=range(len(scopes)),yticklabels=tick_labels)
        ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis()
    save(fig,'mechanism_paired_intervals','2,000 paired whole-series bootstrap draws per original fold; each mechanism includes the same no-break controls. Hollow markers: <100 break series. Intervals are conditional and not multiplicity/selection adjusted.')
    latency = []
    for eid,variant,label,color in [('PRIMARY','reference','Frozen\nprimary','#697586'),
        ('V007_VARIANCE_FORECAST','native','V007\nnative','#b88665'),
        ('V007_VARIANCE_FORECAST','fused','V007\nfused','#77aaa5'),
        ('V007_VARIANCE_FORECAST','pruned','V007\npruned','#31a297'),
        ('V007_VARIANCE_FORECAST','fast_h','V007\nfast H','#007f78'),
        ('FAST_REFERENCE','reference','Old fast\nreference','#9b8eac')]:
        r = record(f'artifacts/next/latency/research_v1/{eid}_{variant}.json')
        latency.append((label,color,r))
    fig,axes = plt.subplots(1,2,figsize=(11,4.8))
    for ax,key,title in [(axes[0],'median','Scalar-call median'),(axes[1],'amortized','Historical initialization + real stream')]:
        values = [r['point_distribution_us']['median'] if key=='median' else r['real_history_initialization']['amortized_us_per_point'] for label,color,r in latency]
        ax.bar(range(len(values)),values,color=[color for label,color,r in latency],width=.7)
        for j,value in enumerate(values):
            ax.text(j,value+.5,f'{value:.1f}',ha='center',fontsize=10)
        ax.set(title=title,ylabel='Microseconds per point',xticks=range(len(values)),xticklabels=[label for label,color,r in latency])
        ax.set_ylim(0,max(values)*1.16)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    save(fig,'latency_attribution','Same 128 held-out series and 65,541 real points; isolated local CPU measurements. All four V007 runtimes also emit exactly the same 10 million synthetic-duration outputs. No cloud/IO overhead included.')
    files = list(out.glob('*.png'))+list(out.glob('*.svg'))
    write_json(out/'MANIFEST.json',{'status':'PASS','created_utc':utc_now(),'tag':args.tag,
        'source_reports_sha256':{str(path.relative_to(ROOT)):sha256(path) for path in sorted(sources)},
        'files_sha256':{path.name:sha256(path) for path in files},'script_sha256':sha256(Path(__file__)),
        'purpose':'Interim standalone research figures; not a final handoff or session completion.'})
    print('EXPORTED RESEARCH FIGURES',out,len(files),flush=True)


if __name__=='__main__':
    main()

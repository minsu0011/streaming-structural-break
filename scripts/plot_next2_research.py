"""Standalone research figures; values and captions distinguish scientific and diagnostic evidence."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512';Y='Y001_LONG_VARIANCE_ARCH1';R='B104_LONG_VARIANCE_NO_GLR_MEMORY'
C='C205_DEPENDENCE';B='BL2_C205_DEPENDENCE_80'
OUT=ROOT/'artifacts/next2/figures';SOURCES={};FIGURES=[]


def read(path):
    path=ROOT/path;SOURCES[str(path.relative_to(ROOT))]=sha256(path)
    return pd.read_csv(path,float_precision='round_trip')


def save(fig,name,caption):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig);FIGURES.append({'figure':name,'caption':caption})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})
    paired=read('PAIRED_BOOTSTRAP_NEXT2.csv')
    chosen=[(Y,'Y001: no serial correction'),(R,'B104: no GLR + memory'),('D202_NO_EXPLICIT_GLR','D202: no explicit GLR'),
        ('D218_FULL_PIPELINE_NO_ARCH','D218: all ARCH removed'),('M2_EWMA','EWMA memory'),('S205_PARENT_PLUS_LOG_RATIOS','Added log-scale ratios'),
        ('J201_JOINT_DEPENDENCE_EWMA32','Joint dependence features'),(B,'J102 / C205 = 80 / 20')]
    fig,ax=plt.subplots(figsize=(9,4.9))
    for k,(eid,label) in enumerate(chosen):
        row=paired[(paired.candidate==eid)&(paired.reference==J)&(paired.scope=='all_time')]
        assert len(row)==1;r=row.iloc[0];point=r.estimate_mean_delta
        ax.errorbar(point,k,xerr=[[point-r.ci_2_5],[r.ci_97_5-point]],fmt='o',capsize=3,color='#147d92' if eid==B else '#596979')
    ax.axvline(0,color='#222222',lw=1);ax.axvline(.0015,color='#147d92',ls='--',lw=1,label='Performance mean-gain gate')
    ax.set_yticks(range(len(chosen)),[x[1] for x in chosen]);ax.invert_yaxis();ax.set_xlabel('Mean 5-fold DEV TS-AUC difference from J102')
    ax.set_title('Fixed-fit paired uncertainty and observed component effects');ax.legend(loc='lower left',fontsize=8)
    fig.text(.12,-.02,'2,000 whole-series paired bootstrap draws; conditional on fitted OOF predictions.\nIntervals do not adjust for model fitting, adaptive selection or multiple comparisons.',fontsize=8)
    save(fig,'01_dev_component_effects','Real DEV primary metric. Error bars are fixed-OOF whole-series paired bootstrap percentile intervals; selection uncertainty is not included. Mean gain alone does not establish promotion.')

    runtime=read('RUNTIME_PRUNING_RESULTS.csv').set_index('runtime_variant')
    variants=['J_ORIGINAL','J_COMPACT','J_USED','B_ORIGINAL','B_COMPACT','B_USED','BL80_COMPACT','BL80_SHARED_V1']
    labels=['J original','J compact','J tree-used','B original','B compact','B tree-used','80/20 compact','80/20 shared']
    fields=[('point_median_us','Scalar median (microseconds)'),('historical_init_median_us','H-init median (microseconds)'),
        ('private_per_series_numeric_bytes','Private numeric state (bytes)'),('direct_10M_seconds','Direct 10M prediction time (seconds)')]
    fig,axes=plt.subplots(2,2,figsize=(12,8));colors=['#738293','#379bb0','#166b80','#8a8792','#c1835d','#9b592f','#365ba4','#1e9273']
    for ax,(field,title) in zip(axes.ravel(),fields):
        values=runtime.loc[variants,field].to_numpy();ax.barh(labels,values,color=colors);ax.invert_yaxis();ax.set_title(title,fontsize=10)
        ax.set_xlim(0,max(values)*1.22)
        for k,value in enumerate(values):ax.text(value+max(values)*.015,k,f'{value:,.2f}' if field!='private_per_series_numeric_bytes' else f'{value:,.0f}',va='center',fontsize=8)
    fig.suptitle('Exact runtime pruning: initialization, scalar work and retained state',y=1.01)
    fig.tight_layout();save(fig,'02_runtime_and_state','Fixed 128 real DEV probes, three prefix repetitions, and one direct 10M continuous stream per runtime. Private NumPy state excludes model/shared arrays and Python/JIT overhead. Timing processes were isolated from workspace research workers; OS/power were uncontrolled.')

    cross=read('artifacts/next2/cross_background_bootstrap/PAIRED_RESULTS.csv')
    fig,axes=plt.subplots(1,3,figsize=(11.4,4.2),sharey=True)
    bgs=['N0','N1','N2','N3'];factors=[1.5,2.,.7,.5]
    for ax,eid,title in zip(axes,[J,R,B],['J102','B104','J102/C205 80/20']):
        values=cross[cross.candidate==eid].pivot(index='background',columns='variance_factor',values='ts_auc').loc[bgs,factors].to_numpy()
        im=ax.imshow(values,cmap='RdYlBu',norm=TwoSlopeNorm(vmin=.35,vcenter=.5,vmax=1.),aspect='auto')
        for i in range(4):
            for k in range(4):ax.text(k,i,f'{values[i,k]:.3f}',ha='center',va='center',color='black',fontsize=9)
        ax.set_xticks(range(4),['x1.5','x2','x0.7','x0.5']);ax.set_yticks(range(4),['Gaussian','AR','ARCH','Heavy-tail ARCH'])
        ax.set_xlabel('Variance multiplier');ax.set_title(title)
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.04,label='Synthetic TS-AUC')
    fig.suptitle('Scale-down weakness depends on the pre-change background',y=1.04)
    save(fig,'03_cross_background_scale','Prospective partition-3 cross-background supplement to the unchanged original suite: 512 matched seed pairs per cell. Frozen all-DEV models. Synthetic mechanism diagnostic only; variance factors are variances, not standard deviations.')

    long=read('artifacts/next2/age_alarm_diagnostics/LONG_NULL.csv')
    late=read('artifacts/next2/late_alarm_diagnostic/N0_ONSET_49152/ANALYSIS.csv')
    fig,axes=plt.subplots(1,3,figsize=(12.4,3.8))
    colors_beta={0.:'#3e5266',.5:'#168597',1.:'#bd6b35'}
    for beta in (0.,.5,1.):
        frame=long[(long.candidate==J)&(long.regime=='N0')&(long.alpha==.05)&(long.beta==beta)].sort_values('horizon')
        axes[0].plot(frame.horizon,frame.eventual_alarm_rate,'o-',color=colors_beta[beta],label=f'beta={beta:g}')
    axes[0].axhline(.05,color='#999999',ls=':');axes[0].set_xscale('log',base=2);axes[0].set_xlabel('Null horizon');axes[0].set_ylabel('Eventual false-alarm fraction');axes[0].set_title('J102: original null cohort');axes[0].legend(fontsize=8)
    for ax,eid,title in zip(axes[1:],[J,B],['J102','80/20 blend']):
        frame=late[(late.candidate==eid)&(late.alpha==.05)].sort_values('beta')
        x=np.arange(3);ax.bar(x-.18,frame.null_eventual_alarm_rate,.36,label='Null alarm',color='#b56b48')
        ax.bar(x+.18,frame.postbreak_first_detection_rate,.36,label='Clean detection',color='#167e91')
        ax.set_xticks(x,['0','0.5','1']);ax.set_xlabel('Fixed age exponent beta');ax.set_ylim(0,1.)
        ax.set_title(f'{title}: break at 49,152\nraw TS-AUC = {frame.raw_ts_auc.iloc[0]:.3f}',fontsize=10)
    axes[1].legend(fontsize=8);fig.tight_layout()
    save(fig,'04_null_control_and_late_detection','Gaussian background, alpha 0.05. Original 1,024-point null calibration reused. Age transformation preserves within-time ordering but can eliminate late detection. Clean first detection excludes premature alarms; raw TS-AUC does not penalize them equivalently. Late supplement has 128 matched pairs and horizon 65,536.')

    sensitivity=read('artifacts/next2/finalist_fit_sensitivity/COUNTEREXAMPLES_BY_FIT.csv')
    fig,axes=plt.subplots(1,3,figsize=(11.7,3.4),sharey=True)
    for ax,scenario,title in zip(axes,['V3','V4','D1'],['Variance x0.7','Variance x0.5','Dependence change']):
        for eid,label,color in [(J,'J102','#596979'),(R,'B104','#af6b3e'),(B,'80/20 blend','#167e91')]:
            frame=sensitivity[(sensitivity.candidate==eid)&(sensitivity.scenario==scenario)&(sensitivity.alpha==.05)].sort_values('fitted_fold')
            assert len(frame)==5
            ax.plot(frame.fitted_fold,frame.ts_auc,'o-',label=label,color=color)
        ax.axhline(.5,color='#aaaaaa',ls=':');ax.set_xticks(range(5));ax.set_xlabel('Existing full-CV fitted fold');ax.set_title(title)
    axes[0].set_ylabel('Synthetic TS-AUC');axes[-1].legend(fontsize=8);fig.tight_layout()
    save(fig,'05_synthetic_fit_sensitivity','Five existing overlapping training fits per model, evaluated on the same fixed synthetic seeds. Descriptive sensitivity only: the five fits are not independent replicates and their predictions were not averaged into a new ensemble.')
    write_json(OUT/'FIGURE_MANIFEST.json',{'status':'PASS','created_utc':utc_now(),'script_sha256':sha256(Path(__file__)),
        'sources_sha256':SOURCES,'figures':FIGURES,'files_sha256':{p.name:sha256(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf')}})
    print('NEXT2 RESEARCH FIGURES',len(FIGURES),flush=True)


if __name__=='__main__':main()

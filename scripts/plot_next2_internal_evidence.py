"""Mechanistic figures with separate current evidence, path memory and predictive attribution."""
from pathlib import Path
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    out=ROOT/'artifacts/next2/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    source=ROOT/'artifacts/next2/null_evidence/FEATURE_GROWTH.csv';frame=pd.read_csv(source,float_precision='round_trip')
    j='J102_SERIAL_VARIANCE_512';fig,axes=plt.subplots(1,3,figsize=(11.8,3.6))
    for ax,suffix,title in zip(axes,['raw_energy_glr_max','raw_energy_memory_max','raw_energy_bayes025'],
            ['Current fixed-window GLR maximum','Cumulative maximum memory','Current Bayes approximation']):
        for bg,label,color in [('N0','Gaussian','#597185'),('N2','ARCH','#168497'),('N3','Heavy-tail ARCH','#b76a3d')]:
            values=frame[(frame.candidate==j)&(frame.background==bg)&(frame.feature=='serial_variance__'+suffix)&(frame.quantity=='current')].sort_values('horizon')
            assert len(values)==3
            ax.plot(values.horizon,values['quantile_0.5'],'o-',label=label,color=color)
        ax.set_title(title,fontsize=10);ax.set_xscale('log',base=2);ax.set_xlabel('Null horizon');ax.set_ylabel('Across-series median')
    axes[0].legend(fontsize=8);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/('06_current_evidence_and_null_memory.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    shap_source=ROOT/'artifacts/next2/synthetic_attribution/PAIRED_GROUP_DELTAS.csv';shap=pd.read_csv(shap_source,float_precision='round_trip')
    groups=['LEGACY_ABCD','ENERGY_LOCAL','ENERGY_GLR','ENERGY_BAYES','ENERGY_MEMORY','FORECAST_LOCAL','FORECAST_GLR','FORECAST_BAYES','FORECAST_MEMORY']
    fig,axes=plt.subplots(1,3,figsize=(12.2,4.5),sharey=True)
    for ax,scenario,title in zip(axes,['V3','V4','D1'],['Variance x0.7','Variance x0.5','Dependence change']):
        subset=shap[(shap.candidate==j)&(shap.scenario==scenario)&(shap.slot=='post511')].set_index('metric')
        values=subset.loc[groups,'paired_mean_delta'].to_numpy();margin=float(subset.loc['raw_margin','paired_mean_delta'])
        ax.barh(range(len(groups)),values,color=['#177f91' if value>=0 else '#b96941' for value in values])
        ax.axvline(0,color='#555555',lw=1);ax.set_yticks(range(len(groups)),[g.replace('_',' ').title() for g in groups])
        ax.set_title(f'{title}\nmean raw-margin change = {margin:+.3f}',fontsize=10)
        ax.set_xlabel('Mean intervention - paired control contribution')
    axes[0].invert_yaxis();fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/('07_predictive_attribution_counterexamples.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    write_json(out/'INTERNAL_EVIDENCE_FIGURE_MANIFEST.json',{'status':'PASS','created_utc':utc_now(),
        'script_sha256':sha256(Path(__file__)),'sources_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in (source,shap_source)},
        'captions':{'06_current_evidence_and_null_memory':'Original fixed partition-2 null cohorts, 512 series per background, all-DEV J102. Medians of different transformed statistics have different scales. Every replayed prediction matched the saved array. This is not a calibrated sequential likelihood or causal estimate.',
            '07_predictive_attribution_counterexamples':'All 512 matched pairs from the original suite, J102 all-DEV model, 511 observations after the break. TreeSHAP components add to a raw-margin prediction difference. Correlated feature attribution is predictive explanation, not causal ablation. The fixed probability blend has no additive-logit attribution claim.'},
        'files_sha256':{p.name:sha256(p) for p in out.iterdir() if p.name.startswith(('06_','07_')) and p.suffix in ('.png','.pdf')}})
    print('NEXT2 INTERNAL EVIDENCE FIGURES 2',flush=True)


if __name__=='__main__':main()

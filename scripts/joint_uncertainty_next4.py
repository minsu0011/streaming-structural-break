"""A conditional joint look at all closed NEXT4 full candidates; no new draws."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.joint_uncertainty import describe_joint


def main():
    pp=BASE/'diagnostic_policy/JOINT_UNCERTAINTY_POLICY.json';policy=read_json(pp)
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    out=BASE/'joint_uncertainty';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(pp),'new_draws':0,'new_fits':0},immutable=True)
    columns=[];records=[];evidence={};baseline=None
    for candidate in policy['candidates']:
        directory=BASE/'bootstrap'/candidate;receipt=read_json(directory/'RESULTS.json');path=directory/'AGGREGATE_DRAWS.npz'
        assert receipt['status']=='PASS' and receipt['policy_sha256']==policy['original_research_policy_sha256']
        assert digest(path)==receipt['files_sha256'][path.name]
        start=read_json(directory/'START.json')
        assert start['seed_base']==20260908 and start['draws']==2000
        with np.load(path,allow_pickle=False) as arrays:
            new=arrays['candidate_fold_scores'];old=arrays['reference_fold_scores']
            assert new.shape==old.shape==(2001,5)
            if baseline is None:baseline=old.copy()
            else:np.testing.assert_array_equal(old,baseline)
            delta=(new-old).mean(axis=1);columns.append(delta)
        full_path=BASE/'experiments'/candidate/'full/RESULTS.json';full=read_json(full_path)
        np.testing.assert_allclose(delta[0],full['mean_delta'],atol=1e-14,rtol=0)
        screen=read_json(full_path.parent.parent/'screen/RESULTS.json')
        records.append({'candidate':candidate,'family':full['family'],'screen_delta':screen['mean_delta'],
            'full_delta':full['mean_delta'],'individual_ci_lower':receipt['paired_all_time']['ci_2_5'],
            'individual_ci_upper':receipt['paired_all_time']['ci_97_5'],'standalone_mean':full['standalone_mean'],
            'original_primary_admission':receipt['performance_primary_and_bootstrap_admission']})
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    result=describe_joint(np.asarray(columns).T)
    for i,row in enumerate(records):
        for key in ('bootstrap_standard_error','one_sided_95_joint_lower','empirical_centered_max_tail_fraction','bootstrap_best_frequency'):
            row[key]=float(result[key][i])
    frame=pd.DataFrame(records);frame.to_csv(out/'CONDITIONAL_JOINT_DIAGNOSTIC.csv',index=False)
    pd.DataFrame(result['correlation'],index=policy['candidates'],columns=policy['candidates']).to_csv(out/'PAIRED_GAIN_CORRELATION.csv')
    plot_created=False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ordered=frame.sort_values('full_delta');y=np.arange(len(ordered));fig,ax=plt.subplots(figsize=(10,7))
        ax.errorbar(ordered.full_delta,y,xerr=np.vstack([ordered.full_delta-ordered.individual_ci_lower,
            ordered.individual_ci_upper-ordered.full_delta]),fmt='o',color='#245c83',ecolor='#95adbd',capsize=3)
        ax.axvline(0,color='#555555',linewidth=1);ax.axvline(.0013,color='#b95324',linestyle='--',label='Frozen performance gain gate')
        ax.set_yticks(y,ordered.candidate);ax.set_xlabel('Primary five-fold TS-AUC gain vs frozen BL80')
        ax.set_title('NEXT4 fixed OOF: individual paired 95% intervals')
        ax.text(0,-.14,'Conditional on fitted OOF; not adjusted for prior adaptive research. No seal data.',transform=ax.transAxes,fontsize=9)
        ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right');fig.tight_layout()
        fig.savefig(out/'PRIMARY_GAIN_INTERVALS.png',dpi=170,bbox_inches='tight');fig.savefig(out/'PRIMARY_GAIN_INTERVALS.svg',bbox_inches='tight');plt.close(fig)
        plot_created=True
    except ModuleNotFoundError:pass
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidates':policy['candidates'],
        'candidate_count':len(records),'original_draws_reused':2000,'new_draws':0,'new_fits':0,'seal_rows':0,
        'centered_max_95_critical':result['centered_max_95_critical'],
        'correlation_participation_ratio':result['correlation_participation_ratio'],
        'positive_one_sided_joint_lower_count':int((frame.one_sided_95_joint_lower>0).sum()),
        'plot_created':plot_created,'original_gates_overridden':False,
        'scope':'Single-step empirical centered-maximum diagnostic on a fixed snapshot of 14 learned OOF candidates. It is not the Romano-Wolf stepdown algorithm, not a guarantee of unconditional family-wise control, and does not account for training or earlier adaptive NEXT/NEXT2/NEXT3 selection. Participation ratio is descriptive dependence, not a candidate budget.',
        'method_context_source':'https://www.econ.uzh.ch/dam/jcr%3Affffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf',
        'source_and_input_evidence_sha256':evidence,
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    progress('Joint uncertainty diagnosis completed','All closed full-CV recipes compared with exactly the original paired bootstrap draws; no selection adjustment claim for past research')
    print('NEXT4 JOINT UNCERTAINTY COMPLETE',len(records),flush=True)


if __name__=='__main__':main()

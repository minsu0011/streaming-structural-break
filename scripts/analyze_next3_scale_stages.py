"""Pair-matched information/range diagnostics from current channel to frozen trees."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next3.stage_trace import NAMES,replay_stage_trace,orientation_for_decrease
from src.next2.counterexamples import policy as suite_policy
from src.next2.cross_background_scale import series
from src.next2.shared_inference import load_artifact
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor


def rank_auc(control,changed):
    values=np.r_[control,changed]
    ranks=rankdata(values,method='average')
    n=len(control)
    return float((ranks[n:].sum()-n*(n+1)/2)/(n*n))


def main():
    hypothesis_path=BASE/'hypotheses/F002_SCALE_STAGE_TRACE.json'
    hypothesis=read_json(hypothesis_path)
    assert digest(ROOT/'src/next3/stage_trace.py')==hypothesis['implementation_sha256']
    out=BASE/'scale_stages';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():raise RuntimeError('Do not overwrite completed stage analysis')
    model,descriptor=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    prepared=PreparedSharedBlendPredictor(model,resident=False)
    lock=suite_policy();n=hypothesis['seed_pairs'];windows=hypothesis['post_windows']
    rng=np.random.default_rng(hypothesis['bootstrap_seed'])
    counts=np.asarray([np.bincount(row,minlength=n) for row in rng.integers(0,n,(2000,n))],dtype=np.float64)/n
    rows=[];files={};prefix_checks=0
    for background in hypothesis['backgrounds']:
        for factor in hypothesis['variance_factors']:
            cell=out/f'{background}_{factor}'
            if (cell/'RESULTS.json').exists():
                old=read_json(cell/'RESULTS.json');assert old['status']=='PASS'
                for name,expected in old['files_sha256'].items():assert digest(cell/name)==expected
                rows.extend(pd.read_csv(cell/'SUMMARY.csv').to_dict('records'))
                prefix_checks+=old['prefix_checks'];continue
            cell.mkdir(parents=True,exist_ok=True)
            values=np.empty((2,n,len(windows),len(NAMES)),dtype=np.float64)
            boundary=np.zeros_like(values)
            for seed in range(n):
                h0,o0,t0=series(lock,background,factor,seed,control=True)
                h1,o1,tau=series(lock,background,factor,seed)
                np.testing.assert_array_equal(h0,h1)
                a=replay_stage_trace(o0,prepared.new_state(h0).call_args)
                b=replay_stage_trace(o1,prepared.new_state(h1).call_args)
                np.testing.assert_array_equal(a[:tau],b[:tau]);prefix_checks+=1
                for j,horizon in enumerate(windows):
                    values[0,seed,j]=a[tau:tau+horizon].mean(axis=0)
                    values[1,seed,j]=b[tau:tau+horizon].mean(axis=0)
                    for group,trace in enumerate((a,b)):
                        window=trace[tau:tau+horizon]
                        for channel in (0,13):
                            boundary[group,seed,j,channel]=np.mean(np.abs(window[:,channel])>=12.)
                        for k in (4,5,6,17,18,19):
                            boundary[group,seed,j,k]=np.mean(window[:,k]==0.)
                        for k in range(30,38):
                            boundary[group,seed,j,k]=np.mean(window[:,k]<=-80.)
                if (seed+1)%64==0:
                    progress('B2 downstream scale trace',f'{background} variance {factor}: {seed+1}/{n} paired seeds',experiment='F002_SCALE_STAGE_TRACE')
                    print('NEXT3 SCALE STAGES',background,factor,seed+1,flush=True)
            cell_rows=[]
            for j,horizon in enumerate(windows):
                difference=values[1,:,j]-values[0,:,j]
                draws=counts@difference
                for k,name in enumerate(NAMES):
                    control,changed=values[0,:,j,k],values[1,:,j,k]
                    sign=orientation_for_decrease(name)
                    row={'background':background,'variance_factor':factor,'post_window':horizon,'stage':name,'seed_pairs':n,
                        'mean_control':float(control.mean()),'mean_changed':float(changed.mean()),
                        'mean_paired_delta':float(difference[:,k].mean()),'ci_2_5':float(np.quantile(draws[:,k],.025)),
                        'ci_97_5':float(np.quantile(draws[:,k],.975)),
                        'control_q05':float(np.quantile(control,.05)),'control_q95':float(np.quantile(control,.95)),
                        'changed_q05':float(np.quantile(changed,.05)),'changed_q95':float(np.quantile(changed,.95)),
                        'fixed_orientation':sign,'paired_correct_direction_fraction':float(np.mean(sign*difference[:,k]>0)+.5*np.mean(difference[:,k]==0)),
                        'between_seed_rank_auc_fixed_orientation':rank_auc(sign*control,sign*changed),
                        'equal_pair_fraction':float(np.mean(control==changed)),
                        'control_samplewise_boundary_fraction':float(boundary[0,:,j,k].mean()) if k in (0,13,4,5,6,17,18,19) or 30<=k<38 else None,
                        'changed_samplewise_boundary_fraction':float(boundary[1,:,j,k].mean()) if k in (0,13,4,5,6,17,18,19) or 30<=k<38 else None,
                        'directional_Bayes_floor_fraction':float(np.mean(np.r_[control,changed]<=-79.999)) if ('negative_bayes' in name or 'positive_bayes' in name) else None}
                    cell_rows.append(row)
            pd.DataFrame(cell_rows).to_csv(cell/'SUMMARY.csv',index=False)
            # Synthetic per-seed window summaries, not observations or real OOF.
            np.save(cell/'WINDOW_SUMMARIES.npy',values,allow_pickle=False)
            atomic_json(cell/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'prefix_checks':n,
                'hypothesis_sha256':digest(hypothesis_path),'new_fits':0,
                'files_sha256':{name:digest(cell/name) for name in ('SUMMARY.csv','WINDOW_SUMMARIES.npy')}},immutable=True)
            rows.extend(cell_rows)
    pd.DataFrame(rows).to_csv(out/'STAGE_SEPARATION.csv',index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'new_fits':0,'seal_rows':0,
        'hypothesis_sha256':digest(hypothesis_path),'script_sha256':digest(Path(__file__)),
        'stage_count':len(NAMES),'pair_prefix_checks':prefix_checks,'summary_rows':len(rows),
        'files_sha256':{'STAGE_SEPARATION.csv':digest(out/'STAGE_SEPARATION.csv')},
        'synthetic_only':True,'feature_rank_auc_is_not_competition_TS_AUC':True,
        'interpretation':'Paired current-to-tree stage window separation with fixed orientation; directional Bayes decomposition is diagnostic and does not modify frozen prediction.'},immutable=True)
    print('NEXT3 SCALE STAGE ANALYSIS PASS',len(rows),flush=True)


if __name__=='__main__':main()

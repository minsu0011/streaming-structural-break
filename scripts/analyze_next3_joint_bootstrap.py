"""Joint conditional bootstrap envelope for the finite completed full-CV pool."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.evaluation import load_full_oof
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC
from src.validation.group_bootstrap import draw_group_multiplicities


def main():
    set_num_threads(2);hyp=BASE/'hypotheses/D002_JOINT_CONDITIONAL_UNCERTAINTY.json';registration=read_json(hyp)
    for name,sha in registration['source_sha256'].items():assert digest(ROOT/name)==sha
    out=BASE/'selection_uncertainty';out.mkdir(parents=True,exist_ok=False)
    data=prepare_dev(ROOT);metadata=data.rows()
    candidates=[p.parent.parent.name for p in sorted((BASE/'experiments').glob('*/full/RESULTS.json')) if read_json(p)['status']=='COMPLETE']
    assert candidates
    baseline=ROOT/'artifacts/next2/forensic/BL2_C205_DEPENDENCE_80';base_receipt=read_json(baseline/'RESULTS.json')
    ref=[]
    for fold in range(5):
        name=f'fold_{fold}_all_time.npy';assert digest(baseline/name)==base_receipt['files_sha256'][name]
        ref.append(np.load(baseline/name,allow_pickle=False))
    ref=np.asarray(ref).T.mean(axis=1);assert len(ref)==2001
    atomic_json(out/'POLICY.json',{'created_utc':utc_now(),'hypothesis_sha256':digest(hyp),
        'candidates':candidates,'pool_rule':'Every completed full-CV NEXT3 recipe, including full-admission failures; failed screens without full OOF are absent.',
        'reference_receipt_sha256':digest(baseline/'RESULTS.json'),'replicates':2000,'paired_original_seeds':True},immutable=True)
    vectors=[];sources={}
    for candidate in candidates:
        forensic=BASE/'forensic'/candidate
        if (forensic/'RESULTS.json').exists():
            r=read_json(forensic/'RESULTS.json');assert r['status']=='PASS';columns=[]
            for fold in range(5):
                name=f'fold_{fold}_all_time.npy';assert digest(forensic/name)==r['files_sha256'][name]
                columns.append(np.load(forensic/name,allow_pickle=False))
            values=np.asarray(columns).T.mean(axis=1)
            sources[candidate]={'reused_conditional_receipt_sha256':digest(forensic/'RESULTS.json')}
        else:
            _,_,meta,prediction,source=load_full_oof(data,candidate)
            for key in metadata:np.testing.assert_array_equal(metadata[key],meta[key])
            columns=[]
            for fold in range(5):
                rows=np.flatnonzero(metadata['fold']==fold)
                code,counts=draw_group_multiplicities(metadata['dataset_id'][rows],2000,registration['seed_base']+fold)
                values=PreparedAUC.create(metadata['target'][rows],prediction[rows],metadata['time_online'][rows],code,counts.shape[1]).evaluate(counts)
                columns.append(values)
            values=np.asarray(columns).T.mean(axis=1);sources[candidate]=source
        difference=values-ref;reported=read_json(BASE/'experiments'/candidate/'full/RESULTS.json')['mean_delta']
        np.testing.assert_allclose(difference[0],reported,rtol=0,atol=1e-14)
        vectors.append(difference)
        progress('Joint conditional uncertainty',f'{candidate}: original paired series draws aligned; no new fit',experiment='D002_JOINT_CONDITIONAL_UNCERTAINTY')
        print('NEXT3 JOINT BOOTSTRAP',candidate,flush=True)
    matrix=np.asarray(vectors).T
    assert matrix.shape==(2001,len(candidates)) and np.isfinite(matrix).all()
    estimate=matrix[0];standard_error=matrix[1:].std(axis=0,ddof=1)
    assert (standard_error>0).all()
    standardized=(matrix[1:]-estimate)/standard_error
    maximum=np.max(np.abs(standardized),axis=1)
    critical=float(np.quantile(maximum,.95))
    rows=[]
    for column,candidate in enumerate(candidates):
        lower=float(estimate[column]-critical*standard_error[column]);upper=float(estimate[column]+critical*standard_error[column])
        rows.append({'candidate':candidate,'reference':'BL2_C205_DEPENDENCE_80','estimate_delta':float(estimate[column]),
            'conditional_percentile_ci_2_5':float(np.quantile(matrix[1:,column],.025)),
            'conditional_percentile_ci_97_5':float(np.quantile(matrix[1:,column],.975)),
            'conditional_standard_error':float(standard_error[column]),'joint_max_abs_standardized_quantile95':critical,
            'joint_conditional_envelope_lower':lower,'joint_conditional_envelope_upper':upper,
            'positive_under_joint_conditional_envelope':lower>0,'performance_effect_threshold':.0015,
            'effect_size_gate_pass':bool(estimate[column]>=.0015),'selection_adjusted':False})
    pd.DataFrame(rows).to_csv(out/'CONDITIONAL_POOL_ENVELOPE.csv',index=False)
    np.savez_compressed(out/'AGGREGATE_CONTRAST_DRAWS.npz',contrasts=matrix,candidates=np.asarray(candidates),max_abs_standardized_error=maximum)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'hypothesis_sha256':digest(hyp),
        'candidate_count':len(candidates),'replicates':2000,'new_fits':0,'seal_rows':0,'critical_value':critical,
        'sources':sources,'selection_adjusted':False,'scientific_promotion_authority':False,
        'limitation':'This is a descriptive simultaneous envelope conditional on the fitted observed candidate pool and original resampling draws. It does not correct adaptive candidate generation, screen survival, base fitting, prior-session DEV reuse or later model selection; no unconditional 95% familywise coverage is claimed.',
        'files_sha256':{name:digest(out/name) for name in ('POLICY.json','CONDITIONAL_POOL_ENVELOPE.csv','AGGREGATE_CONTRAST_DRAWS.npz')}},immutable=True)
    print('NEXT3 JOINT CONDITIONAL ENVELOPE COMPLETE',len(candidates),critical,flush=True)


if __name__=='__main__':main()

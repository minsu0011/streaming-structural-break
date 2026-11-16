"""Preregistered paired whole-series uncertainty of every completed full recipe."""
from pathlib import Path
import sys
import argparse
import hashlib
import numpy as np
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.frozen import fold_predictions
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    source=BASE/'experiments'/args.candidate/'full';result=read_json(source/'RESULTS.json')
    if result['status']!='COMPLETE' or result['scope']!='full':raise ValueError('Completed full five-fold result required')
    out=BASE/'bootstrap'/args.candidate;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':args.candidate,
        'source_result_sha256':digest(source/'RESULTS.json'),'policy_sha256':digest(policy_path),
        'seed_base':policy['bootstrap']['seed'],'draws':policy['bootstrap']['draws'],
        'source_sha256':digest(Path(__file__)),'scope':'All-time fixed-OOF whole-series paired uncertainty only'},immutable=True)
    set_num_threads(2);data=prepare_dev(ROOT);metadata=data.rows();new=[];old=[];files={}
    for fold in range(5):
        receipt=read_json(source/f'fold_{fold}.json');rows=np.flatnonzero(metadata['fold']==fold)
        for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy'):
            if digest(source/name)!=receipt['files_sha256'][name]:raise RuntimeError('Candidate OOF bytes changed')
        np.testing.assert_array_equal(np.load(source/f'fold_{fold}_global_rows.npy',allow_pickle=False),rows)
        prediction=np.load(source/f'fold_{fold}.npy',allow_pickle=False)
        ids,target,times=(metadata[k][rows] for k in ('dataset_id','target','time_online'))
        code,counts=draw_group_multiplicities(ids,policy['bootstrap']['draws'],policy['bootstrap']['seed']+fold)
        reference_dir=BASE/'bootstrap/frozen_BL80';reference_dir.mkdir(parents=True,exist_ok=True)
        reference_path=reference_dir/f'fold_{fold}.npy';identity_path=reference_dir/f'fold_{fold}.json'
        identity={'policy_sha256':digest(policy_path),'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
            'rows_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),'fold':fold,'draws':policy['bootstrap']['draws']}
        if reference_path.exists():
            prior=read_json(identity_path)
            if prior['identity']!=identity or digest(reference_path)!=prior['sha256']:raise RuntimeError('Reference bootstrap cache identity mismatch')
            reference_values=np.load(reference_path,allow_pickle=False)
        else:
            baseline,baseline_sources=fold_predictions(data,fold,'full',rows)
            progress('NEXT4 paired reference bootstrap','BL80 fold '+str(fold)+' with frozen counts; compute once for all candidates')
            reference_values=PreparedAUC.create(target,baseline['BL80'],times,code,counts.shape[1]).evaluate(counts)
            np.testing.assert_allclose(reference_values[0],ts_auc(target,baseline['BL80'],times),rtol=0,atol=1e-14)
            np.save(reference_path,reference_values,allow_pickle=False)
            atomic_json(identity_path,{'status':'PASS','created_utc':utc_now(),'identity':identity,
                'sha256':digest(reference_path),'frozen_OOF_sources':baseline_sources,'new_fits':0},immutable=True)
        progress('NEXT4 paired candidate bootstrap',args.candidate+' fold '+str(fold)+'; 2000 fixed whole-series draws')
        values=PreparedAUC.create(target,prediction,times,code,counts.shape[1]).evaluate(counts)
        np.testing.assert_allclose(values[0],receipt['ts_auc'],rtol=0,atol=1e-14)
        path=out/f'fold_{fold}.npy';np.save(path,values,allow_pickle=False)
        files[path.name]=digest(path);new.append(values);old.append(reference_values)
        print('NEXT4 BOOTSTRAP',args.candidate,fold,flush=True)
    candidate=np.asarray(new).T;reference=np.asarray(old).T;summary=paired_summary(reference,candidate)
    output=out/'AGGREGATE_DRAWS.npz';np.savez_compressed(output,candidate_fold_scores=candidate,reference_fold_scores=reference)
    files[output.name]=digest(output)
    gate=policy['performance'];numeric=result['performance_numeric_admission']
    admitted=bool(numeric and summary['ci_2_5']>0 and summary['probability_delta_positive']>=gate['bootstrap_probability_min'])
    report={'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,'family':result['family'],
        'paired_all_time':summary,'draws':2000,'folds':5,'unit':'whole_series_paired_within_original_fold',
        'performance_primary_and_bootstrap_admission':admitted,'promotion':False,
        'remaining_required_for_performance':['ALT_1/2/3','H-regime','mechanism/time fracture','causal/deterministic/runtime audit'] if admitted else [],
        'new_fits':0,'seal_rows':0,'conditional_fitted_OOF_not_selection_adjusted':True,
        'source_result_sha256':digest(source/'RESULTS.json'),'policy_sha256':digest(policy_path),
        'source_sha256':digest(Path(__file__)),'files_sha256':files}
    atomic_json(out/'RESULTS.json',report,immutable=True);data.guard.check_state()
    progress('NEXT4 paired uncertainty completed',args.candidate+' primary+bootstrap admission: '+str(admitted))
    print('NEXT4 PAIRED RESULT',args.candidate,summary['estimate_mean_delta'],summary['ci_2_5'],summary['ci_97_5'],flush=True)


if __name__=='__main__':main()

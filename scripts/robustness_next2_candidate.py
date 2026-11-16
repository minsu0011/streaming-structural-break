"""Twenty fixed ALT/H-regime refits compared with the corresponding J102 fits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import sys
import json
import argparse
import time
import gc
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.io import design,fit,model_class
from src.next2.evaluation import load_full_oof
from src.next2.session import checkpoint
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    args=parser.parse_args()
    data=prepare_dev(ROOT)
    c,prior,_,_,oof_sources=load_full_oof(data,args.candidate)
    if prior['status'] not in ('FINALIST_DIAGNOSTICS_PENDING','COMPLEMENT_ELIGIBLE_FIXED_BLEND'):
        raise RuntimeError('Fixed finalist or complement admission required before stress refits')
    outroot=ROOT/'artifacts/next2/robustness'/args.candidate
    outroot.mkdir(parents=True,exist_ok=True)
    policy=json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    regime_dir=ROOT/'artifacts/next/current_forensic'
    lock=json.loads((regime_dir/'REGIME_ASSIGNMENT_LOCK.json').read_text(encoding='utf-8'))
    assert sha256(regime_dir/'DEV_SIGNATURES.parquet')==lock['files_sha256']['DEV_SIGNATURES.parquet']
    regimes=pd.read_parquet(regime_dir/'DEV_SIGNATURES.parquet',columns=['dataset_id','historical_regime']).set_index('dataset_id').historical_regime.to_dict()
    data.guard.admit(regimes,purpose='NEXT2 H-only regime stress assignment')
    splits=[(f'ALT_{j+1}',{int(k):v for k,v in item['fold_by_dataset_id'].items()}) for j,item in enumerate(policy['alternates'])]
    splits.append(('LEAVE_ONE_H_REGIME_OUT',regimes))
    x,metadata,names,sources,global_rows=design(data,c,'full')
    np.testing.assert_array_equal(global_rows,np.arange(len(global_rows)))
    order=np.concatenate([np.flatnonzero(metadata['fold']==k) for k in range(5)])
    identity={'candidate':c,'feature_sources':sources,'oof_sources':oof_sources,
        'next2_policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'old_split_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'regime_lock_sha256':sha256(regime_dir/'REGIME_ASSIGNMENT_LOCK.json'),
        'script_sha256':sha256(Path(__file__)),'nomination_status':prior['status'],
        'order':'stress fold then original fold then dataset ID then online time'}
    if (outroot/'POLICY.json').exists():
        assert json.loads((outroot/'POLICY.json').read_text(encoding='utf-8'))==identity
    else:
        write_json(outroot/'POLICY.json',identity)
    summaries=[]
    for split,assignment in splits:
        code=pd.Series(metadata['dataset_id']).map(assignment).to_numpy()
        assert np.isfinite(code).all() and set(np.unique(code))==set(range(5))
        partitions=[order[code[order]==fold] for fold in range(5)]
        out=outroot/split
        out.mkdir(exist_ok=True)
        scores,reference_scores=[],[]
        for fold in range(5):
            valid=partitions[fold]
            train=np.concatenate([partitions[k] for k in range(5) if k!=fold])
            refdir=ROOT/'artifacts/next/robustness/candidates/J102_SERIAL_VARIANCE_512'/split
            refreceipt=json.loads((refdir/f'fold_{fold}.json').read_text(encoding='utf-8'))
            for name,digest in refreceipt['files_sha256'].items():
                assert sha256(refdir/name)==digest
            np.testing.assert_array_equal(valid,np.load(refdir/f'fold_{fold}_global_rows.npy',allow_pickle=False))
            reference_score=refreceipt['ts_auc']
            path=out/f'fold_{fold}.json'
            if path.exists():
                r=json.loads(path.read_text(encoding='utf-8'))
                assert r['status']=='PASS'
                for name,digest in r['files_sha256'].items():
                    assert sha256(out/name)==digest
                score=r['ts_auc']
            else:
                tick=time.perf_counter()
                model,prediction,detail=fit(data,c,x,metadata,names,train,valid)
                model.save(out/f'fold_{fold}.joblib')
                loaded=model_class(c).load(out/f'fold_{fold}.joblib')
                np.testing.assert_array_equal(loaded.predict(x[valid[:2048]]).astype(np.float32),prediction[:2048])
                sid=metadata['dataset_id'][valid[0]]
                h,o,_=data.get(sid); state=loaded.make_state(h)
                live=np.asarray([loaded.predict_one(state.update(point)) for point in o],dtype=np.float32)
                np.testing.assert_array_equal(live,prediction[metadata['dataset_id'][valid]==sid])
                score=ts_auc(metadata['target'][valid],prediction,metadata['time_online'][valid])
                np.save(out/f'fold_{fold}.npy',prediction,allow_pickle=False)
                np.save(out/f'fold_{fold}_global_rows.npy',valid,allow_pickle=False)
                files=[f'fold_{fold}.joblib',f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy']
                r={'status':'PASS','candidate':args.candidate,'split_name':split,'fold':fold,'ts_auc':score,
                    'reference_ts_auc':reference_score,'delta':score-reference_score,'engineering':detail,
                    'elapsed_seconds':time.perf_counter()-tick,'seal_rows':0,
                    'files_sha256':{name:sha256(out/name) for name in files}}
                write_json(path,r)
                del loaded,model,prediction
                gc.collect()
            scores.append(score); reference_scores.append(reference_score)
            print('NEXT2 STRESS',args.candidate,split,fold,score,flush=True)
        summary={'status':'PASS','split_name':split,'fold_scores':scores,'reference_fold_scores':reference_scores,
            'mean':float(np.mean(scores)),'median':float(np.median(scores)),'worst':min(scores),
            'mean_delta':float(np.mean(scores)-np.mean(reference_scores)),
            'worst_delta':min(scores)-min(reference_scores),
            'minimum_matched_fold_delta':min(a-b for a,b in zip(scores,reference_scores))}
        write_json(out/'RESULTS.json',summary)
        summaries.append(summary)
        checkpoint('NEXT2 stress '+args.candidate+'/'+split,['Remaining fixed splits','Apply unchanged gates'],
            artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    write_json(outroot/'RESULTS.json',{'status':'PASS','candidate':args.candidate,'completed_refits':20,
        'created_utc':utc_now(),'summaries':summaries,'seal_rows':0,'new_reduced_uses':0})


if __name__=='__main__':
    main()

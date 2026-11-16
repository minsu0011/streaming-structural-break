"""Versioned all-DEV fit and independent raw-iterator reproduction fingerprints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import sys
import json
import argparse
import hashlib
import time
import gc
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.io import design,model_class
from src.next2.final_fitting import fit_all_development,source_hash
from src.next2.training import train_frozen_candidate,training_hash
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now
from scripts.refit_next_candidate_dev import guarded_raw_iterator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--source',choices=['cache','iterator'],required=True)
    parser.add_argument('--tag',default='v1')
    parser.add_argument('--compare-to')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Invalid refit tag')
    data=prepare_dev(ROOT)
    candidate,prior,_,_,oof_sources=load_full_oof(data,args.candidate)
    if prior['status'] not in ('FINALIST_DIAGNOSTICS_PENDING','COMPLEMENT_ELIGIBLE_FIXED_BLEND'):
        raise RuntimeError('Only numerically eligible NEXT2 candidates may receive complete-DEV refits')
    out=ROOT/'artifacts/next2/refits'/args.candidate/(args.source+'_'+args.tag)
    out.mkdir(parents=True,exist_ok=True)
    identity={'candidate':candidate,'source':args.source,'tag':args.tag,'oof_sources':oof_sources,
        'script_sha256':sha256(Path(__file__)),'final_fitting_sha256':source_hash(),
        'training_sha256':training_hash(),'compare_to':args.compare_to}
    spec={'status':'FROZEN_NEXT2_RESEARCH_REFIT','candidate':candidate,
        'development_ids':sorted(data.guard.dev_ids),'known_training_ids':sorted(data.guard.split.mapping),
        'split_sha256':data.guard.split.digest}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text())!=identity:
            raise RuntimeError('Refit policy changed; use an explicit revision')
    else:
        write_json(out/'POLICY.json',identity)
        write_json(out/'SPEC.json',spec)
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text())
        if r['status']!='PASS' or sha256(out/'model.joblib')!=r['model_sha256'] or sha256(out/'TRAINING.json')!=r['training_receipt_sha256']:
            raise RuntimeError('Refit artifact changed')
        print('NEXT2 REFIT RETAINED',args.candidate,args.source,flush=True)
        return
    tick=time.perf_counter()
    if not (out/'TRAINING.json').exists():
        if (out/'model.joblib').exists():
            raise RuntimeError('Partial model requires explicit review')
        if args.source=='iterator':
            receipt=train_frozen_candidate(guarded_raw_iterator(data),out,spec,root=ROOT)
        else:
            x,meta,names,sources,rows=design(data,candidate)
            np.testing.assert_array_equal(rows,np.arange(len(rows)))
            model,detail=fit_all_development(x,meta,candidate,data.guard,names)
            model.save(out/'model.joblib')
            receipt={'status':'PASS','candidate':candidate,'training':detail,'model_sha256':sha256(out/'model.joblib'),
                'feature_sources':sources,'input_origin':'VERIFIED_DEV_FEATURE_CACHE','created_utc':utc_now()}
            write_json(out/'TRAINING.json',receipt)
            del x,meta,model
            gc.collect()
    else:
        receipt=json.loads((out/'TRAINING.json').read_text())
        if receipt['candidate']!=candidate or receipt['model_sha256']!=sha256(out/'model.joblib'):
            raise RuntimeError('Incomplete receipt identity differs')
    loaded=model_class(candidate).load(out/'model.joblib')
    x,meta,names,sources,rows=design(data,candidate)
    prediction=np.asarray(loaded.predict(x),dtype=np.float32)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite reproduction fingerprints')
    fingerprint=hashlib.sha256(prediction.tobytes()).hexdigest()
    comparison=None
    if args.compare_to:
        folder=ROOT/args.compare_to
        other=json.loads((folder/'RESULTS.json').read_text())
        details=json.loads((folder/'TRAINING.json').read_text())
        if other['status']!='PASS' or details['candidate']!=candidate:
            raise RuntimeError('Exact candidate reproduction required')
        checks={'model_bytes_equal':sha256(out/'model.joblib')==other['model_sha256'],
            'all_dev_fingerprint_equal':fingerprint==other['all_dev_prediction_sha256']}
        for key in ('feature_matrix_sha256','ordered_ids_sha256','ordered_targets_sha256'):
            checks[key+'_equal']=receipt['training'][key]==details['training'][key]
        comparison={'reference':args.compare_to,'checks':checks}
        if not all(checks.values()):
            write_json(out/'COMPARISON_FAILURE.json',comparison)
            raise RuntimeError('Independent raw refit differs')
    result={'status':'PASS','candidate':args.candidate,'source':args.source,'tag':args.tag,'created_utc':utc_now(),
        'model_sha256':sha256(out/'model.joblib'),'training_receipt_sha256':sha256(out/'TRAINING.json'),
        'training':receipt['training'],'all_dev_prediction_sha256':fingerprint,'fingerprint_points':len(prediction),
        'comparison':comparison,'elapsed_seconds':time.perf_counter()-tick,'training_quality_score_computed':False,
        'promoted_by_this_refit':False,'seal_rows':0,'reduced_new_usage':0}
    write_json(out/'RESULTS.json',result)
    checkpoint('All-DEV refit '+args.candidate+'/'+args.source,['Prepared runtime parity','Independent iterator reproduction'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 REFIT PASS',args.candidate,args.source,result['model_sha256'],flush=True)


if __name__=='__main__':
    main()

"""Registered-bank all-DEV refit from verified cache or guarded raw iterator.

No training-set quality score is computed. Full-DEV prediction hashes are only
determinism/reproduction fingerprints and are excluded from model selection.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev,iter_dev_series,RAW_SHA
from src.next.candidate_io import load_full_oof,candidate_design,model_class
from src.next.registered_final_fitting import fit_all_development,training_feature_names,source_hash
from src.next.serial_training import train_frozen_candidate,training_hash
from src.next.serial_inference import implementation_hash as inference_hash
from src.next.serial_variance_runtime import source_hash as runtime_hash
from src.next.references import verify_reference_sources
from src.utils.artifacts import write_json,sha256,utc_now


class UnopenedSealValue:
    def __array__(self,*args,**kwargs):
        raise AssertionError('Sealed array conversion attempted')
    def __len__(self):
        raise AssertionError('Sealed length inspection attempted')
    def __float__(self):
        raise AssertionError('Sealed value conversion attempted')
    def __int__(self):
        raise AssertionError('Sealed target conversion attempted')
    def __bool__(self):
        raise AssertionError('Sealed target inspection attempted')


def guarded_raw_iterator(data):
    for name,digest in RAW_SHA.items():
        if sha256(ROOT/'data/raw'/name)!=digest:
            raise RuntimeError('Raw canonical training bytes changed')
    yield from iter_dev_series(data.guard)
    for sid in sorted(data.guard.seal_ids):
        yield sid,UnopenedSealValue(),UnopenedSealValue(),UnopenedSealValue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',default='J102_SERIAL_VARIANCE_512')
    parser.add_argument('--source',choices=['cache','iterator'],required=True)
    parser.add_argument('--tag',default='research_v1')
    parser.add_argument('--compare-to')
    args = parser.parse_args()
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Invalid refit tag')
    verify_reference_sources(ROOT)
    data = prepare_dev(ROOT)
    candidate,prior,_,_,oof_sources = load_full_oof(data,args.candidate)
    if prior['status'] not in ('ROBUSTNESS_PENDING','NEW_DEV_CANDIDATE'):
        raise RuntimeError('Research full-DEV refit requires a candidate passing its primary numerical gates')
    names = training_feature_names(candidate)
    out = ROOT/'artifacts/next/refits'/args.candidate/(args.source+'_'+args.tag)
    out.mkdir(parents=True,exist_ok=True)
    spec = {'status':'FROZEN_NEXT_RESEARCH_REFIT','candidate':candidate,
        'development_ids':sorted(data.guard.dev_ids),'known_training_ids':sorted(data.guard.split.mapping),
        'split_sha256':data.guard.split.digest,'input_origin':'VERIFIED_DEV_FEATURE_CACHE' if args.source=='cache' else 'PREDICATE_FILTERED_RAW_DEV_WITH_OPAQUE_SEAL_PLACEHOLDERS'}
    identity = {'candidate':candidate,'source':args.source,'tag':args.tag,'oof_sources':oof_sources,
        'script_sha256':sha256(Path(__file__)),'final_fitting_sha256':source_hash(),'training_api_sha256':training_hash(),
        'inference_sha256':inference_hash(),'compare_to':args.compare_to}
    policy_path = out/'POLICY.json'
    if policy_path.exists():
        if json.loads(policy_path.read_text(encoding='utf-8'))!=identity:
            raise RuntimeError('Refit identity changed; retain original and use an explicit revision')
    else:
        write_json(policy_path,identity)
        write_json(out/'SPEC.json',spec)
    final = out/'RESULTS.json'
    if final.exists():
        report = json.loads(final.read_text(encoding='utf-8'))
        if report['status']!='PASS' or sha256(out/'model.joblib')!=report['model_sha256'] or sha256(out/'model.json')!=report['descriptor_sha256']:
            raise RuntimeError('Completed research refit artifact changed')
        print('NEXT REFIT retained PASS',args.candidate,args.source,flush=True)
        return
    tick = time.perf_counter()
    if not (out/'model.json').exists():
        if (out/'model.joblib').exists():
            raise RuntimeError('A partial model artifact needs explicit review before any retraining')
        if args.source=='iterator':
            descriptor = train_frozen_candidate(guarded_raw_iterator(data),out,spec,root=ROOT)
        else:
            design,metadata,design_names,feature_sources,global_rows = candidate_design(data,candidate)
            np.testing.assert_array_equal(global_rows,np.arange(len(global_rows)))
            model,details = fit_all_development(design,metadata,candidate,data.guard,design_names)
            model.save(out/'model.joblib')
            descriptor = {'kind':'NEXT_REGISTERED_BINARY','runtime':'serial_variance_fast_h','candidate':candidate,
                'model_sha256':sha256(out/'model.joblib'),'runtime_sha256':runtime_hash(),'inference_sha256':inference_hash(),
                'status':spec['status'],'created_utc':utc_now(),'training':details,'feature_sources':feature_sources,
                'training_source_sha256':source_hash(),'input_origin':spec['input_origin'],'seal_rows_fitted':0,'reduced_new_usage':0}
            write_json(out/'model.json',descriptor)
            del design,metadata,model
            gc.collect()
    else:
        descriptor = json.loads((out/'model.json').read_text(encoding='utf-8'))
        if descriptor['candidate']!=candidate or sha256(out/'model.joblib')!=descriptor['model_sha256']:
            raise RuntimeError('Partial refit descriptor/model identity changed')
    loaded = model_class(candidate).load(out/'model.joblib')
    design,metadata,design_names,feature_sources,rows = candidate_design(data,candidate)
    prediction = np.asarray(loaded.predict(design),dtype=np.float32)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite all-DEV fingerprint predictions')
    fingerprint = hashlib.sha256(prediction.tobytes()).hexdigest()
    comparison = None
    if args.compare_to:
        reference_path = (ROOT/args.compare_to).resolve()
        reference = json.loads((reference_path/'RESULTS.json').read_text(encoding='utf-8'))
        other = json.loads((reference_path/'model.json').read_text(encoding='utf-8'))
        if reference['status']!='PASS' or other['candidate']!=candidate:
            raise RuntimeError('Refit comparison must use the same valid frozen candidate')
        checks = {'model_bytes_equal':sha256(out/'model.joblib')==reference['model_sha256'],
            'all_training_fingerprint_predictions_equal':fingerprint==reference['all_dev_prediction_sha256']}
        for key in ('feature_matrix_sha256','ordered_ids_sha256','ordered_targets_sha256'):
            checks[key+'_equal'] = descriptor['training'][key]==other['training'][key]
        if not all(checks.values()):
            write_json(out/'COMPARISON_FAILURE.json',{'status':'FAIL','checks':checks,'created_utc':utc_now()})
            raise RuntimeError('Independent all-DEV refit did not reproduce exact data/model/prediction bytes')
        comparison = {'reference':str(reference_path.relative_to(ROOT)),'checks':checks}
    report = {'status':'PASS','candidate':args.candidate,'source':args.source,'tag':args.tag,'created_utc':utc_now(),
        'model_sha256':sha256(out/'model.joblib'),'descriptor_sha256':sha256(out/'model.json'),
        'training':descriptor['training'],'all_dev_prediction_sha256':fingerprint,'fingerprint_points':len(prediction),
        'comparison':comparison,'elapsed_this_invocation_seconds':time.perf_counter()-tick,
        'training_quality_score_computed':False,'promoted_by_this_refit':False,'seal_rows':0,'reduced_new_usage':0}
    write_json(final,report)
    print('NEXT REFIT PASS',args.candidate,args.source,sha256(out/'model.joblib'),flush=True)


if __name__=='__main__':
    main()

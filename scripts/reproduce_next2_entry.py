"""Independent raw official-iterator train entry versus frozen research model."""
from pathlib import Path
import sys
import json
import argparse
import importlib
import hashlib
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',choices=['PERFORMANCE','DISTILLED'],required=True)
    parser.add_argument('--tag',default='raw_entry_v1')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Invalid reproduction tag')
    # Import the actual production entry before any project feature modules.
    entry=importlib.import_module('next2_'+args.role.lower()+'_main')
    from src.next.data import prepare_dev
    from scripts.refit_next_candidate_dev import guarded_raw_iterator
    from src.next2.inference import load_artifact
    from src.next2.runtime import prepare
    from src.next2.counterexamples import predict
    from src.next2.session import checkpoint
    from src.utils.artifacts import write_json,sha256,utc_now
    data=prepare_dev(ROOT)
    reference_dir=ROOT/'artifacts/next2/deployment_research'/args.role
    reference,reference_receipt=load_artifact(reference_dir)
    out=ROOT/'artifacts/next2/entry_reproduction'/args.role/args.tag
    out.mkdir(parents=True,exist_ok=True)
    identity={'role':args.role,'tag':args.tag,'script_sha256':sha256(Path(__file__)),
        'reference_descriptor_sha256':sha256(reference_dir/'model.json'),'reference_model_sha256':reference_receipt['model_sha256'],
        'entry_sha256':sha256(ROOT/('next2_'+args.role.lower()+'_main.py')),
        'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json')}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text())!=identity:
            raise RuntimeError('Entry reproduction policy changed')
    else:
        write_json(out/'POLICY.json',identity)
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text())
        if r['status']!='PASS' or sha256(out/'model.joblib')!=r['model_sha256']:
            raise RuntimeError('Entry reproduction result changed')
        print('NEXT2 ENTRY REPRODUCTION RETAINED',args.role,flush=True)
        return
    started=time.perf_counter()
    if not (out/'model.json').exists():
        entry.train(guarded_raw_iterator(data),out)
    model,receipt=load_artifact(out)
    # Exact serialized bytes cover estimator state as well as compact trees.
    if sha256(out/'model.joblib')!=reference_receipt['model_sha256']:
        write_json(out/'MODEL_BYTE_DIFFERENCE.json',{'status':'FAIL','actual':sha256(out/'model.joblib'),
            'expected':reference_receipt['model_sha256'],'message':'Keep both models for deterministic discrepancy diagnosis'})
        raise RuntimeError('Raw training entry changed serialized model bytes')
    left,right=prepare(reference,resident=False),prepare(model,resident=False)
    ho,oo=data.series['historical_offsets'],data.series['online_offsets']
    digest=hashlib.sha256()
    for j,sid in enumerate(data.ids):
        h=data.historical[ho[j]:ho[j+1]]
        o=data.online[oo[j]:oo[j+1]]
        a,b=predict(left,h,o),predict(right,h,o)
        np.testing.assert_array_equal(a,b)
        digest.update(b.tobytes())
        if (j+1)%1000==0:
            print('NEXT2 RAW ENTRY EXACT',args.role,j+1,round(time.perf_counter()-started,1),flush=True)
    data.guard.check_state()
    result={'status':'PASS','identity':identity,'created_utc':utc_now(),'role':args.role,
        'model_sha256':sha256(out/'model.joblib'),'descriptor_sha256':sha256(out/'model.json'),
        'independent_raw_iterator_training':True,'model_bytes_equal':True,'all_dev_prediction_bitwise_equal':True,
        'all_dev_prediction_sha256':digest.hexdigest(),'fingerprint_points':len(data.online),'fingerprint_series':len(data.ids),
        'training_quality_score_computed':False,'training':receipt['training'],
        'elapsed_seconds':time.perf_counter()-started,'seal_rows':0,'reduced_new_usage':0}
    write_json(out/'RESULTS.json',result)
    checkpoint('Raw entry reproduction '+args.role,['Independent-copy reproduction and final audit','Remaining fixed research diagnostics'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 RAW ENTRY REPRODUCTION PASS',args.role,result['model_sha256'],flush=True)


if __name__=='__main__':
    main()

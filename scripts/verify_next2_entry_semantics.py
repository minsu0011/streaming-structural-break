"""Separate scientific reproducibility from Python object-graph serialization.

The earlier byte-identity failure is retained. This verifies all predictive
arrays, named features, source/configuration identity and every DEV prediction.
It never replaces the independently trained model with reference bytes.
"""
from pathlib import Path
import sys
import json
import hashlib
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.inference import load_artifact
from src.next2.runtime import prepare
from src.next2.counterexamples import predict
from src.next.data import prepare_dev
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def compare_components(reference,actual):
    checks={'type_equal':type(reference)==type(actual),'named_features_equal':tuple(reference.names)==tuple(actual.names),
        'columns_equal':np.array_equal(reference.columns,actual.columns),'configuration_equal':reference.config==actual.config,
        'extension_equal':reference.extension==actual.extension,'engine_hash_equal':reference.engine_hash==actual.engine_hash,
        'implementation_hash_equal':reference.implementation_hash==actual.implementation_hash,
        'old_arch_hash_equal':reference.old_arch_hash==actual.old_arch_hash,
        'booster_text_equal':reference.estimator.booster_.model_to_string()==actual.estimator.booster_.model_to_string()}
    for k,(left,right) in enumerate(zip(reference.compact.args(),actual.compact.args())):
        checks['compact_argument_'+str(k)+'_equal']=np.array_equal(left,right)
    if not all(checks.values()):
        raise RuntimeError('Predictive model semantics differ: '+str(checks))
    return checks


def main():
    folder=ROOT/'artifacts/next2/entry_reproduction/PERFORMANCE/raw_entry_v1'
    reference_folder=ROOT/'artifacts/next2/deployment_research/PERFORMANCE'
    reference,rr=load_artifact(reference_folder)
    actual,ar=load_artifact(folder)
    identity={'script_sha256':sha256(Path(__file__)),'reference_model_sha256':sha256(reference_folder/'model.joblib'),
        'actual_model_sha256':sha256(folder/'model.joblib'),'original_failure_sha256':sha256(folder/'MODEL_BYTE_DIFFERENCE.json'),
        'mismatch_diagnosis_sha256':sha256(folder/'MISMATCH_DIAGNOSIS.json')}
    if (folder/'SEMANTIC_REPRODUCTION.json').exists():
        old=json.loads((folder/'SEMANTIC_REPRODUCTION.json').read_text())
        if old['identity']!=identity or old['status']!='PASS_PREDICTION_EXACT_WITH_SERIALIZATION_DIFFERENCE':
            raise RuntimeError('Semantic reproduction receipt changed')
        print('NEXT2 SEMANTIC REPRODUCTION RETAINED',flush=True)
        return
    components={name:compare_components(getattr(reference,name),getattr(actual,name)) for name in ('primary','complement')}
    if reference.weight!=actual.weight:
        raise RuntimeError('Blend weight differs')
    data=prepare_dev(ROOT)
    left,right=prepare(reference,resident=False),prepare(actual,resident=False)
    ho,oo=data.series['historical_offsets'],data.series['online_offsets']
    digest=hashlib.sha256()
    started=time.perf_counter()
    for j,sid in enumerate(data.ids):
        h=data.historical[ho[j]:ho[j+1]]
        o=data.online[oo[j]:oo[j+1]]
        a,b=predict(left,h,o),predict(right,h,o)
        np.testing.assert_array_equal(a,b)
        digest.update(b.tobytes())
        if (j+1)%1000==0:
            print('NEXT2 ENTRY PREDICTION EXACT',j+1,round(time.perf_counter()-started,1),flush=True)
    r={'status':'PASS_PREDICTION_EXACT_WITH_SERIALIZATION_DIFFERENCE','created_utc':utc_now(),'identity':identity,
        'component_semantic_checks':components,'model_bytes_equal':False,'all_dev_prediction_bitwise_equal':True,
        'independent_raw_iterator_training':True,'fingerprint_points':len(data.online),'fingerprint_series':len(data.ids),
        'all_dev_prediction_sha256':digest.hexdigest(),'original_failure_preserved':True,
        'reference_model_overwritten':False,'actual_model_replaced_with_reference':False,
        'training_quality_score_computed':False,'elapsed_seconds':time.perf_counter()-started,'seal_rows':0,
        'limitations':'Joblib bytes encode estimator metadata and Python object sharing. Byte identity across these entry paths failed; exact predictive semantics and all-point outputs are verified separately.'}
    write_json(folder/'SEMANTIC_REPRODUCTION.json',r)
    checkpoint('Independent raw entry predictive semantics reproduced',['Canonical compact deployment serialization','Final independent-copy audit'],
        artifacts=[(folder/'SEMANTIC_REPRODUCTION.json').relative_to(ROOT).as_posix()])
    print('NEXT2 SEMANTIC REPRODUCTION PASS',digest.hexdigest(),flush=True)


if __name__=='__main__':
    main()

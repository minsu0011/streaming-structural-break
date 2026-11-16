"""Preserve evidence for a production-entry versus research serialization mismatch."""
from pathlib import Path
import sys
import json
import io
import hashlib
import difflib
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


def object_hash(value):
    stream=io.BytesIO()
    joblib.dump(value,stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def main():
    out=ROOT/'artifacts/next2/entry_reproduction/PERFORMANCE/raw_entry_v1'
    refs={'primary':ROOT/'artifacts/next/refits/J102_SERIAL_VARIANCE_512/cache_research_v1',
        'complement':ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1'}
    rows=[]
    for role,folder in refs.items():
        original=joblib.load(folder/'model.joblib')
        new_folder=out/'training_components'/role
        actual=joblib.load(new_folder/'model.joblib')
        a=original.estimator.get_params()
        b=actual.estimator.get_params()
        arrays=[]
        for k,(x,y) in enumerate(zip(original.compact.args(),actual.compact.args())):
            if isinstance(x,np.ndarray):
                arrays.append({'position':k,'shape_original':list(x.shape),'shape_actual':list(y.shape),
                    'equal':np.array_equal(x,y),'max_abs_diff':float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.shape==y.shape and x.size else None})
            else:
                arrays.append({'position':k,'equal':x==y,'original':x,'actual':y})
        old_training=json.loads((folder/('model.json' if role=='primary' else 'TRAINING.json')).read_text())['training']
        new_training=json.loads((new_folder/'TRAINING.json').read_text())['training']
        diff=list(difflib.unified_diff(original.estimator.booster_.model_to_string().splitlines(),actual.estimator.booster_.model_to_string().splitlines(),lineterm=''))
        (out/(role+'_BOOSTER_TEXT_DIFF.txt')).write_text('\n'.join(diff),encoding='utf-8')
        row={'role':role,'candidate_class_equal':type(original)==type(actual),
            'parameter_changes':{k:{'original':a.get(k),'actual':v} for k,v in b.items() if a.get(k)!=v},
            'compact_arrays':arrays,'all_compact_arrays_equal':all(r['equal'] for r in arrays),
            'attribute_serialization_changes':[k for k,v in vars(actual).items() if object_hash(v)!=object_hash(getattr(original,k))],
            'data_fingerprints':{k:old_training[k]==new_training[k] for k in ('feature_matrix_sha256','ordered_ids_sha256','ordered_targets_sha256')},
            'booster_text_diff_lines':len(diff),'booster_text_diff_sha256':sha256(out/(role+'_BOOSTER_TEXT_DIFF.txt'))}
        rows.append(row)
        print(json.dumps(row),flush=True)
        print('BOOSTER TEXT DIFF HEAD',diff[:25],flush=True)
    write_json(out/'MISMATCH_DIAGNOSIS.json',{'status':'DIAGNOSED','created_utc':utc_now(),'components':rows,
        'original_failure_preserved':True,'model_bytes_overwritten':False})


if __name__=='__main__':
    main()

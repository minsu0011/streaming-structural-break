"""Compare exact compact predictive state, independent of joblib byte identity."""
import dataclasses
import hashlib
import json
import struct
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.learners import ModelBundle
from src.models.representations import bundle_class
from src.utils.artifacts import sha256,utc_now,write_json


def canonical(value):
    if isinstance(value,np.ndarray):
        return {'array_dtype':value.dtype.str,'dtype_descriptor':value.dtype.descr,'shape':list(value.shape),'c_order_bytes_sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}
    if isinstance(value,np.generic):return canonical(value.item())
    if isinstance(value,float):return {'float64_bytes':struct.pack('>d',value).hex()}
    if dataclasses.is_dataclass(value):
        fields={field.name:getattr(value,field.name) for field in dataclasses.fields(value)}
        if isinstance(value,ModelBundle):
            if value.compact is None or value.linear_weight is not None or value.scaler is not None:
                raise RuntimeError('Only the selected compact-tree inference path is covered')
            del fields['estimator']
            fields['native_estimator_type_not_used_by_compact_inference']=type(value.estimator).__module__+'.'+type(value.estimator).__qualname__
        return {'dataclass':type(value).__module__+'.'+type(value).__qualname__,'fields':{key:canonical(item) for key,item in fields.items()}}
    if isinstance(value,(tuple,list)):return {'sequence_type':type(value).__name__,'items':[canonical(item) for item in value]}
    if isinstance(value,dict):return {key:canonical(item) for key,item in value.items()}
    if value is None or isinstance(value,(str,int,bool)):return value
    raise TypeError('Unsupported predictive-state value type: '+type(value).__name__)


def digest(value):return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    frozen=json.loads((ROOT/'DEV_CHAMPION_CANDIDATES.json').read_text())
    proof=json.loads((ROOT/'artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json').read_text())
    if proof['status']!='PASS' or proof['selection_sha256']!=sha256(ROOT/'DEV_CHAMPION_CANDIDATES.json'):
        raise RuntimeError('Matching independent raw reproduction required')
    source=ROOT/proof['source_copy'];rows=[]
    for role,item in frozen['candidates'].items():
        loader=bundle_class(item['candidate'].get('representation','base'))
        cv=proof['candidates'][role]['cv_report'];final=proof['candidates'][role]['final_model_report']
        pairs=[('cv_fold_'+str(k),ROOT/'artifacts/models/real'/item['experiment_id']/f'fold_{k}.joblib',source/Path(cv).parent/f'fold_{k}.joblib') for k in range(5)]
        pairs.append(('all_dev_final',ROOT/item['model_directory']/'model.joblib',source/Path(final).parent/'model.joblib'))
        for kind,left,right in pairs:
            a,b=loader.load(left),loader.load(right);a_hash,b_hash=digest(a),digest(b)
            if a_hash!=b_hash:raise RuntimeError('Exact compact predictive state differs for '+role+' '+kind)
            a_bases=[child.base_model for child in a.components] if hasattr(a,'components') else [a.base_model]
            b_bases=[child.base_model for child in b.components] if hasattr(b,'components') else [b.base_model]
            native=[]
            for x,y in zip(a_bases,b_bases):
                if x.name=='M4_lightgbm':
                    text_left=x.estimator.booster_.model_to_string();text_right=y.estimator.booster_.model_to_string()
                    native.append({'learner':x.name,'model_text_exact':text_left==text_right,
                                   'original_text_sha256':hashlib.sha256(text_left.encode()).hexdigest(),
                                   'reproduced_text_sha256':hashlib.sha256(text_right.encode()).hexdigest()})
                elif x.name=='M3_histgb':
                    x_state={'baseline':x.estimator._baseline_prediction,'trees':[[tree.nodes for tree in trees] for trees in x.estimator._predictors]}
                    y_state={'baseline':y.estimator._baseline_prediction,'trees':[[tree.nodes for tree in trees] for trees in y.estimator._predictors]}
                    native.append({'learner':x.name,'tree_nodes_and_bias_exact':digest(x_state)==digest(y_state),
                                   'original_nodes_sha256':digest(x_state),'reproduced_nodes_sha256':digest(y_state)})
            rows.append({'role':role,'candidate':item['experiment_id'],'kind':kind,'original_model_sha256':sha256(left),
                         'reproduced_model_sha256':sha256(right),'serialized_model_bytes_identical':sha256(left)==sha256(right),
                         'original_predictive_state_sha256':a_hash,'reproduced_predictive_state_sha256':b_hash,
                         'exact_predictive_state_match':True,'native_estimator_checks':native})
    report={'status':'PASS','timestamp':utc_now(),'selection_sha256':sha256(ROOT/'DEV_CHAMPION_CANDIDATES.json'),
            'archive_raw_proof_sha256':sha256(ROOT/'artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json'),
            'script_sha256':sha256(__file__),'comparisons':rows,'comparison_count':len(rows),
            'scope':'Exact typed compact-tree arrays, column selection, configuration, calibration, representation/order and implementation identities in 15 CV plus 3 final artifact pairs. Native fitted estimator diagnostics are separately recorded. No training, prediction, scoring or target reads.',
            'interpretation':'Different joblib bytes do not imply different compact predictive state. No claim of complete native-estimator object equality or cross-platform execution equivalence is made.',
            'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(ROOT/'artifacts/real/PREDICTIVE_STATE_REPRODUCTION_REAL.json',report)
    print('EXACT PREDICTIVE STATE PASS',len(rows),'artifact pairs')


if __name__=='__main__':main()

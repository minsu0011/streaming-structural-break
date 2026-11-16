"""Every DEV feature consumed by frozen trees under the sparse used runtime."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.io import model_class
from src.next2.pruned_runtime import PreparedCompactPredictor
from src.next2.used_feature_runtime import PreparedUsedPredictor,source_hash
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate')
    args=parser.parse_args()
    data=prepare_dev(ROOT)
    lock=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    for ref in lock['references'].values():
        eid=ref['experiment_id']
        if args.candidate and args.candidate!=eid:
            continue
        out=ROOT/'artifacts/next2/used_runtime'/eid
        out.mkdir(parents=True,exist_ok=True)
        identity={'model_sha256':ref['model_sha256'],'runtime_sha256':source_hash(),'script_sha256':sha256(Path(__file__)),
            'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json')}
        if (out/'RESULTS.json').exists():
            r=json.loads((out/'RESULTS.json').read_text())
            if r['status']!='PASS' or r['identity']!=identity:
                raise RuntimeError('Used-runtime completed proof changed')
            print('NEXT2 USED RUNTIME RETAINED',eid,flush=True)
            continue
        if sha256(ROOT/ref['model_path'])!=ref['model_sha256']:
            raise RuntimeError('Frozen model bytes changed')
        model=model_class(ref['candidate']).load(ROOT/ref['model_path'])
        left=PreparedCompactPredictor(model)
        right=PreparedUsedPredictor(model)
        write_json(out/'POLICY.json',identity)
        ho,oo=data.series['historical_offsets'],data.series['online_offsets']
        started=time.perf_counter()
        sizes=[]
        for j,sid in enumerate(data.ids):
            h=data.historical[ho[j]:ho[j+1]]
            o=data.online[oo[j]:oo[j+1]]
            a,b=left.new_state(h),right.new_state(h)
            for point in o:
                p=a.predict_one(point)
                q=b.predict_one(point)
                if p!=q or not np.array_equal(a.output[right.used],b.output):
                    raise AssertionError('Used-runtime mismatch in DEV series '+str(sid))
            sizes.append(b.state_array_bytes)
            if (j+1)%1000==0:
                print('NEXT2 USED RUNTIME EXACT',eid,j+1,round(time.perf_counter()-started,1),flush=True)
        r={'status':'PASS','identity':identity,'created_utc':utc_now(),'candidate':eid,
            'series':len(data.ids),'points':len(data.online),'tree_used_feature_values_compared':len(data.online)*len(right.names),
            'original_model_feature_contract':len(model.names),'materialized_tree_input_features':len(right.names),
            'materialized_old_features':len(right.old_columns),'old_ewma_cells_before':51,'old_ewma_cells_after':b.old.ewma_cells,
            'v1_state_bytes':a.state_array_bytes,'used_state_bytes_min':min(sizes),'used_state_bytes_max':max(sizes),
            'selected_feature_max_abs_diff':0.,'prediction_max_abs_diff':0.,'model_binary_unchanged':True,
            'elapsed_seconds':time.perf_counter()-started,'seal_rows':0,
            'interpretation':'Unused tree inputs and their unnecessary old EWMA/calibration work removed; scientific model definition and learned decision function unchanged.'}
        write_json(out/'RESULTS.json',r)
        checkpoint('Exact used-feature runtime '+eid,['Isolated latency comparison','OOF and production contract checks'],
            artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
        print('NEXT2 USED RUNTIME PASS',eid,len(right.names),r['used_state_bytes_max'],flush=True)


if __name__=='__main__':
    main()

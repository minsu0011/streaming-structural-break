"""Every DEV OOF point and C205 feature under the exact prepared blend runtime."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.io import design,model_class
from src.next2.blend import FixedBlendBundle
from src.next2.runtime import prepare,PreparedBankPredictor,source_hash
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    eid='BL2_C205_DEPENDENCE_80'
    data=prepare_dev(ROOT)
    candidate,report,meta,expected,sources=load_full_oof(data,eid)
    c=json.loads((ROOT/'artifacts/next2/experiments/C205_DEPENDENCE/RESULTS.json').read_text())['candidate']
    x,cmeta,names,features,rows=design(data,c)
    np.testing.assert_array_equal(rows,np.arange(len(expected)))
    engines=[prepare(FixedBlendBundle.load(ROOT/'artifacts/next2/experiments'/eid/'full'/f'fold_{fold}.joblib')) for fold in range(5)]
    out=ROOT/'artifacts/next2/runtime_blend'
    out.mkdir(parents=True,exist_ok=True)
    identity={'candidate':eid,'oof_sources':sources,'runtime_sha256':source_hash(),
        'script_sha256':sha256(Path(__file__)),'feature_sources':features,'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json')}
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text())
        if r['status']!='PASS' or r['identity']!=identity:
            raise RuntimeError('Prepared-runtime completed identity changed')
        print('NEXT2 BLEND RUNTIME RETAINED',flush=True)
        return
    write_json(out/'POLICY.json',identity)
    ho,oo=data.series['historical_offsets'],data.series['online_offsets']
    actual=np.lib.format.open_memmap(out/'oof_predictions.npy',mode='w+',dtype=np.float32,shape=expected.shape)
    started=time.perf_counter()
    state_sizes=[]
    for j,sid in enumerate(data.ids):
        fold=data.guard.split.assignment(int(sid))[1]
        h=data.historical[ho[j]:ho[j+1]]
        o=data.online[oo[j]:oo[j+1]]
        state=engines[fold].new_state(h)
        for k,point in enumerate(o):
            row=oo[j]+k
            actual[row]=state.predict_one(point)
            if not np.array_equal(state.complement.output,x[row]):
                raise AssertionError('Compact C205 feature mismatch at DEV series '+str(sid)+' time '+str(k))
        np.testing.assert_array_equal(actual[oo[j]:oo[j+1]],expected[oo[j]:oo[j+1]])
        state_sizes.append(state.state_array_bytes)
        if (j+1)%500==0:
            print('NEXT2 BLEND OOF EXACT',j+1,round(time.perf_counter()-started,1),flush=True)
    actual.flush()
    baseline=PreparedBankPredictor(engines[0].model.complement).new_state(h)
    r={'status':'PASS','identity':identity,'created_utc':utc_now(),'series':len(data.ids),'oof_points':len(actual),
        'c205_feature_entries_compared':int(x.size),'selected_feature_max_abs_diff':0.,'oof_max_abs_diff':0.,
        'compound_state_bytes_min':min(state_sizes),'compound_state_bytes_max':max(state_sizes),
        'original_complement_state_bytes_last_h':baseline.state_array_bytes,
        'compact_complement_state_bytes':state.complement.state_array_bytes,
        'primary_state_bytes':state.primary.state_array_bytes,'elapsed_seconds':time.perf_counter()-started,
        'oof_sha256':sha256(out/'oof_predictions.npy'),'seal_rows':0}
    write_json(out/'RESULTS.json',r)
    checkpoint('Complete DEV prepared blend OOF exact',['Independent raw refit','Fixed synthetic finalist evaluation'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 BLEND RUNTIME PASS',r['compound_state_bytes_max'],flush=True)


if __name__=='__main__':
    main()

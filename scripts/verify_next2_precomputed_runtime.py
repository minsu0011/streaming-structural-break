"""All-DEV selected-feature proof and independent held-out proof for precomputation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import hashlib
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.canonical_model import load,dumps,loads
from src.next2.io import model_class
from src.next2.precomputed_serial_runtime import PreparedPrecomputedSerialPredictor,source_hash
from src.next2.used_feature_runtime import PreparedUsedPredictor
from src.next2.numeric_memory import accounting
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now
J='J102_SERIAL_VARIANCE_512'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all_dev','oof'],required=True);args=parser.parse_args()
    data=prepare_dev(ROOT);out=ROOT/'artifacts/next2/precomputed_runtime'/args.stage;out.mkdir(parents=True,exist_ok=True)
    identity={'stage':args.stage,'runtime_sha256':source_hash(),'script_sha256':sha256(Path(__file__)),
        'protocol_sha256':sha256(ROOT/'docs/NEXT2_PRECOMPUTED_SERIAL_PROTOCOL.md'),'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json')}
    engines=[];references=[];sources={};expected=None
    if args.stage=='all_dev':
        path=ROOT/'artifacts/next2/canonical_deployment/DISTILLED_tree_used_v1/model.sbnext2'
        model=load(path);sources[str(path.relative_to(ROOT))]=sha256(path)
        engines=[PreparedPrecomputedSerialPredictor(model)];references=[PreparedUsedPredictor(model)]
    else:
        candidate,_,metadata,expected,sources=load_full_oof(data,J)
        for fold in range(5):
            path=ROOT/'artifacts/next/experiments'/J/'full'/f'fold_{fold}.joblib'
            model=model_class(candidate).load(path);content=dumps(model);restored=loads(content)
            assert dumps(restored)==content
            engines.append(PreparedPrecomputedSerialPredictor(restored));references.append(PreparedUsedPredictor(restored))
    identity['model_sources']=sources
    if (out/'POLICY.json').exists():assert json.loads((out/'POLICY.json').read_text())==identity
    else:write_json(out/'POLICY.json',identity)
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text());assert r['status']=='PASS' and r['identity']==identity
        print('NEXT2 PRECOMPUTED PROOF RETAINED',args.stage,flush=True);return
    ho,oo=data.series['historical_offsets'],data.series['online_offsets'];tick=time.perf_counter()
    digest=hashlib.sha256();sizes=[];private=[];features=0;memory_rows=[]
    for j,sid in enumerate(data.ids):
        fold=0 if args.stage=='all_dev' else data.guard.split.assignment(int(sid))[1]
        h=data.historical[ho[j]:ho[j+1]];o=data.online[oo[j]:oo[j+1]]
        left=references[fold].new_state(h);right=engines[fold].new_state(h)
        actual=np.empty(len(o),dtype=np.float32)
        for k,point in enumerate(o):
            a,b=left.predict_one(point),right.predict_one(point)
            feature_exact=np.array_equal(left.output.view(np.uint32),right.output.view(np.uint32))
            if a!=b or not feature_exact:
                write_json(out/'FAILURE.json',{'status':'FAIL','identity':identity,'dataset_id':int(sid),'online_index':k,
                    'feature_exact':bool(feature_exact),'prediction_exact':a==b})
                raise AssertionError('Precomputed runtime changed selected feature or prediction')
            actual[k]=b
        if expected is not None:np.testing.assert_array_equal(actual,expected[oo[j]:oo[j+1]])
        digest.update(actual.tobytes());features+=len(o)*len(engines[fold].names);sizes.append(right.state_array_bytes)
        if j in (0,len(data.ids)//2,len(data.ids)-1):
            mem,_,_=accounting(right);memory_rows.append({'dataset_id':int(sid),'historical_points':len(h),**mem})
        if (j+1)%1000==0:print('NEXT2 PRECOMPUTED EXACT',args.stage,j+1,round(time.perf_counter()-tick,1),flush=True)
    if expected is not None:assert digest.hexdigest()==hashlib.sha256(expected.tobytes()).hexdigest()
    data.guard.check_state()
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'series':len(data.ids),'points':len(data.online),
        'selected_feature_entries_compared':features,'selected_feature_max_abs_diff':0.,'prediction_max_abs_diff':0.,
        'prediction_fingerprint_sha256':digest.hexdigest(),'materialized_features':[len(e.names) for e in engines],
        'state_array_bytes_min':min(sizes),'state_array_bytes_max':max(sizes),'numeric_memory_probes':memory_rows,
        'historical_precomputed_table_bytes':right.table.nbytes+right.logs.nbytes,'elapsed_seconds':time.perf_counter()-tick,
        'new_fits':0,'seal_rows':0,'in_sample_quality_score_computed':False,'heldout_oof_checked':expected is not None})
    checkpoint('Precomputed exact runtime '+args.stage,['Isolated fixed runtime cost comparison','Retain original deployment unless fixed engineering criteria pass'])


if __name__=='__main__':main()

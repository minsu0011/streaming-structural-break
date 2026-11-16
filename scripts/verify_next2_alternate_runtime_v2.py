"""Exact runtime predictions for every existing ALT/H held-out model and series."""
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
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle,source_hash as blend_hash
from src.next2.canonical_model import dumps,loads
from src.next2.runtime import prepare,source_hash as compact_hash
from src.next2.used_feature_runtime import PreparedUsedPredictor,source_hash as used_hash
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512';C='C205_DEPENDENCE';B='BL2_C205_DEPENDENCE_80'
SPLITS=('ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['DISTILLED','PERFORMANCE'],required=True);args=parser.parse_args()
    data=prepare_dev(ROOT);meta=data.rows();ho,oo=data.series['historical_offsets'],data.series['online_offsets']
    out=ROOT/'artifacts/next2/alternate_runtime_v2'/args.role;out.mkdir(parents=True,exist_ok=True)
    jc=json.loads((ROOT/'artifacts/next/experiments'/J/'RESULTS.json').read_text())['candidate']
    cc=json.loads((ROOT/'artifacts/next2/experiments'/C/'RESULTS.json').read_text())['candidate']
    identity={'role':args.role,'script_sha256':sha256(Path(__file__)),'runtime_sha256':used_hash() if args.role=='DISTILLED' else compact_hash(),
        'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json'),'new_fits':0,'splits':list(SPLITS)}
    lock=out/'POLICY.json'
    if lock.exists():assert json.loads(lock.read_text())==identity
    else:write_json(lock,identity)
    results=[];index={int(sid):j for j,sid in enumerate(data.ids)};began=time.perf_counter()
    for split in SPLITS:
        for fold in range(5):
            name=f'fold_{fold}';folder=out/split;folder.mkdir(parents=True,exist_ok=True)
            left=ROOT/'artifacts/next/robustness/candidates'/J/split
            right=ROOT/'artifacts/next2/robustness'/C/split
            expected_folder=left if args.role=='DISTILLED' else ROOT/'artifacts/next2/robustness'/B/split
            sources={}
            def checked_model(base,candidate):
                receipt=base/(name+'.json');r=json.loads(receipt.read_text());assert r['status']=='PASS'
                path=base/(name+'.joblib');assert sha256(path)==r['files_sha256'][path.name]
                sources[str(path.relative_to(ROOT))]=sha256(path);sources[str(receipt.relative_to(ROOT))]=sha256(receipt)
                return model_class(candidate).load(path)
            model=checked_model(left,jc)
            if args.role=='PERFORMANCE':model=FixedBlendBundle(model,checked_model(right,cc),.8,blend_hash())
            expected_receipt=expected_folder/(name+'.json');r=json.loads(expected_receipt.read_text());assert r['status']=='PASS'
            sources[str(expected_receipt.relative_to(ROOT))]=sha256(expected_receipt)
            for relative,digest in r.get('component_sources_sha256',{}).items():assert sha256(ROOT/relative)==digest
            for suffix in ('.npy','_global_rows.npy'):
                path=expected_folder/(name+suffix);assert sha256(path)==r['files_sha256'][path.name];sources[str(path.relative_to(ROOT))]=sha256(path)
            expected=np.load(expected_folder/(name+'.npy'),allow_pickle=False)
            coordinates=np.load(expected_folder/(name+'_global_rows.npy'),allow_pickle=False)
            row_ids=meta['dataset_id'][coordinates];starts=np.r_[0,np.flatnonzero(row_ids[1:]!=row_ids[:-1])+1];ids=row_ids[starts];assert len(ids)==len(np.unique(ids));data.guard.admit(ids,purpose='Exact existing held-out ALT/H runtime proof')
            canonical=dumps(model);model=loads(canonical);assert dumps(model)==canonical
            engine=PreparedUsedPredictor(model) if args.role=='DISTILLED' else prepare(model)
            receipt_path=folder/(name+'.json')
            key={**identity,'split':split,'fold':fold,'source_sha256':sources,'canonical_model_sha256':hashlib.sha256(canonical).hexdigest()}
            if receipt_path.exists():
                previous=json.loads(receipt_path.read_text());assert previous['status']=='PASS' and previous['identity']==key
                results.append(previous);continue
            cursor=0;digest=hashlib.sha256();sizes=[]
            for sid in ids:
                j=index[int(sid)];h=data.historical[ho[j]:ho[j+1]];o=data.online[oo[j]:oo[j+1]]
                np.testing.assert_array_equal(coordinates[cursor:cursor+len(o)],np.arange(oo[j],oo[j+1]))
                state=engine.new_state(h);p=np.asarray([state.predict_one(v) for v in o],dtype=np.float32)
                np.testing.assert_array_equal(p,expected[cursor:cursor+len(o)]);digest.update(p.tobytes());cursor+=len(o);sizes.append(state.state_array_bytes)
            assert cursor==len(expected) and digest.hexdigest()==hashlib.sha256(expected.tobytes()).hexdigest()
            report={'status':'PASS','identity':key,'created_utc':utc_now(),'series':len(ids),'points':cursor,
                'prediction_max_abs_diff':0.,'prediction_fingerprint_sha256':digest.hexdigest(),'state_bytes_min':min(sizes),'state_bytes_max':max(sizes),
                'model_materialized_features':len(engine.names) if args.role=='DISTILLED' else len(model.primary.names)+len(model.complement.names),
                'new_fits':0,'seal_rows':0}
            write_json(receipt_path,report);results.append(report)
            print('NEXT2 ALTERNATE RUNTIME EXACT',args.role,split,fold,len(ids),round(time.perf_counter()-began,1),flush=True)
        checkpoint('Exact alternate runtime '+args.role+'/'+split,['Remaining fixed alternate/H models','Separate component timing'])
    assert len(results)==20 and sum(r['points'] for r in results)==4*len(data.online)
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'model_fits_checked':20,
        'series_replays':sum(r['series'] for r in results),'points':sum(r['points'] for r in results),'prediction_max_abs_diff':0.,
        'canonical_roundtrip_exact':True,'new_fits':0,'seal_rows':0,'elapsed_seconds':time.perf_counter()-began,
        'receipts_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(out.glob('*/fold_?.json'))},
        'scope':'Every existing held-out prediction from three alternate partitions and leave-one-H-regime-out, using the exact runtime considered for deployment. No refitting or changed scores.'})


if __name__=='__main__':main()

"""Materialize one unchanged B2 model for matched all-DEV synthetic comparisons."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import hashlib
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.cache import build_design
from src.next4.diagnostic_refit import fit_all_dev
from src.next4.model import FeatureModel
from src.next.data import prepare_dev


def main():
    candidate='B2_LOG_ENERGY_SEMITAIL';nom=BASE/'diagnostic_policy/B2_ALL_DEV_REFIT.json';policy=read_json(nom)
    for name,sha in policy['source_sha256'].items():assert digest(ROOT/name)==sha
    registration_path=BASE/'registrations'/(candidate+'.json');registration=read_json(registration_path)
    assert digest(registration_path)==policy['registration_sha256']
    out=BASE/'refits'/candidate/'diagnostic_all_dev_v1';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'nomination_sha256':digest(nom),'new_fits_intended':1},immutable=True)
    data=prepare_dev(ROOT);x,metadata,names,rows,cache=build_design(data,registration,'full')
    started=time.perf_counter()
    atomic_json(out/'FIT_START.json',{'created_utc':utc_now(),'training_rows':len(rows),'training_series':len(data.ids),'new_fits':1},immutable=True)
    progress('B2 all-DEV diagnostic fit','One fixed candidate fit; no real training-set quality score and no frozen base refit')
    model,training=fit_all_dev(x,metadata,guard=data.guard,registration=registration,expected_ids=data.ids)
    model.save(out/'model.joblib');restored=FeatureModel.load(out/'model.joblib')
    fingerprint=hashlib.sha256();position=0;checked=0
    for sid in data.ids:
        h,o,_=data.get(int(sid));p=restored.predict(x[position:position+len(o)]).astype(np.float32)
        if checked<2:
            state=restored.make_state(h)
            live=np.asarray([restored.predict_one(state.update(float(v))) for v in o],dtype=np.float32)
            np.testing.assert_array_equal(p,live);checked+=1
        fingerprint.update(p.tobytes());position+=len(o)
        if position%500000<len(o):
            progress('B2 all-DEV inference fingerprint',str(position)+' training coordinates; labels excluded from inference checks')
            print('NEXT4 B2 DIAGNOSTIC INFERENCE',position,flush=True)
    assert position==len(rows)
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':candidate,
        'nomination_sha256':digest(nom),'registration_sha256':digest(registration_path),'cache_identity':cache['identity'],
        'training':training,'new_fits':1,'frozen_base_new_fits':0,'seal_rows':0,
        'all_DEV_prediction_sha256':fingerprint.hexdigest(),'prediction_points':position,
        'streaming_cached_exact_series':checked,'save_load_validation':True,'elapsed_seconds':time.perf_counter()-started,
        'files_sha256':{'model.joblib':digest(out/'model.joblib')},'competition_quality_claim':False,
        'promotion':False,'fresh_raw_reproduction':False},immutable=True)
    print('NEXT4 B2 DIAGNOSTIC ALL-DEV REFIT COMPLETE',flush=True)


if __name__=='__main__':main()

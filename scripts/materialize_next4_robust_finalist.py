"""One nominated all-DEV fit, or its independent raw-feature training reproduction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.robust_finalist import fit_registered_all_dev
from src.next4.model import FeatureModel,feature_module
from src.next4.cache import build_design
from src.next.data import prepare_dev,iter_dev_series,RAW_SHA
from src.data.labels import labels_from_tau


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--fresh-raw',action='store_true');args=parser.parse_args()
    cid=args.candidate;nompath=BASE/'finalist_nominations'/(cid+'.json');nom=read_json(nompath)
    for name,sha in nom['execution_sources_sha256'].items():assert digest(ROOT/name)==sha
    regpath=BASE/'registrations'/(cid+'.json');reg=read_json(regpath);assert digest(regpath)==nom['registration_sha256']
    if args.fresh_raw:
        session=read_json(BASE/'SESSION.json');assert session.get('audit_started_utc')
        selection=read_json(ROOT/'DEV_CANDIDATES_NEXT4.json');assert selection['roles']['NEXT4_ROBUSTNESS']['experiment_id']==cid
    out=BASE/'reproduction'/cid if args.fresh_raw else BASE/'refits'/cid/'nominated_all_dev_v1'
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'nomination_sha256':digest(nompath),'fresh_raw':args.fresh_raw},immutable=True)
    data=prepare_dev(ROOT);cached,meta,names,rows,cache=build_design(data,reg,'full');raw_proof={}
    phase='FINAL_AUDIT' if args.fresh_raw else 'RESEARCH'
    if args.fresh_raw:
        for name,sha in RAW_SHA.items():
            assert digest(ROOT/'data/raw'/name)==sha;raw_proof[name]=sha
        x=np.lib.format.open_memmap(out/'fresh_features.npy',mode='w+',dtype=np.float32,shape=cached.shape)
        ids=[];times=[];labels=[];folds=[];position=0;module=feature_module(reg['feature_module'])
        for j,(sid,h,o,tau) in enumerate(iter_dev_series(data.guard)):
            assert sid==data.ids[j];values=module.make_state(h,reg['feature_settings']).replay(o);end=position+len(o)
            np.testing.assert_array_equal(values,cached[position:end]);x[position:end]=values
            ids.append(np.full(len(o),sid,dtype=np.int32));times.append(np.arange(len(o),dtype=np.int16))
            labels.append(labels_from_tau(len(o),tau));folds.append(np.full(len(o),data.guard.split.assignment(sid)[1],dtype=np.int8))
            position=end
            if (j+1)%250==0:
                x.flush();progress('Fresh raw DEV feature reproduction',cid+' '+str(j+1)+'/8000; predicate-filtered raw only, every feature exact',phase=phase,counts_as_research=False)
                print('NEXT4 FRESH RAW FEATURES',cid,j+1,flush=True)
        assert position==len(rows);x.flush()
        rebuilt={key:np.concatenate(values) for key,values in [('dataset_id',ids),('time_online',times),('target',labels),('fold',folds)]}
        for key in meta:np.testing.assert_array_equal(meta[key],rebuilt[key])
        meta=rebuilt
        atomic_json(out/'FRESH_FEATURES.json',{'status':'PASS','created_utc':utc_now(),'raw_sha256':raw_proof,
            'fresh_feature_sha256':digest(out/'fresh_features.npy'),'cached_feature_sha256':cache['features_sha256'],
            'every_feature_and_training_coordinate_exact':True,'source':'Raw DEV predicate scanner; labels reconstructed from admitted DEV tau only',
            'cached_feature_values_used_as_training_input':False,'seal_rows':0},immutable=True)
    else:x=cached
    atomic_json(out/'FIT_START.json',{'created_utc':utc_now(),'candidate':cid,'new_fits':1,'training_rows':len(rows),'training_series':8000},immutable=True)
    progress('Nominated unchanged all-DEV fit',cid+(' fresh raw reconstruction' if args.fresh_raw else ' original cached design'),phase=phase,counts_as_research=not args.fresh_raw)
    model,training=fit_registered_all_dev(x,meta,guard=data.guard,registration=reg,expected_ids=data.ids,fresh_raw=args.fresh_raw)
    model.save(out/'model.joblib');restored=FeatureModel.load(out/'model.joblib');fingerprint=hashlib.sha256();position=0
    for j,sid in enumerate(data.ids):
        h,o,_=data.get(int(sid));prediction=np.asarray(restored.predict(x[position:position+len(o)]),dtype=np.float32)
        np.testing.assert_array_equal(prediction,np.asarray(model.predict(x[position:position+len(o)]),dtype=np.float32))
        if j<20:
            state=restored.make_state(h);actual=np.asarray([restored.predict_one(state.update(float(v))) for v in o],dtype=np.float32)
            np.testing.assert_array_equal(actual,prediction)
        fingerprint.update(prediction.tobytes());position+=len(o)
        if (j+1)%500==0:
            progress('Nominated all-DEV save/load and inference',cid+' '+str(j+1)+'/8000',phase=phase,counts_as_research=not args.fresh_raw)
            print('NEXT4 NOMINATED INFERENCE',cid,j+1,flush=True)
    assert position==len(rows)
    if args.fresh_raw:
        original=read_json(BASE/'refits'/cid/'nominated_all_dev_v1/RESULTS.json')
        assert fingerprint.hexdigest()==original['all_DEV_prediction_sha256']
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':cid,'new_fits':1,
        'training':training,'nomination_sha256':digest(nompath),'registration_sha256':digest(regpath),
        'all_DEV_prediction_sha256':fingerprint.hexdigest(),'prediction_points':position,'all_DEV_series':8000,
        'save_load_all_DEV_exact':True,'scalar_batch_exact_series':20,'frozen_base_new_fits':0,'seal_rows':0,
        'fresh_raw_reproduction':args.fresh_raw,'fresh_raw_feature_and_inference_exact':bool(args.fresh_raw),
        'quality_selection_uses_original_heldout_scores_only':True,'real_in_sample_quality_score_computed':False,
        'cache_identity':cache['identity'],'raw_sha256_checked_for_reproduction':raw_proof,
        'files_sha256':{'model.joblib':digest(out/'model.joblib')}},immutable=True)
    print('NEXT4 NOMINATED MATERIALIZATION COMPLETE',cid,'fresh_raw',args.fresh_raw,flush=True)


if __name__=='__main__':main()

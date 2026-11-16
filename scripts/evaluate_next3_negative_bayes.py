"""Matched frozen-recipe comparison with four appended directional Bayes inputs."""
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.negative_bayes import NAMES,source_hash,negative_feature_replay,NegativeBayesBundle
from src.next3.frozen import candidate,fold_artifact,J102,C205
from src.next.data import prepare_dev
from src.next2.io import design
from src.next.fitting import fit_binary
from src.next.engine import EngineConfig
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc


KIND='S3_NEGATIVE_BAYES'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['screen','full'],required=True);args=parser.parse_args()
    hypothesis_path=BASE/'hypotheses'/f'{KIND}.json';hypothesis=read_json(hypothesis_path)
    assert source_hash()==hypothesis['implementation_sha256']
    assert read_json(BASE/'preflight/NEXT3_PREFLIGHT.json')['status']=='PASS'
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    base=BASE/'experiments'/KIND;out=base/args.scope
    if args.scope=='full' and not read_json(base/'screen/RESULTS.json')['screen_pass']:
        raise RuntimeError('Failed screen cannot receive full CV')
    if out.exists():raise RuntimeError('Preserve prior attempt; no overwrite')
    out.mkdir(parents=True)
    atomic_json(out/'POLICY.json',{'kind':KIND,'scope':args.scope,'created_utc':utc_now(),
        'hypothesis_sha256':digest(hypothesis_path),'script_sha256':digest(Path(__file__)),
        'implementation_sha256':source_hash()},immutable=True)
    data=prepare_dev(ROOT);spec,_=candidate(J102)
    original,metadata,names,sources,global_rows=design(data,spec,args.scope)
    cache=ROOT/'data/next3/negative_bayes'/args.scope
    cache.mkdir(parents=True,exist_ok=True)
    identity={'implementation_sha256':source_hash(),'global_rows_sha256':__import__('hashlib').sha256(global_rows.tobytes()).hexdigest(),
        'dev_manifest_sha256':digest(data.directory/'MANIFEST.json'),'feature_names':NAMES}
    if (cache/'MANIFEST.json').exists():
        manifest=read_json(cache/'MANIFEST.json');assert manifest['status']=='COMPLETE'
        assert manifest['identity']=={**identity,'feature_names':list(NAMES)}
        assert digest(cache/'features.npy')==manifest['sha256']
    else:
        extra=np.lib.format.open_memmap(cache/'features.npy',mode='w+',dtype=np.float32,shape=(len(global_rows),4))
        pointer=0;count=0;allowed=set(metadata['dataset_id'])
        for sid in data.ids:
            if sid not in allowed:continue
            h,o,tau=data.get(int(sid))
            values=negative_feature_replay(h,o)
            extra[pointer:pointer+len(o)]=values;pointer+=len(o);count+=1
            if count%500==0:
                progress('Negative Bayes feature extension',f'{args.scope}: {count} DEV series; all prior 77 inputs reused',experiment=KIND)
                print('NEXT3 NEGATIVE BAYES FEATURES',args.scope,count,flush=True)
        assert pointer==len(global_rows)
        extra.flush();del extra
        atomic_json(cache/'MANIFEST.json',{'status':'COMPLETE','identity':identity,
            'sha256':digest(cache/'features.npy'),'seal_rows':0},immutable=True)
    extra=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    x=np.empty((len(original),len(names)+4),dtype=np.float32);x[:,:len(names)]=original;x[:,-4:]=extra
    del original
    all_names=names+list(NAMES)
    records=[];folds=policy['screen']['folds'] if args.scope=='screen' else list(range(5))
    for fold in folds:
        train=np.flatnonzero(metadata['fold']!=fold);valid=np.flatnonzero(metadata['fold']==fold)
        atomic_json(out/f'fold_{fold}_FIT_START.json',{'created_utc':utc_now(),'intended_new_learner_fits':1},immutable=True)
        progress('Negative Bayes primary fitting',f'{args.scope} fold {fold}: one 100-tree W3 learner',experiment=KIND)
        fitted,new_primary,detail=fit_binary(x,metadata,train,valid,guard=data.guard,config=EngineConfig(**spec['engine_config']),
            names=all_names,learner=spec['learner'],weighting=spec['weighting'],params=spec['params'])
        model=NegativeBayesBundle.wrap(fitted,spec);model.save(out/f'fold_{fold}_primary.joblib')
        loaded=NegativeBayesBundle.load(out/f'fold_{fold}_primary.joblib')
        np.testing.assert_array_equal(np.asarray(loaded.predict(x[valid[:2048]]),dtype=np.float32),new_primary[:2048])
        # Verify actual streaming features from a complete validation series.
        sid=int(metadata['dataset_id'][valid[0]]);h,o,tau=data.get(sid);state=loaded.make_state(h)
        actual=np.asarray([state.update(point).copy() for point in o],dtype=np.float32)
        series_rows=np.flatnonzero(metadata['dataset_id']==sid)
        np.testing.assert_array_equal(actual,x[series_rows])
        _,old_primary,jrows,jproof=fold_artifact(data,J102,fold,args.scope)
        _,complement,crows,cproof=fold_artifact(data,C205,fold,args.scope)
        np.testing.assert_array_equal(global_rows[valid],jrows);np.testing.assert_array_equal(jrows,crows)
        prediction=combine(new_primary,complement,.8);reference=combine(old_primary,complement,.8)
        y,t=metadata['target'][valid],metadata['time_online'][valid]
        score=ts_auc(y,prediction,t);refscore=ts_auc(y,reference,t)
        np.save(out/f'fold_{fold}.npy',prediction,allow_pickle=False)
        np.save(out/f'fold_{fold}_global_rows.npy',global_rows[valid],allow_pickle=False)
        np.save(out/f'fold_{fold}_primary.npy',new_primary,allow_pickle=False)
        record={'status':'PASS','fold':fold,'ts_auc':score,'reference_ts_auc':refscore,'delta':score-refscore,
            'modified_primary_ts_auc':ts_auc(y,new_primary,t),'frozen_J102_ts_auc':ts_auc(y,old_primary,t),
            'training':detail,'new_fits':1,'seal_rows':0,'streaming_feature_parity':True,
            'feature_count':len(all_names),'frozen_complement_sources':cproof,
            'files_sha256':{name:digest(out/name) for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy',f'fold_{fold}_primary.npy',f'fold_{fold}_primary.joblib')}}
        atomic_json(out/f'fold_{fold}.json',record,immutable=True);records.append(record)
        print('NEXT3 NEGATIVE BAYES',args.scope,fold,score,score-refscore,flush=True)
    scores=np.array([r['ts_auc'] for r in records]);refs=np.array([r['reference_ts_auc'] for r in records])
    result={'status':'COMPLETE','kind':KIND,'scope':args.scope,'created_utc':utc_now(),'fold_scores':scores.tolist(),
        'reference_scores':refs.tolist(),'fold_records':records,'mean':float(scores.mean()),'median':float(np.median(scores)),
        'worst_fold':float(scores.min()),'mean_delta':float(scores.mean()-refs.mean()),
        'median_delta':float(np.median(scores)-np.median(refs)),'worst_delta':float(scores.min()-refs.min()),
        'new_fits':len(folds),'seal_rows':0,'feature_names':all_names,'feature_sources':sources,
        'negative_feature_manifest_sha256':digest(cache/'MANIFEST.json'),'policy_sha256':digest(out/'POLICY.json')}
    if args.scope=='screen':
        result['screen_pass']=bool(result['mean_delta']>=policy['screen']['mean_delta_min'] and (scores-refs>=policy['screen']['each_fold_delta_min']).all())
        result['scientific_disposition']='SCREEN_PASS' if result['screen_pass'] else 'NEGATIVE_RESULT'
    else:
        gate=hypothesis['finalist_admission']
        result['finalist_admitted']=bool(result['mean_delta']>=gate['mean_delta_min'] and result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
        result['scientific_disposition']='FULL_VALIDATION_ELIGIBLE' if result['finalist_admitted'] else 'NEGATIVE_RESULT'
    atomic_json(out/'RESULTS.json',result,immutable=True);atomic_json(base/'RESULTS.json',result)
    print('NEXT3 NEGATIVE BAYES RESULT',args.scope,result['scientific_disposition'],result['mean_delta'],flush=True)


if __name__=='__main__':main()

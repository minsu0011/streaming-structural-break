"""Append H-only scale calibration to exact J102; keep fixed C205 and 80/20 blend."""
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next3.frozen import fold_artifact,J102
from src.next.data import prepare_dev
from src.next2.io import design,fit,model_class
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['S1_LOWER_SEMISCALE','S2_LOWER_QUANTILE'],required=True)
    parser.add_argument('--scope',choices=['screen','full'],required=True);args=parser.parse_args()
    hyp_path=BASE/'hypotheses'/f'{args.kind}.json';hyp=read_json(hyp_path);spec=hyp['candidate']
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    assert digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')==hyp['policy_sha256']
    base=BASE/'experiments'/args.kind;out=base/args.scope
    if args.scope=='full' and not read_json(base/'screen/RESULTS.json')['screen_pass']:
        raise RuntimeError('Failed screen cannot proceed')
    if out.exists():raise RuntimeError('Preserve prior attempt')
    out.mkdir(parents=True)
    atomic_json(out/'POLICY.json',{'created_utc':utc_now(),'kind':args.kind,'scope':args.scope,
        'hypothesis_sha256':digest(hyp_path),'script_sha256':digest(Path(__file__))},immutable=True)
    data=prepare_dev(ROOT);x,metadata,names,sources,global_rows=design(data,spec,args.scope)
    assert len(names)==85 and names[-34:]==spec['keep_names']
    records=[];folds=policy['screen']['folds'] if args.scope=='screen' else list(range(5))
    for fold in folds:
        train=np.flatnonzero(metadata['fold']!=fold);valid=np.flatnonzero(metadata['fold']==fold)
        atomic_json(out/f'fold_{fold}_FIT_START.json',{'created_utc':utc_now(),'intended_new_learner_fits':1},immutable=True)
        progress('H-only scale calibration primary fitting',f'{args.kind} {args.scope} fold {fold}: {len(names)} features',experiment=args.kind)
        model,cp,detail=fit(data,spec,x,metadata,names,train,valid)
        model.save(out/f'fold_{fold}_primary.joblib')
        loaded=model_class(spec).load(out/f'fold_{fold}_primary.joblib')
        np.testing.assert_array_equal(np.asarray(loaded.predict(x[valid[:2048]]),dtype=np.float32),cp[:2048])
        sid=int(metadata['dataset_id'][valid[0]]);h,o,tau=data.get(sid);state=loaded.make_state(h)
        actual=np.asarray([state.update(point).copy() for point in o],dtype=np.float32)
        np.testing.assert_array_equal(actual,x[metadata['dataset_id']==sid])
        _,jp,jrows,jproof=fold_artifact(data,J102,fold,args.scope)
        from src.next3.frozen import C205
        _,oldc,crows,cproof=fold_artifact(data,C205,fold,args.scope)
        np.testing.assert_array_equal(global_rows[valid],jrows);np.testing.assert_array_equal(jrows,crows)
        prediction=combine(cp,oldc,.8);reference=combine(jp,oldc,.8)
        y,t=metadata['target'][valid],metadata['time_online'][valid]
        score=ts_auc(y,prediction,t);ref=ts_auc(y,reference,t)
        np.save(out/f'fold_{fold}.npy',prediction,allow_pickle=False)
        np.save(out/f'fold_{fold}_global_rows.npy',global_rows[valid],allow_pickle=False)
        np.save(out/f'fold_{fold}_primary.npy',cp,allow_pickle=False)
        record={'status':'PASS','fold':fold,'ts_auc':score,'reference_ts_auc':ref,'delta':score-ref,
            'standalone_ts_auc':ts_auc(y,cp,t),'frozen_J102_standalone_ts_auc':ts_auc(y,jp,t),
            'training':detail,'new_fits':1,'seal_rows':0,'feature_count':len(names),'streaming_feature_parity':True,
            'frozen_J102_sources':jproof,'frozen_C205_sources':cproof,
            'files_sha256':{name:digest(out/name) for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy',f'fold_{fold}_primary.npy',f'fold_{fold}_primary.joblib')}}
        atomic_json(out/f'fold_{fold}.json',record,immutable=True);records.append(record)
        print('NEXT3 SCALE CALIBRATION',args.kind,args.scope,fold,score,score-ref,flush=True)
    scores=np.array([r['ts_auc'] for r in records]);refs=np.array([r['reference_ts_auc'] for r in records])
    result={'status':'COMPLETE','kind':args.kind,'scope':args.scope,'created_utc':utc_now(),'candidate':spec,
        'fold_scores':scores.tolist(),'reference_scores':refs.tolist(),'fold_records':records,
        'mean':float(scores.mean()),'median':float(np.median(scores)),'worst_fold':float(scores.min()),
        'mean_delta':float(scores.mean()-refs.mean()),'median_delta':float(np.median(scores)-np.median(refs)),
        'worst_delta':float(scores.min()-refs.min()),'new_fits':len(folds),'seal_rows':0,
        'feature_names':names,'feature_sources':sources,'policy_sha256':digest(out/'POLICY.json')}
    if args.scope=='screen':
        result['screen_pass']=bool(result['mean_delta']>=policy['screen']['mean_delta_min'] and (scores-refs>=policy['screen']['each_fold_delta_min']).all())
        result['scientific_disposition']='SCREEN_PASS' if result['screen_pass'] else 'NEGATIVE_RESULT'
    else:
        gate=hyp['finalist_admission']
        result['finalist_admitted']=bool(result['mean_delta']>=gate['mean_delta_min'] and result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
        result['scientific_disposition']='FULL_VALIDATION_ELIGIBLE' if result['finalist_admitted'] else 'NEGATIVE_RESULT'
    atomic_json(out/'RESULTS.json',result,immutable=True);atomic_json(base/'RESULTS.json',result)
    print('NEXT3 SCALE CALIBRATION RESULT',args.kind,args.scope,result['scientific_disposition'],result['mean_delta'],flush=True)


if __name__=='__main__':main()

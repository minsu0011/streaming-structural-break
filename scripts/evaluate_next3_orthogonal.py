"""Evaluate one residual recipe with strict grouped inner base OOF."""
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.orthogonal import KIND,fit_residual,predict_residual,residual_values
from src.next.data import prepare_dev
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['screen','full'],required=True);args=parser.parse_args()
    path=BASE/'hypotheses'/f'{KIND}.json';hyp=read_json(path)
    for name,sha in hyp['sources_sha256'].items():assert digest(ROOT/name)==sha
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    assert digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')==hyp['policy_sha256']
    base=BASE/'experiments'/KIND;out=base/args.scope
    if args.scope=='full' and not read_json(base/'screen/RESULTS.json')['screen_pass']:raise RuntimeError('Screen rejection is binding')
    if out.exists():raise RuntimeError('Preserve existing experiment')
    out.mkdir(parents=True)
    nested=BASE/'nested'/args.scope;assert read_json(nested/'RESULTS.json')['status']=='PASS'
    atomic_json(out/'POLICY.json',{'created_utc':utc_now(),'hypothesis_sha256':digest(path),
        'script_sha256':digest(Path(__file__)),'nested_sha256':digest(nested/'RESULTS.json')},immutable=True)
    data=prepare_dev(ROOT);metadata=data.rows();records=[]
    for fold in (policy['screen']['folds'] if args.scope=='screen' else range(5)):
        source=nested/f'outer_{fold}';receipt=read_json(source/'RESULTS.json')
        assert receipt['status']=='PASS' and receipt['outer_validation_never_used_for_inner_training']
        for name,sha in receipt['files_sha256'].items():assert digest(source/name)==sha
        for child in receipt['lineage']:
            assert digest(ROOT/child['receipt'])==child['sha256']
            assert child['binding']['outer_validation_intersection']==0
        train=np.load(source/'TRAIN_GLOBAL_ROWS.npy',allow_pickle=False);valid=np.load(source/'VALID_GLOBAL_ROWS.npy',allow_pickle=False)
        data.guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
        assert np.all(metadata['fold'][train]!=fold) and np.all(metadata['fold'][valid]==fold)
        xtrain=np.load(source/'GATE_TRAIN.npy',mmap_mode='r',allow_pickle=False);xvalid=np.load(source/'GATE_VALID.npy',mmap_mode='r',allow_pickle=False)
        atomic_json(out/f'fold_{fold}_FIT_START.json',{'created_utc':utc_now(),'intended_residual_fits':1,'new_base_fits':0},immutable=True)
        progress('Strictly nested residual fit',f'{args.scope} fold {fold}: C205 explained by J102 and scale only',experiment=KIND)
        model=fit_residual(xtrain,{k:v[train] for k,v in metadata.items()})
        atomic_json(out/f'fold_{fold}_residual.json',model,immutable=True)
        prediction=predict_residual(model,xvalid)
        np.testing.assert_array_equal(prediction,predict_residual(read_json(out/f'fold_{fold}_residual.json'),xvalid))
        reference=combine(xvalid[:,0],xvalid[:,1],.8);y,t=metadata['target'][valid],metadata['time_online'][valid]
        score=ts_auc(y,prediction,t);ref=ts_auc(y,reference,t)
        np.save(out/f'fold_{fold}.npy',prediction,allow_pickle=False);np.save(out/f'fold_{fold}_global_rows.npy',valid,allow_pickle=False)
        record={'status':'PASS','fold':fold,'ts_auc':score,'reference_ts_auc':ref,'delta':score-ref,
            'frozen_J102_ts_auc':ts_auc(y,xvalid[:,0],t),'new_residual_fits':1,'new_base_fits':0,
            'training_detail':model,'residual_validation_mean':float(residual_values(model,xvalid).mean()),
            'clip_fraction':float(np.mean((prediction==0)|(prediction==1))),
            'strict_inner_base_OOF':True,'seal_rows':0,'serialization_prediction_exact':True,
            'files_sha256':{name:digest(out/name) for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy',f'fold_{fold}_residual.json')}}
        atomic_json(out/f'fold_{fold}.json',record,immutable=True);records.append(record)
        print('NEXT3 RESIDUAL',args.scope,fold,score,score-ref,flush=True)
    scores=np.array([r['ts_auc'] for r in records]);refs=np.array([r['reference_ts_auc'] for r in records])
    result={'status':'COMPLETE','kind':KIND,'scope':args.scope,'created_utc':utc_now(),'fold_records':records,
        'fold_scores':scores.tolist(),'reference_scores':refs.tolist(),'mean':float(scores.mean()),
        'median':float(np.median(scores)),'worst_fold':float(scores.min()),'mean_delta':float(scores.mean()-refs.mean()),
        'median_delta':float(np.median(scores)-np.median(refs)),'worst_delta':float(scores.min()-refs.min()),
        'new_fits':len(records),'new_base_fits':0,'strict_inner_base_OOF':True,'seal_rows':0}
    if args.scope=='screen':
        result['screen_pass']=bool(result['mean_delta']>=policy['screen']['mean_delta_min'] and (scores-refs>=policy['screen']['each_fold_delta_min']).all())
        result['scientific_disposition']='SCREEN_PASS' if result['screen_pass'] else 'NEGATIVE_RESULT'
    else:
        gate=hyp['finalist_admission'];result['finalist_admitted']=bool(result['mean_delta']>=gate['mean_delta_min'] and result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
        result['scientific_disposition']='FULL_VALIDATION_ELIGIBLE' if result['finalist_admitted'] else 'NEGATIVE_RESULT'
    atomic_json(out/'RESULTS.json',result,immutable=True);atomic_json(base/'RESULTS.json',result)
    print('NEXT3 RESIDUAL RESULT',result['scientific_disposition'],result['mean_delta'],flush=True)


if __name__=='__main__':main()

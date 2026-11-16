"""Evaluate tiny gates using only preregistered strictly nested training rows."""
from pathlib import Path
import sys
import argparse
import json
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.learned_router import fit_gate
from src.next3.gate_serialization import describe_gate,descriptor_predict
from src.next.data import prepare_dev
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['R2_LOGISTIC_GATE','R3_DEPTH2_GATE'],required=True)
    parser.add_argument('--scope',choices=['screen','full'],required=True);args=parser.parse_args()
    hypothesis_path=BASE/'hypotheses'/f'{args.kind}.json';hypothesis=read_json(hypothesis_path)
    for name,expected in hypothesis['sources_sha256'].items():assert digest(ROOT/'src/next3'/name)==expected
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    assert digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')==hypothesis['policy_sha256']
    assert read_json(BASE/'preflight/NEXT3_PREFLIGHT.json')['status']=='PASS'
    base=BASE/'experiments'/args.kind;out=base/args.scope
    if args.scope=='full' and not read_json(base/'screen/RESULTS.json')['screen_pass']:
        raise RuntimeError('Failed screen cannot receive full CV')
    if out.exists():raise RuntimeError('Preserve prior attempt; do not overwrite')
    out.mkdir(parents=True)
    nested=BASE/'nested'/args.scope
    assert read_json(nested/'RESULTS.json')['status']=='PASS'
    atomic_json(out/'POLICY.json',{'created_utc':utc_now(),'kind':args.kind,'scope':args.scope,
        'hypothesis_sha256':digest(hypothesis_path),'nested_scope_sha256':digest(nested/'RESULTS.json'),
        'script_sha256':digest(Path(__file__)),'serialization_sha256':digest(ROOT/'src/next3/gate_serialization.py')},immutable=True)
    data=prepare_dev(ROOT);metadata=data.rows()
    records=[]
    folds=policy['screen']['folds'] if args.scope=='screen' else list(range(5))
    for fold in folds:
        directory=nested/f'outer_{fold}'
        receipt=read_json(directory/'RESULTS.json')
        assert receipt['status']=='PASS' and receipt['outer_validation_never_used_for_inner_training']
        for name,expected in receipt['files_sha256'].items():assert digest(directory/name)==expected
        for child in receipt['lineage']:
            assert digest(ROOT/child['receipt'])==child['sha256']
            assert child['binding']['outer_validation_intersection']==0
        train=np.load(directory/'TRAIN_GLOBAL_ROWS.npy',allow_pickle=False)
        valid=np.load(directory/'VALID_GLOBAL_ROWS.npy',allow_pickle=False)
        data.guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
        assert np.all(metadata['fold'][train]!=fold) and np.all(metadata['fold'][valid]==fold)
        xtrain=np.load(directory/'GATE_TRAIN.npy',mmap_mode='r',allow_pickle=False)
        xvalid=np.load(directory/'GATE_VALID.npy',mmap_mode='r',allow_pickle=False)
        train_meta={k:v[train] for k,v in metadata.items()}
        progress('Nested gate fitting',f'{args.kind} {args.scope} fold {fold}; base predictions exclusively inner OOF',experiment=args.kind)
        tick=time.monotonic()
        atomic_json(out/f'fold_{fold}_FIT_START.json',{'created_utc':utc_now(),'kind':args.kind,
            'intended_gate_fits':1,'new_base_fits':0,'nested_outer_receipt_sha256':digest(directory/'RESULTS.json')},immutable=True)
        model=fit_gate(args.kind,xtrain,train_meta)
        prediction=model.predict(xvalid);weight=model.weights(xvalid)
        descriptor=describe_gate(model)
        atomic_json(out/f'fold_{fold}_gate.json',descriptor,immutable=True)
        np.testing.assert_array_equal(descriptor_predict(read_json(out/f'fold_{fold}_gate.json'),xvalid),prediction)
        baseline=combine(xvalid[:,0],xvalid[:,1],.8)
        y,t=metadata['target'][valid],metadata['time_online'][valid]
        score=ts_auc(y,prediction,t);ref=ts_auc(y,baseline,t)
        np.save(out/f'fold_{fold}.npy',prediction,allow_pickle=False)
        np.save(out/f'fold_{fold}_global_rows.npy',valid,allow_pickle=False)
        record={'status':'PASS','fold':fold,'ts_auc':score,'reference_ts_auc':ref,'delta':score-ref,
            'gate_training':model.training_detail,'gate_fit_seconds':time.monotonic()-tick,
            'new_gate_fits':1,'new_base_fits':0,'nested_outer_sha256':digest(directory/'RESULTS.json'),
            'weight_mean':float(weight.mean()),'weight_min':float(weight.min()),'weight_max':float(weight.max()),
            'weight_quantiles':{str(q):float(np.quantile(weight,q)) for q in (.05,.5,.95)},
            'serialization_prediction_exact':True,'seal_rows':0,
            'files_sha256':{name:digest(out/name) for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy',f'fold_{fold}_gate.json')}}
        atomic_json(out/f'fold_{fold}.json',record,immutable=True);records.append(record)
        print('NEXT3 LEARNED GATE',args.kind,args.scope,fold,score,score-ref,flush=True)
    scores=np.array([r['ts_auc'] for r in records]);refs=np.array([r['reference_ts_auc'] for r in records])
    result={'status':'COMPLETE','kind':args.kind,'scope':args.scope,'created_utc':utc_now(),
        'fold_scores':scores.tolist(),'reference_scores':refs.tolist(),'fold_records':records,
        'mean':float(scores.mean()),'median':float(np.median(scores)),'worst_fold':float(scores.min()),
        'mean_delta':float(scores.mean()-refs.mean()),'median_delta':float(np.median(scores)-np.median(refs)),
        'worst_delta':float(scores.min()-refs.min()),'new_gate_fits':len(folds),'new_base_fits':0,
        'new_fits':len(folds),'base_fits_reused_from_nested_scope':True,'seal_rows':0,
        'policy_sha256':digest(out/'POLICY.json'),'strict_inner_base_OOF':True}
    if args.scope=='screen':
        result['screen_pass']=bool(result['mean_delta']>=policy['screen']['mean_delta_min'] and
            (scores-refs>=policy['screen']['each_fold_delta_min']).all())
        result['scientific_disposition']='SCREEN_PASS' if result['screen_pass'] else 'NEGATIVE_RESULT'
    else:
        gate=hypothesis['finalist_admission']
        result['finalist_admitted']=bool(result['mean_delta']>=gate['mean_delta_min'] and
            result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
        result['scientific_disposition']='FULL_VALIDATION_ELIGIBLE' if result['finalist_admitted'] else 'NEGATIVE_RESULT'
    atomic_json(out/'RESULTS.json',result,immutable=True);atomic_json(base/'RESULTS.json',result)
    print('NEXT3 LEARNED RESULT',args.kind,args.scope,result['scientific_disposition'],result['mean_delta'],flush=True)


if __name__=='__main__':main()

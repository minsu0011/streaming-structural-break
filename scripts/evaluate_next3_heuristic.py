"""Two-stage evaluation of one preregistered deterministic causal gate."""
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next3.frozen import candidate,fold_artifact,J102,C205
from src.next3.router import gate_design,heuristic_predict,HEURISTICS
from src.next.data import prepare_dev
from src.next2.io import design
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=HEURISTICS,required=True)
    parser.add_argument('--scope',choices=['screen','full'],required=True)
    args=parser.parse_args()
    hypothesis_path=BASE/'hypotheses'/f'{args.kind}.json'
    hypothesis=read_json(hypothesis_path)
    assert digest(ROOT/'src/next3/router.py')==hypothesis['implementation_sha256']
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    assert digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')==hypothesis['policy_sha256']
    assert read_json(BASE/'preflight/NEXT3_PREFLIGHT.json')['status']=='PASS'
    base=BASE/'experiments'/args.kind
    if args.scope=='full':
        screen=read_json(base/'screen/RESULTS.json')
        if not screen['screen_pass']:
            raise RuntimeError('Failed screen cannot receive full CV')
    out=base/args.scope
    if out.exists():
        raise RuntimeError('Preserve existing attempt; investigate incomplete output before any retry')
    out.mkdir(parents=True)
    atomic_json(out/'POLICY.json',{'kind':args.kind,'scope':args.scope,'created_utc':utc_now(),
        'hypothesis_sha256':digest(hypothesis_path),'policy_sha256':hypothesis['policy_sha256'],
        'script_sha256':digest(Path(__file__)),'sources_sha256':{
            'router':digest(ROOT/'src/next3/router.py'),'frozen_loader':digest(ROOT/'src/next3/frozen.py')},
        'new_fits':0},immutable=True)
    data=prepare_dev(ROOT)
    jc,_=candidate(J102);cc,_=candidate(C205)
    jx,metadata,jnames,jsources,global_rows=design(data,jc,args.scope)
    cx,cm,cname,csources,crows=design(data,cc,args.scope)
    np.testing.assert_array_equal(global_rows,crows)
    for key in metadata:np.testing.assert_array_equal(metadata[key],cm[key])
    records=[]
    folds=policy['screen']['folds'] if args.scope=='screen' else list(range(5))
    for fold in folds:
        local=np.flatnonzero(metadata['fold']==fold)
        jm,jp,jrows,jproof=fold_artifact(data,J102,fold,args.scope)
        model,cp,cprows,cproof=fold_artifact(data,C205,fold,args.scope)
        np.testing.assert_array_equal(global_rows[local],jrows)
        np.testing.assert_array_equal(jrows,cprows)
        # Recompute a fixed probe from saved models; full fingerprints and
        # all OOF hashes were independently checked in restoration/forensics.
        probe=local[:2048]
        np.testing.assert_array_equal(np.asarray(jm.predict(jx[probe]),dtype=np.float32),jp[:len(probe)])
        np.testing.assert_array_equal(np.asarray(model.predict(cx[probe]),dtype=np.float32),cp[:len(probe)])
        gate=gate_design(jp,cp,jx[local],cx[local],jnames,cname)
        prediction,weight=heuristic_predict(gate,args.kind)
        reference=combine(jp,cp,.8)
        y,t=metadata['target'][local],metadata['time_online'][local]
        score=ts_auc(y,prediction,t);refscore=ts_auc(y,reference,t)
        np.save(out/f'fold_{fold}.npy',prediction,allow_pickle=False)
        np.save(out/f'fold_{fold}_global_rows.npy',global_rows[local],allow_pickle=False)
        record={'fold':fold,'ts_auc':score,'reference_ts_auc':refscore,'delta':score-refscore,
            'validation_series':len(np.unique(metadata['dataset_id'][local])),'validation_rows':len(local),
            'weight_frequency':{str(w):float(np.mean(weight==w)) for w in (0.,.1,.2,.3)},
            'weight_mean':float(weight.mean()),'seal_rows':0,'new_fits':0,
            'frozen_primary_files_sha256':jproof,'frozen_complement_files_sha256':cproof,
            'files_sha256':{name:digest(out/name) for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy')}}
        atomic_json(out/f'fold_{fold}.json',{'status':'PASS',**record},immutable=True)
        records.append(record)
        progress('Causal heuristic evaluation',f'{args.kind} {args.scope} fold {fold}: delta {score-refscore:+.6f}',experiment=args.kind)
        print('NEXT3 HEURISTIC',args.kind,args.scope,fold,score,score-refscore,flush=True)
    scores=np.array([r['ts_auc'] for r in records]);refs=np.array([r['reference_ts_auc'] for r in records])
    result={'status':'COMPLETE','kind':args.kind,'scope':args.scope,'created_utc':utc_now(),
        'fold_scores':scores.tolist(),'reference_scores':refs.tolist(),'fold_records':records,
        'mean':float(scores.mean()),'median':float(np.median(scores)),'worst_fold':float(scores.min()),
        'mean_delta':float(scores.mean()-refs.mean()),'median_delta':float(np.median(scores)-np.median(refs)),
        'worst_delta':float(scores.min()-refs.min()),'new_fits':0,'seal_rows':0,
        'causal_inputs_only':True,'posthoc_mechanism_labels_used':False,
        'feature_sources':{'J102':jsources,'C205':csources},'policy_sha256':digest(out/'POLICY.json')}
    if args.scope=='screen':
        result['screen_pass']=bool(result['mean_delta']>=policy['screen']['mean_delta_min'] and
            (scores-refs>=policy['screen']['each_fold_delta_min']).all())
        result['scientific_disposition']='SCREEN_PASS' if result['screen_pass'] else 'NEGATIVE_RESULT'
    else:
        gate=hypothesis['finalist_admission']
        result['finalist_admitted']=bool(result['mean_delta']>=gate['mean_delta_min'] and
            result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
        result['scientific_disposition']='FULL_VALIDATION_ELIGIBLE' if result['finalist_admitted'] else 'NEGATIVE_RESULT'
    atomic_json(out/'RESULTS.json',result,immutable=True)
    atomic_json(base/'RESULTS.json',result)
    print('NEXT3 HEURISTIC RESULT',args.kind,args.scope,result['scientific_disposition'],result['mean_delta'],flush=True)


if __name__=='__main__':main()

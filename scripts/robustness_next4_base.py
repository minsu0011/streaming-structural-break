"""Fixed ALT/H replication for an unchanged registered causal base candidate."""
from pathlib import Path
import sys
import argparse
import hashlib
import time
import numpy as np
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.cache import build_design
from src.next4.model import fit_base,FeatureModel,source_hash
from src.next4.robustness import partitions,frozen_predictions
from src.next.data import prepare_dev
from src.next2.blend import combine
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    hyp_path=BASE/'registrations'/(args.candidate+'.json');hyp=read_json(hyp_path)
    assert hyp['family'] in ('A','B','C','D','G')
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    nomination_path=BASE/'robustness_nominations'/(args.candidate+'.json');nomination=read_json(nomination_path)
    coordinate_audit=read_json(BASE/'stress_coordinate_audit/RESULTS.json')
    assert coordinate_audit['status']=='PASS' and coordinate_audit['source_sha256']==digest(ROOT/'src/next4/robustness.py')
    assert nomination['candidate_registration_sha256']==digest(hyp_path)
    for path,sha in nomination['execution_source_sha256'].items():assert digest(ROOT/path)==sha
    assert hyp['model_source_sha256']==source_hash() and hyp['policy_sha256']==digest(policy_path)
    source=BASE/'experiments'/args.candidate/'full';full=read_json(source/'RESULTS.json')
    assert full['status']=='COMPLETE' and digest(source/'RESULTS.json')==nomination['primary_result_sha256']
    outroot=BASE/'robustness'/args.candidate;outroot.mkdir(parents=True,exist_ok=False)
    atomic_json(outroot/'START.json',{'created_utc':utc_now(),'nomination_sha256':digest(nomination_path),'intended_new_fits':20,
        'frozen_base_new_fits':0,'new_candidate_identity':False},immutable=True)
    data=prepare_dev(ROOT);x,meta,names,global_rows,cache=build_design(data,hyp,'full')
    np.testing.assert_array_equal(global_rows,np.arange(len(global_rows)))
    splits,splitproof=partitions(data,meta);summaries=[];set_num_threads(2)
    for split,folds in splits.items():
        out=outroot/split;out.mkdir(exist_ok=False);records=[];new=[];old=[]
        for fold,(train,valid) in enumerate(folds):
            baseline,baseline_sources=frozen_predictions(data,split,fold,global_rows[valid])
            atomic_json(out/f'fold_{fold}_FIT_START.json',{'created_utc':utc_now(),'intended_new_fits':1,'new_frozen_base_fits':0,
                'training_rows_sha256':hashlib.sha256(global_rows[train].tobytes()).hexdigest(),
                'validation_rows_sha256':hashlib.sha256(global_rows[valid].tobytes()).hexdigest(),
                'nomination_sha256':digest(nomination_path),'splitproof':splitproof},immutable=True)
            progress('NEXT4 fixed held-out refit',args.candidate+' '+split+' fold '+str(fold))
            start=time.perf_counter();model,expert,detail=fit_base(x,meta,train,valid,guard=data.guard,registration=hyp)
            model_path=out/f'fold_{fold}_expert.joblib';model.save(model_path);loaded=FeatureModel.load(model_path)
            np.testing.assert_array_equal(np.asarray(loaded.predict(x[valid[:2048]]),dtype=np.float32),expert[:2048])
            sid=int(meta['dataset_id'][valid[0]]);h,o,_=data.get(sid);state=loaded.make_state(h)
            live=np.array([loaded.predict_one(state.update(point)) for point in o],dtype=np.float32)
            np.testing.assert_array_equal(live,expert[meta['dataset_id'][valid]==sid])
            prediction=combine(expert,baseline['C205'],hyp['parameters']['new_J102_weight']) if hyp['family'] in ('B','G') else combine(baseline['BL80'],expert,hyp['parameters']['frozen_BL80_weight'])
            y,t,ids=(meta[k][valid] for k in ('target','time_online','dataset_id'))
            score=ts_auc(y,prediction,t);ref=ts_auc(y,baseline['BL80'],t)
            code,counts=draw_group_multiplicities(ids,2000,policy['bootstrap']['seed']+fold)
            a=PreparedAUC.create(y,prediction,t,code,counts.shape[1]).evaluate(counts)
            b=PreparedAUC.create(y,baseline['BL80'],t,code,counts.shape[1]).evaluate(counts)
            np.testing.assert_allclose(a[0],score,atol=1e-14,rtol=0);np.testing.assert_allclose(b[0],ref,atol=1e-14,rtol=0)
            new.append(a);old.append(b)
            for name,values in [(f'fold_{fold}.npy',prediction),(f'fold_{fold}_expert.npy',expert),(f'fold_{fold}_global_rows.npy',global_rows[valid]),
                (f'fold_{fold}_bootstrap.npy',a),(f'fold_{fold}_reference_bootstrap.npy',b)]:np.save(out/name,values,allow_pickle=False)
            record={'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,'split':split,'fold':fold,'ts_auc':score,
                'reference_ts_auc':ref,'delta':score-ref,'training':detail,'new_fits':1,'frozen_base_new_fits':0,
                'elapsed_seconds':time.perf_counter()-start,'frozen_OOF_sources':baseline_sources,'seal_rows':0,
                'save_load_parity':True,'streaming_cached_parity':True,'paired_draws':2000,
                'files_sha256':{p.name:digest(p) for p in out.glob(f'fold_{fold}*') if p.is_file()}}
            atomic_json(out/f'fold_{fold}.json',record,immutable=True);records.append(record)
            print('NEXT4 STRESS',args.candidate,split,fold,score,'delta',score-ref,flush=True)
        scores=np.array([r['ts_auc'] for r in records]);refs=np.array([r['reference_ts_auc'] for r in records])
        summary={'status':'PASS','candidate':args.candidate,'split':split,'fold_scores':scores.tolist(),'reference_fold_scores':refs.tolist(),
            'mean':float(scores.mean()),'mean_delta':float((scores-refs).mean()),'median':float(np.median(scores)),
            'worst':float(scores.min()),'minimum_matched_fold_delta':float((scores-refs).min()),'new_fits':5,
            'paired_all_time':paired_summary(np.array(old).T,np.array(new).T),'seal_rows':0}
        atomic_json(out/'RESULTS.json',summary,immutable=True);summaries.append(summary)
    alt=[r for r in summaries if r['split'].startswith('ALT_')];h=next(r for r in summaries if r['split']=='LEAVE_ONE_H_REGIME_OUT')
    atomic_json(outroot/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,'summaries':summaries,
        'completed_refits':20,'new_fits':20,'frozen_base_new_fits':0,'ALT_positive_count':sum(r['mean_delta']>0 for r in alt),
        'H_minimum_matched_delta':h['minimum_matched_fold_delta'],'nomination_sha256':digest(nomination_path),
        'splitproof':splitproof,'primary_bootstrap_gate_overridden':False,'promotion':False,'seal_rows':0,
        'scope':'Reused DEV series under existing alternative group assignments, not independent new data; fixed-OOF bootstrap conditional on each refitted model',
        'files_sha256':{split+'/RESULTS.json':digest(outroot/split/'RESULTS.json') for split in splits}},immutable=True)
    data.guard.check_state();progress('NEXT4 fixed ALT/H replication complete',args.candidate+'; original primary admission remains unchanged')


if __name__=='__main__':main()

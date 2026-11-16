"""Bounded K=8/32 pairwise experiments on the frozen calibrated reference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,json,sys,time,gc,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.pairwise import fit_pairwise,implementation_hash
from src.models.calibrated import CalibratedBundle
from src.features.config import CONFIG
from src.features.streaming import feature_names
from src.features.historical_calibration import CalibrationPolicy
from src.validation.splits import FrozenSplit,assert_development_ids
from src.validation.research import summarize
from src.validation.diagnostics import profile
from src.validation.adaptive import ResearchBudget
from src.scoring.ts_auc import ts_auc
from src.data.loader import load_training
from src.utils.artifacts import write_json,utc_now,sha256
from scripts.run_calibrated_research import calibrated_cache
from scripts.run_real_research import admission
from scripts.verify_engineering import benchmark

ANCHOR='M4_lightgbm_ABCD_S3_dbd7ecc59304'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',type=int,choices=[8,32],required=True)
    parser.add_argument('--ar8-followup',action='store_true');args=parser.parse_args()
    if args.ar8_followup and args.pairs!=32:raise ValueError('The prospective AR8 objective follow-up has exactly one K=32 candidate')
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if not budget.can_start(1800):raise RuntimeError('Final audit buffer protected')
    admission(raw_directory=ROOT/'data/raw');split=FrozenSplit(ROOT/'artifacts/validation')
    anchor_id='M4_ar_order8_ABCD_S3_242216e87a34' if args.ar8_followup else ANCHOR
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());anchor=next(r for r in state['results'] if r['experiment_id']==anchor_id)
    candidate=anchor['candidate'];config=CONFIG.variant(**{k:tuple(v) if isinstance(v,list) else v for k,v in candidate['feature_changes'].items()})
    from src.models.representations import calibration_policy,wrap_model
    policy=calibration_policy(candidate['representation']);names=feature_names(config)
    eid=f'M6_pairwise_{"AR8_" if args.ar8_followup else ""}K{args.pairs}_{candidate["families"]}_{implementation_hash()[:12]}'
    result_path=ROOT/'artifacts/real'/f'{eid}.json'
    if result_path.exists():raise RuntimeError('Pairwise candidate already has a result; explicit reproduction required')
    frozen={'anchor':anchor_id,'pairs_per_positive_families':[32] if args.ar8_followup else [8,32],'objective_hash':implementation_hash(),'source_policy':'docs/PAIRWISE_OBJECTIVE_POLICY.md',
        'training_rows':'all development training-fold rows','comparison_change':'row classification objective replaced by sampled within-time pairwise loss',
        'validation_rows':'all rows of the held-out series fold','split_sha256':split.digest,'frozen_before_pairwise_scores':True}
    if args.ar8_followup:frozen['hypothesis']='Historical AR8 innovations materially improved all folds; test the unchanged K32 within-time pair objective once on that stronger representation, with the same LightGBM tree settings.'
    frozen_path=ROOT/'artifacts/real'/('PAIRWISE_AR8_FOLLOWUP_POLICY.json' if args.ar8_followup else 'PAIRWISE_POLICY.json')
    if frozen_path.exists() and json.loads(frozen_path.read_text())!=frozen:raise RuntimeError('Pairwise policy changed after freezing')
    write_json(frozen_path,frozen)
    report={'status':'RUNNING','experiment_id':eid,'timestamp':utc_now(),'model':'M6_pairwise','families':candidate['families'],
        'sampling':'S0_rows_with_within_time_pairs','representation':candidate['representation'],'pair_budget':args.pairs,
        'objective_hash':implementation_hash(),'anchor':anchor_id,'fold_scores':[],'cv_rows':[],'fractures':[],'fold_metadata':[],
        'latencies':[],'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    start=time.perf_counter()
    try:
        if args.ar8_followup:
            source=json.loads((ROOT/'data/processed/oof'/anchor_id/'MANIFEST.json').read_text());cache=ROOT/source['cache_directory']
            manifest=json.loads((cache/'MANIFEST.json').read_text())
            if manifest['split_sha256']!=split.digest or not all(sha256(cache/f'fold_{k}.parquet')==manifest['fold_sha256'][str(k)] for k in range(5)):raise RuntimeError('AR8 feature cache identity changed')
        else:cache=calibrated_cache(ROOT/'data/raw',split,policy,config)
        folds=[pd.read_parquet(cache/f'fold_{k}.parquet') for k in range(5)]
        ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
        metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',ids)])
        sample=next(load_training(ROOT/'data/raw',split=split));h=sample.historical.astype(np.float32);points=np.tile(sample.online,5).astype(np.float32)[:1000]
        for fold,validation in enumerate(folds):
            training=pd.concat([f for k,f in enumerate(folds) if k!=fold],ignore_index=True)
            train_ids=set(training.dataset_id);validation_ids=set(validation.dataset_id);assert_development_ids(train_ids|validation_ids,split)
            if train_ids&validation_ids or {split.groups[i] for i in train_ids}&{split.groups[i] for i in validation_ids}:raise RuntimeError('Pairwise group leakage')
            fold_start=time.perf_counter()
            base,details=fit_pairwise(training.loc[:,names].to_numpy(dtype=np.float32),training.target.to_numpy(),training.time_online.to_numpy(),
                families=candidate['families'],config=config,pairs_per_positive=args.pairs,seed=20260908+fold)
            bundle=wrap_model(base,candidate['representation'],config);model_path=ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib';bundle.save(model_path);loaded=type(bundle).load(model_path)
            matrix=validation.loc[:,names].to_numpy(dtype=np.float32);prediction=loaded.predict(matrix).astype(np.float32)
            np.testing.assert_array_equal(bundle.predict(matrix[:1000]),loaded.predict(matrix[:1000]))
            score=ts_auc(validation.target,prediction,validation.time_online)
            def factory():
                feature_state=loaded.make_state(h)
                return lambda point:loaded.predict_one(feature_state.update_and_get(point))
            latency=benchmark(factory,points,repeats=3);latency['model_bytes']=model_path.stat().st_size
            report['latencies'].append(latency);report['fold_scores'].append(score)
            report['cv_rows'].append({'experiment_id':eid,'model':'M6_pairwise','families':candidate['families'],'sampling':report['sampling'],'fold':fold,'ts_auc':score})
            report['fractures'].extend([{'experiment_id':eid,'fold':fold,**row} for row in profile(validation.target,prediction,validation.time_online,validation.dataset_id,metadata)])
            report['fold_metadata'].append({'fold':fold,'fit_and_validation_seconds':time.perf_counter()-fold_start,'rss_bytes':psutil.Process().memory_info().rss,**details})
            oof=ROOT/'data/processed/oof'/eid;oof.mkdir(parents=True,exist_ok=True);np.save(oof/f'fold_{fold}.npy',prediction,allow_pickle=False)
            write_json(oof/'MANIFEST.json',{'cache_directory':str(cache.relative_to(ROOT)),'split_sha256':split.digest,'feature_hash':bundle.feature_hash,
                'objective_hash':implementation_hash(),'pair_budget':args.pairs,'representation':candidate['representation'],'wire_dtype':'float32'})
            write_json(ROOT/'artifacts/real'/f'{eid}_CHECKPOINT.json',report)
            print(eid,'fold',fold,'TS-AUC',score,'pairs',details['pair_count'],'RSS_GB',round(psutil.Process().memory_info().rss/1024**3,2),flush=True)
            del training,matrix,base,bundle,loaded;gc.collect()
        report.update(summarize(report['fold_scores']));report['status']='COMPLETE';report['latency_us']=float(np.median([x['throughput_median_us_per_point'] for x in report['latencies']]))
    except Exception as exc:
        report['status']='FAILED';report['exception_type']=type(exc).__name__
        (ROOT/'experiments/logs'/f'{eid}_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
    report['elapsed_seconds']=time.perf_counter()-start;write_json(result_path,report)
    print('PAIRWISE',report['status'],eid,'mean',report.get('mean'),'worst',report.get('worst_fold'),flush=True)

if __name__=='__main__':main()

"""Resumable real development research, with wall-clock admission for each hypothesis."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
import argparse
from dataclasses import asdict
import gc,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.cache import build_cache,read_fold
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState,feature_version,feature_names,FEATURE_FAMILIES
from src.models.baselines import BASELINES,BaselineDetector
from src.models.learners import ModelBundle
from src.scoring.ts_auc import ts_auc
from src.validation.research import fit_fold,summarize
from src.validation.diagnostics import profile
from src.validation.splits import FrozenSplit
from src.validation.adaptive import ResearchBudget,select_next,promotion_decision
from src.data.loader import load_training
from src.utils.artifacts import write_json,ledger,utc_now,sha256
from scripts.verify_engineering import benchmark

def admission(root=ROOT,raw_directory=None):
    requirements={
        'artifacts/audits/RAW_DATA_MANIFEST.json':('status','ACQUIRED'),
        'artifacts/audits/DATA_AUDIT.json':('status','PASS'),
        'artifacts/audits/DUPLICATE_AUDIT.json':('status','PASS'),
        'artifacts/validation/SCORER_PARITY.json':('status','PASS'),
        'artifacts/validation/SCORER_PARITY_REAL.json':('status','PASS'),
        'artifacts/validation/LABEL_ALIGNMENT.json':('status','PASS'),
        'artifacts/validation/PREFIX_INVARIANCE.json':('status','PASS'),
        'artifacts/validation/SPLIT_LOCK_REAL.json':('status','LOCKED')}
    failures=[]
    for path,(key,expected) in requirements.items():
        file=Path(root)/path
        if not file.exists() or json.loads(file.read_text(encoding='utf-8')).get(key)!=expected:failures.append(path)
    prefix=Path(root)/'artifacts/validation/PREFIX_INVARIANCE.json'
    if prefix.exists() and json.loads(prefix.read_text(encoding='utf-8')).get('feature_hash')!=feature_version():failures.append('stale_prefix_validation')
    scorer=Path(root)/'artifacts/validation/SCORER_PARITY.json'
    if scorer.exists() and json.loads(scorer.read_text(encoding='utf-8')).get('official_source_sha256')!=sha256(ROOT/'vendor/official/scoring.py'):failures.append('stale_scorer_validation')
    lock=Path(root)/'artifacts/validation/SPLIT_LOCK_REAL.json'
    if lock.exists():
        try:FrozenSplit(Path(root)/'artifacts/validation')
        except Exception:failures.append('split_integrity_failure')
    if raw_directory is not None:
        raw=Path(root)/'artifacts/audits/RAW_DATA_MANIFEST.json'
        if raw.exists():
            for record in json.loads(raw.read_text(encoding='utf-8')).get('files',[]):
                path=Path(raw_directory)/record['name']
                if not path.exists() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:failures.append('raw_identity_mismatch:'+record['name'])
    if failures:raise RuntimeError('Research admission blocked: '+', '.join(failures))

class RealResearch:
    def __init__(self,raw,budget):
        self.raw=Path(raw);self.budget=budget
        self.directory=ROOT/'artifacts/real';self.directory.mkdir(parents=True,exist_ok=True)
        self.state_path=self.directory/'RESEARCH_STATE.json'
        self.state=json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {
            'created_utc':utc_now(),'results':[],'attempted':[],'failed':[],'cv_rows':[],'fractures':[],
            'promotions':[],'primary_id':None,'robustness_completed':[],
            'LOCAL_FINAL_SEAL_OPENED':False,'reduced_scored_uses':0}
        self.split=FrozenSplit(ROOT/'artifacts/validation')
        dev_ids=set(self.split.frame.loc[self.split.frame.role=='DEVELOPMENT_POOL','dataset_id'])
        self.metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',sorted(dev_ids))])
        first=next(load_training(self.raw,split=self.split))
        self.historical=first.historical.astype(np.float32)
        self.points=np.tile(first.online.astype(np.float32),max(1,5000//len(first.online)+1))[:5000]
        self.loaded_config=None;self.folds=None;self.cache=None

    def load_features(self,config,representation='base'):
        key=(config.digest(),representation)
        if self.loaded_config==key:return
        self.folds=None;gc.collect()
        if representation=='base':
            self.cache=build_cache(self.raw,ROOT/'data/cache',split=self.split,config=config)
            self.folds=[read_fold(self.cache,k,split=self.split,config=config) for k in range(5)]
        elif representation.startswith('ar8_arch1_'):
            from src.features.arch_cache import build_arch_cache
            from src.models.representations import calibration_policy
            self.cache=build_arch_cache(self.raw,ROOT/'data/cache',self.split,config,calibration_policy(representation))
            self.folds=[pd.read_parquet(self.cache/f'fold_{k}.parquet') for k in range(5)]
        elif representation=='cross_order_ar4_ar8_equal':
            from src.features.cross_order_cache import build_cross_order_cache
            self.cache=build_cross_order_cache(self.raw,ROOT/'data/cache',self.split,config)
            self.folds=[pd.read_parquet(self.cache/f'fold_{k}.parquet') for k in range(5)]
        elif representation.startswith(('ar_residual_input_', 'ar_order2_', 'ar_order8_')):
            from src.features.ar_cache import build_ar_cache
            from src.models.representations import calibration_policy, ar_input_order
            self.cache=build_ar_cache(self.raw,ROOT/'data/cache',self.split,config,calibration_policy(representation),input_order=ar_input_order(representation))
            self.folds=[pd.read_parquet(self.cache/f'fold_{k}.parquet') for k in range(5)]
        else:
            from src.features.historical_calibration import CalibrationPolicy
            from scripts.run_calibrated_research import calibrated_cache
            self.cache=calibrated_cache(self.raw,self.split,CalibrationPolicy(method=representation),config)
            self.folds=[pd.read_parquet(self.cache/f'fold_{k}.parquet') for k in range(5)]
        self.loaded_config=key
        print(f'Loaded development-only cache: {sum(map(len,self.folds))} rows; {config.digest()[:12]}; {representation}',flush=True)

    def save(self):
        usage_path=ROOT/'REDUCED_TEST_USAGE_LEDGER.json'
        uses=json.loads(usage_path.read_text(encoding='utf-8'))['scored_uses'] if usage_path.exists() else 0
        self.state['reduced_scored_uses']=uses
        self.state['updated_utc']=utc_now();write_json(self.state_path,self.state)
        results=self.state['results']
        frame=pd.DataFrame([{k:v for k,v in r.items() if k not in ('candidate','latency','fold_scores')} for r in results])
        if len(frame):
            frame.to_csv(ROOT/'MODEL_RESULTS_REAL.csv',index=False)
            frame.loc[frame.model.str.startswith('B')|(frame.phase=='complementary_statistical')].to_csv(ROOT/'BASELINE_RESULTS_REAL.csv',index=False)
            frame.loc[frame.phase.isin(['family_ablation','calibrated_family_refinement','memory_crossscale_ablation'])].to_csv(ROOT/'FEATURE_ABLATIONS_REAL.csv',index=False)
            frame.loc[frame.phase.isin(['sampling_weighting','calibrated_sampling_refinement','robust_ar8_weight_followup'])].to_csv(ROOT/'SAMPLING_WEIGHTING_RESULTS.csv',index=False)
        pd.DataFrame(self.state['cv_rows']).to_csv(ROOT/'CV_RESULTS_REAL.csv',index=False)
        fractures=pd.DataFrame(self.state['fractures'])
        if len(fractures):
            fractures.loc[fractures.kind=='online_time'].to_csv(ROOT/'TIME_FRACTURE_REAL.csv',index=False)
            fractures.loc[fractures.kind=='evidence_age_with_negative_controls'].to_csv(ROOT/'BREAK_AGE_DIAGNOSTICS.csv',index=False)
            fractures.to_csv(self.directory/'ALL_FRACTURES_REAL.csv',index=False)
        write_json(ROOT/'LATENCY_REAL.json',{'scope':'REAL_TRAIN_DEVELOPMENT','candidates':{r['experiment_id']:r['latency'] for r in results}})
        active=self.state.get('active_candidate')
        self.budget.checkpoint(status='RESEARCH_ACTIVE',completed_candidates=len(results),failed_candidates=len(self.state['failed']),primary_id=self.state['primary_id'],reduced_scored_uses=uses,
            active_candidate=active,next_action='Complete the active five-fold candidate, then select the next evidence-based hypothesis.' if active else 'Select the next evidence-based hypothesis or enter final audit at the fixed cutoff.')
        status_path=ROOT/'01_PROJECT_STATUS.json'
        status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        primary=next((r for r in results if r['experiment_id']==self.state['primary_id']),None)
        status.update(status='REAL_DEVELOPMENT_RESEARCH_ACTIVE',timestamp=utc_now(),completed_candidates=len(results),
            primary_dev_candidate_so_far=self.state['primary_id'],primary_dev_mean=primary['mean'] if primary else None,
            primary_dev_median=primary['median'] if primary else None,primary_dev_worst_fold=primary['worst_fold'] if primary else None,
            LOCAL_FINAL_SEAL_OPENED=False,reduced_scored_uses=uses,ten_hour_real_data_research_completed=False)
        write_json(status_path,status)

    def profiles(self,eid,fold,frame,prediction):
        return [{'experiment_id':eid,'fold':fold,**row} for row in profile(frame.target,prediction,frame.time_online,frame.dataset_id,self.metadata)]

    def baselines(self):
        self.load_features(CONFIG)
        for name in BASELINES:
            if any(r['experiment_id']==name for r in self.state['results']):continue
            scores=[];cv=[];fractures=[];started=time.perf_counter()
            for fold,frame in enumerate(self.folds):
                score=ts_auc(frame.target,frame[name],frame.time_online);scores.append(score)
                cv.append({'experiment_id':name,'model':name,'families':'statistical','sampling':'none','fold':fold,'ts_auc':score})
                fractures.extend(self.profiles(name,fold,frame,frame[name]));print(f'{name} fold {fold}: {score:.8f}',flush=True)
            latency=benchmark(lambda:BaselineDetector(name,self.historical).predict_one,self.points,repeats=3)
            result={'experiment_id':name,'model':name,'phase':'statistical_ladder','families':'statistical','sampling':'none',
                'status':'COMPLETE','latency':latency,'latency_us':latency['throughput_median_us_per_point'],'elapsed_seconds':time.perf_counter()-started,**summarize(scores)}
            self.state['results'].append(result);self.state['cv_rows'].extend(cv);self.state['fractures'].extend(fractures)
            ledger(experiment_id=name,model=name,feature_version=feature_version(),ts_auc=result['mean'],worst_fold=result['worst_fold'],latency=result['latency_us'],status='REAL_DEV_BASELINE_COMPLETE',notes=json.dumps(result['fold_scores']))
            self.save()
            if name=='B1_official_ewma':write_json(self.directory/'CHECKPOINT_FIRST_REAL_BENCHMARK.json',{'timestamp':utc_now(),'B0':self.state['results'][0],'B1':result,'split_sha256':self.split.digest,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_scored_uses':0})
        self.feature_audit()

    def feature_audit(self):
        path=self.directory/'FEATURE_AUDIT_REAL.json'
        if path.exists():return
        columns=feature_names(CONFIG);sample=np.concatenate([f.loc[:,columns].to_numpy()[::20] for f in self.folds]);rows=[]
        for j,name in enumerate(columns):
            arrays=[f[name].to_numpy() for f in self.folds];count=sum(len(x) for x in arrays);finite=sum(int(np.isfinite(x).sum()) for x in arrays)
            values=sample[:,j].astype(np.float64);_,counts=np.unique(values,return_counts=True)
            rows.append({'name':name,'family':FEATURE_FAMILIES[j],'rows':count,'finite_rate':finite/count,'sample_variance':float(values.var()),
                'minimum':min(float(x.min()) for x in arrays),'maximum':max(float(x.max()) for x in arrays),
                'sample_quantiles':np.quantile(values,[0,.0001,.001,.01,.5,.99,.999,.9999,1]).tolist(),
                'sample_modal_fraction':float(counts.max()/len(values)),'near_constant_diagnostic':bool(values.var()<1e-12),'saturation_9999pct':bool(counts.max()/len(values)>=.9999)})
        nonconstant=np.std(sample.astype(np.float64),axis=0)>1e-12
        correlation=np.corrcoef(sample[:,nonconstant].astype(np.float64),rowvar=False)
        pd.DataFrame(correlation,index=np.array(columns)[nonconstant],columns=np.array(columns)[nonconstant]).to_csv(self.directory/'FEATURE_CORRELATIONS_REAL.csv')
        pd.DataFrame([{k:v for k,v in row.items() if k!='sample_quantiles'} for row in rows]).to_csv(self.directory/'FEATURE_DISTRIBUTIONS_REAL.csv',index=False)
        write_json(path,{'status':'PASS' if all(r['finite_rate']==1 for r in rows) else 'FAIL','feature_hash':feature_version(),'sample_rows':len(sample),
            'sampling':'every 20th cached row per development fold; no target-based selection','diagnostic_only':True,
            'constant_removal':'fitted independently on each training-fold matrix, never held-out values','features':rows})

    def candidate(self,spec):
        started=time.perf_counter();eid=spec.experiment_id
        config=CONFIG.variant(**{k:tuple(v) if isinstance(v,list) else v for k,v in spec.feature_changes.items()})
        self.state['active_candidate']=asdict(spec);self.save();print(f'START {eid}: {spec.hypothesis}',flush=True)
        try:
            self.load_features(config,spec.representation);scores=[];cv=[];fractures=[];sizes=[];latencies=[];diffs=[]
            oof_dir=ROOT/'data/processed/oof'/eid;oof_dir.mkdir(parents=True,exist_ok=True)
            model_dir=ROOT/'artifacts/models/real'/eid;model_dir.mkdir(parents=True,exist_ok=True)
            for fold,frame in enumerate(self.folds):
                model,prediction,score=fit_fold(self.folds,fold,spec.model,spec.families,spec.sampling,split=self.split,params=spec.params,config=config)
                from src.models.representations import wrap_model,representation_feature_names
                model=wrap_model(model,spec.representation,config)
                names=representation_feature_names(spec.representation,config)
                path=model_dir/f'fold_{fold}.joblib';model.save(path);loaded=type(model).load(path)
                pipeline_feature_hash=loaded.feature_hash
                probe=frame.loc[:,names].to_numpy(dtype=np.float32)[:2000]
                difference=float(np.max(np.abs(model.predict(probe)-loaded.predict(probe))))
                if difference!=0:raise RuntimeError('Real model save/load is not deterministic')
                def factory(m=loaded):
                    state=m.make_state(self.historical) if hasattr(m,'make_state') else StreamingFeatureState(self.historical,config=config)
                    return lambda point:m.predict_one(state.update_and_get(point))
                native_latency=benchmark(factory,self.points[:1000],repeats=3)
                latency=native_latency
                if spec.model in ('M3_histgb','M4_lightgbm','M5_xgboost','M7_cross_order_equal'):
                    from src.streaming.prepared import prepare_model,source_hashes
                    prepared=prepare_model(loaded,fast_initialization=True)
                    native_detector=factory();optimized_detector=prepared.new_state(self.historical)
                    native_wire=np.array([native_detector(p) for p in self.points[:1000]],dtype=np.float32)
                    optimized_wire=np.array([optimized_detector.predict_one(p) for p in self.points[:1000]],dtype=np.float32)
                    np.testing.assert_array_equal(optimized_wire,native_wire)
                    if spec.model=='M5_xgboost':
                        from src.models.compact_xgboost import export_xgboost
                        base_model=loaded.base_model if hasattr(loaded,'base_model') else loaded
                        validation_matrix=frame.loc[:,feature_names(config)].to_numpy(dtype=np.float32)
                        compact_prediction=export_xgboost(base_model.estimator).predict(validation_matrix[:,base_model.columns]).astype(np.float32)
                        np.testing.assert_array_equal(compact_prediction,prediction)
                        del validation_matrix,compact_prediction
                    latency=benchmark(lambda:prepared.new_state(self.historical).predict_one,self.points[:1000],repeats=3)
                    latency.update(native_reference=native_latency,production_mode='fused_with_shared_historical_initialization',
                        native_fused_prefix_float32_diff=0.,prefix_points_verified=1000,
                        full_validation_tree_export_rows_verified=len(frame) if spec.model=='M5_xgboost' else None,
                        **source_hashes(loaded,fast_initialization=True))
                latency['model_bytes']=path.stat().st_size
                latencies.append(latency);sizes.append(path.stat().st_size);diffs.append(difference)
                np.save(oof_dir/f'fold_{fold}.npy',prediction,allow_pickle=False)
                write_json(oof_dir/'MANIFEST.json',{'candidate':asdict(spec),'feature_hash':feature_version(config),'config_hash':config.digest(),
                    'pipeline_feature_hash':pipeline_feature_hash,'representation':spec.representation,
                    'split_sha256':self.split.digest,'cache_directory':str(self.cache.relative_to(ROOT)),'prediction_order':'cache fold order','wire_dtype':'float32','seal_rows':0})
                scores.append(score);cv.append({'experiment_id':eid,'model':spec.model,'families':spec.families,'sampling':spec.sampling,'fold':fold,'ts_auc':score})
                fractures.extend(self.profiles(eid,fold,frame,prediction))
                ledger(experiment_id=eid,model=spec.model,feature_version=pipeline_feature_hash,feature_families=spec.families,params=json.dumps(spec.params),
                    sampling_policy=spec.sampling,fold=fold,ts_auc=score,latency=latency['throughput_median_us_per_point'],memory=psutil.Process().memory_info().rss,status='REAL_DEV_FOLD_COMPLETE',notes=spec.hypothesis)
                print(f'{eid} fold {fold}: {score:.8f}; {latency["throughput_median_us_per_point"]:.3f} us/point',flush=True)
                self.budget.checkpoint(active_candidate=eid,last_completed_fold=fold);del model,loaded,probe;gc.collect()
            latency={'throughput_median_us_per_point':float(np.median([r['throughput_median_us_per_point'] for r in latencies])),
                'fold_benchmarks':latencies,'includes':'feature update plus single-row prediction; no IPC',
                'ten_million_points_seconds':float(np.median([r['throughput_median_us_per_point'] for r in latencies])*10)}
            result={'experiment_id':eid,'model':spec.model,'families':spec.families,'sampling':spec.sampling,'phase':spec.phase,'candidate':asdict(spec),
                'status':'COMPLETE','feature_hash':pipeline_feature_hash,'config_hash':config.digest(),'representation':spec.representation,
                'latency':latency,'latency_us':latency['throughput_median_us_per_point'],
                'model_bytes_max':max(sizes),'save_load_max_abs_diff':max(diffs),'elapsed_seconds':time.perf_counter()-started,**summarize(scores)}
            time_rows=pd.DataFrame([r for r in fractures if r['kind']=='online_time'])
            age_rows=pd.DataFrame([r for r in fractures if r['kind']=='evidence_age_with_negative_controls'])
            result['time_profile']=time_rows.groupby('slice').ts_auc.mean().dropna().to_dict()
            result['age_profile']=age_rows.groupby('slice').ts_auc.mean().dropna().to_dict()
            result['weakest_fold_index']=int(np.argmin(scores))
            champion=next((r for r in self.state['results'] if r['experiment_id']==self.state['primary_id']),None)
            decision=promotion_decision(result,champion);self.state['promotions'].append({'experiment_id':eid,'timestamp':utc_now(),**decision})
            if decision['decision']=='PROMOTE':self.state['primary_id']=eid
            self.state['results'].append(result);self.state['cv_rows'].extend(cv);self.state['fractures'].extend(fractures)
            print(f'COMPLETE {eid}: mean={result["mean"]:.8f}, worst={result["worst_fold"]:.8f}, {decision["decision"]}',flush=True)
        except Exception as exc:
            self.state['failed'].append({'experiment_id':eid,'candidate':asdict(spec),'timestamp':utc_now(),'exception_type':type(exc).__name__})
            (ROOT/'experiments/logs'/f'{eid}_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            ledger(experiment_id=eid,model=spec.model,status='FAILED',notes=f'{type(exc).__name__}; no partial-fold ranking; see local log')
            print(f'FAILED {eid}: {type(exc).__name__}; continuing next hypothesis',flush=True)
        finally:self.state['attempted'].append(spec.identity);self.state['active_candidate']=None;self.save();gc.collect()

    def run(self):
        self.baselines()
        while self.budget.can_start():
            control=ROOT/'artifacts/real/CONTROL.json'
            if control.exists() and json.loads(control.read_text(encoding='utf-8')).get('pause_after_candidate'):
                self.budget.checkpoint(status='ENGINEERING_CHECKPOINT',next_action='Resume with original deadline after the requested code/audit update')
                print('Paused after a complete candidate for an engineering checkpoint.',flush=True);return
            spec=select_next(self.state['results'],set(self.state['attempted']),primary_id=self.state['primary_id'])
            if spec is None:
                self.budget.checkpoint(status='BOUNDED_HYPOTHESES_EXHAUSTED_ROBUSTNESS_REQUIRED',next_action='Complementary mechanisms, selected CV reproduction, latency and robustness audit; no idle repetitions')
                print('Bounded hypotheses exhausted; checkpoint for complementary research and robustness work.',flush=True);return
            observed=[r['elapsed_seconds'] for r in self.state['results'] if r['model']==spec.model]
            estimate=max(observed[-3:],default=180)*1.25
            if not self.budget.can_start(estimate):
                self.budget.checkpoint(status='FINAL_AUDIT_BUFFER',estimated_next_candidate_seconds=estimate)
                print('Protecting final audit buffer; no new experiment admitted.',flush=True);return
            self.candidate(spec)
        self.budget.checkpoint(status='FINAL_AUDIT_BUFFER')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/raw');parser.add_argument('--hours',type=float,default=10)
    args=parser.parse_args();admission(raw_directory=args.data)
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json',hours=args.hours,buffer_minutes=45)
    RealResearch(args.data,budget).run()

if __name__=='__main__':main()

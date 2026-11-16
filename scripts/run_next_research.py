"""Resumable finite-queue DEV screening and full primary-CV research."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.cache import load_design
from src.next.design_selection import select_reference_families, selection_hash
from src.next.extensions import load_extended_design,ExtendedBundle,extension_hash
from src.next.ranking_objectives import fit_ranking,implementation_hash as ranking_hash
from src.next.registered_extensions import RegisteredExtendedBundle,register_extension,registration_hash
from src.next.engine import EngineConfig, feature_hash
from src.next.fitting import fit_binary, fitting_hash
from src.next.predictor import NextBundle, predictor_hash
from src.next.bootstrap import PreparedAUC, paired_summary
from src.next.bootstrap_integrity import load_reference_samples
from src.next.references import load_reference_fold, verify_reference_sources
from src.next.progress import checkpoint
from src.validation.research import fit_fold, summarize
from src.validation.group_bootstrap import draw_group_multiplicities
from src.features.config import CONFIG
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now


def screen_reference(data, policy):
    out = ROOT/'artifacts/next/screen_reference'
    result_path = out/'RESULTS.json'
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding='utf-8'))
        if report['policy_sha256'] != sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'):
            raise RuntimeError('Screen reference policy changed')
        for name, digest in report['files_sha256'].items():
            if sha256(out/name) != digest:
                raise RuntimeError('Screen reference bytes changed')
        return report
    out.mkdir(exist_ok=True)
    ids = set(policy['screen']['ids'])
    frames = []
    for fold in range(5):
        frame, _, _ = load_reference_fold(ROOT, data.guard, 'PRIMARY', fold, features=True)
        frames.append(frame.loc[frame.dataset_id.isin(ids)].reset_index(drop=True))
        del frame
    config = CONFIG.variant(normalization='median_mad', scales=(5,20,160))
    scores, files = [], []
    for fold in policy['screen']['folds']:
        data.guard.partition(set().union(*(set(f.dataset_id) for k,f in enumerate(frames) if k != fold)), set(frames[fold].dataset_id))
        tick = time.perf_counter()
        model, prediction, score = fit_fold(frames, fold, 'M7_cross_order_equal', 'ABCD', 'S3', split=data.guard.split, params={}, config=config)
        model.save(out/f'fold_{fold}.joblib')
        np.save(out/f'fold_{fold}.npy', prediction, allow_pickle=False)
        frames[fold][['dataset_id','time_online','target']].to_parquet(out/f'fold_{fold}_coordinates.parquet', index=False)
        scores.append(score)
        files.extend([f'fold_{fold}.joblib',f'fold_{fold}.npy',f'fold_{fold}_coordinates.parquet'])
        print('SCREEN PRIMARY REFIT', fold, score, round(time.perf_counter()-tick,1), flush=True)
    report = {'status': 'PASS', 'folds': policy['screen']['folds'], 'fold_scores': scores,
        'mean': float(np.mean(scores)), 'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'files_sha256': {name: sha256(out/name) for name in files}, 'training_subset_series': len(ids),
        'seal_rows': 0, 'timestamp': utc_now(), 'authority': 'SCREEN ONLY, not full primary CV'}
    write_json(result_path, report)
    del frames
    gc.collect()
    return report


def fractures(metadata, prediction, data, eid, fold):
    y, t, ids = metadata['target'], metadata['time_online'], metadata['dataset_id']
    tau_map = dict(zip(map(int,data.ids), map(int,data.series['tau'])))
    age = t-pd.Series(ids).map(tau_map).to_numpy()
    rows = []
    for label,a,b in [('0-9',0,10),('10-24',10,25),('25-49',25,50),('50-99',50,100),('100-199',100,200),('200-399',200,400),('400+',400,100000)]:
        mask = (t>=a)&(t<b)
        rows.append({'experiment_id': eid,'fold':fold,'kind':'online_time','bin':label,'ts_auc':ts_auc(y[mask],prediction[mask],t[mask])})
    for label,a,b in [('0-4',0,5),('5-9',5,10),('10-19',10,20),('20-49',20,50),('50+',50,100000)]:
        mask = (y==0)|((y==1)&(age>=a)&(age<b))
        rows.append({'experiment_id': eid,'fold':fold,'kind':'break_age','bin':label,'ts_auc':ts_auc(y[mask],prediction[mask],t[mask])})
    return rows


def run_scope(data, candidate, scope, policy, reference):
    eid = candidate['experiment_id']
    out = ROOT/'artifacts/next/experiments'/eid/scope
    out.mkdir(parents=True, exist_ok=True)
    config = EngineConfig(**candidate['engine_config'])
    if candidate.get('extension'):
        if candidate.get('extension_registration'):
            if candidate['registration_source_sha256']!=registration_hash():
                raise RuntimeError('Preregistered feature-bank registration code changed')
            register_extension(candidate['extension'],candidate['extension_registration'])
        if candidate['extension_frozen_hash']!=extension_hash(candidate['extension']):
            raise RuntimeError('Preregistered extension source changed')
        design,metadata,names,sources,global_rows=load_extended_design(data,candidate,scope)
    else:
        design, metadata, names, sources, global_rows = load_design(data, config=config, scope=scope,
            groups=tuple(candidate['groups']), base=candidate['base'], keep_names=candidate.get('keep_names'))
    if candidate.get('base_family_selection'):
        if not candidate.get('extension'):
            design,names = select_reference_families(design,names,candidate['base_family_selection'])
        sources['base_family_selection_code_sha256'] = selection_hash()
    identity = {'candidate': candidate, 'scope': scope, 'feature_hash': feature_hash(config),
        'fitting_hash': fitting_hash(), 'predictor_hash': predictor_hash(), 'feature_sources': sources,
        'split_sha256': data.guard.split.digest, 'seal_lock_sha256': data.guard.lock_sha,
        'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json')}
    manifest_path = out/'POLICY.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Completed candidate code/config/data mismatch')
    else:
        write_json(manifest_path, identity)
    scores, bootstrap_scores, bootstrap_reference, fracture_rows, cv_rows, engineering = [], [], [], [], [], []
    folds = policy['screen']['folds'] if scope == 'screen' else list(range(5))
    for fold in folds:
        # Preserve the old primary train order (fold then original ID/time).
        train = np.concatenate([np.flatnonzero(metadata['fold']==k) for k in range(5) if k != fold])
        valid = np.flatnonzero(metadata['fold']==fold)
        result_path = out/f'fold_{fold}.json'
        tick = time.perf_counter()
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            if saved['status'] != 'PASS' or any(sha256(out/name) != digest for name,digest in saved['files_sha256'].items()):
                raise RuntimeError('Completed fold bytes changed')
            prediction = np.load(out/f'fold_{fold}.npy', allow_pickle=False)
            bundle_class=RegisteredExtendedBundle if candidate.get('extension_registration') else (ExtendedBundle if candidate.get('extension') else NextBundle)
            model = bundle_class.load(out/f'fold_{fold}.joblib')
            detail = saved['engineering']
            print('NEXT RESUME', eid, scope, fold, flush=True)
        else:
            if candidate.get('ranking_objective'):
                if candidate['ranking_source_sha256']!=ranking_hash():
                    raise RuntimeError('Preregistered ranking objective implementation changed')
                model,prediction,detail=fit_ranking(design,metadata,train,valid,guard=data.guard,
                    config=config,names=names,objective=candidate['ranking_objective'])
            else:
                model, prediction, detail = fit_binary(design, metadata, train, valid, guard=data.guard,
                    config=config, names=names, learner=candidate['learner'], weighting=candidate['weighting'], params=candidate['params'])
            if candidate.get('extension'):
                model=RegisteredExtendedBundle.wrap_registered(model,candidate['extension'],candidate['extension_registration']) if candidate.get('extension_registration') else ExtendedBundle.wrap(model,candidate['extension'])
            model.save(out/f'fold_{fold}.joblib')
            bundle_class=RegisteredExtendedBundle if candidate.get('extension_registration') else (ExtendedBundle if candidate.get('extension') else NextBundle)
            loaded = bundle_class.load(out/f'fold_{fold}.joblib')
            np.testing.assert_array_equal(loaded.predict(design[valid[:2048]]).astype(np.float32), prediction[:2048])
            probe_ids = np.unique(metadata['dataset_id'][valid])[:2]
            for sid in probe_ids:
                h,o,_ = data.get(sid)
                state = loaded.make_state(h)
                live = np.asarray([loaded.predict_one(state.update(point)) for point in o], dtype=np.float32)
                np.testing.assert_array_equal(live, prediction[metadata['dataset_id'][valid] == sid])
            detail['live_cached_probe_max_abs_diff'] = 0.
            detail['live_probe_series'] = len(probe_ids)
            np.save(out/f'fold_{fold}.npy', prediction, allow_pickle=False)
            np.save(out/f'fold_{fold}_global_rows.npy', global_rows[valid], allow_pickle=False)
            files = [f'fold_{fold}.joblib',f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy']
            write_json(result_path, {'status':'PASS', 'engineering':detail,
                'files_sha256': {name:sha256(out/name) for name in files}, 'timestamp':utc_now(),
                'elapsed_seconds':time.perf_counter()-tick, 'seal_rows':0})
        y,t,ids = (metadata[k][valid] for k in ['target','time_online','dataset_id'])
        score = ts_auc(y,prediction,t)
        if scope == 'full' and candidate.get('expected_reference_role'):
            old_coordinates,old_prediction,_ = load_reference_fold(ROOT,data.guard,candidate['expected_reference_role'],fold)
            np.testing.assert_array_equal(old_coordinates.dataset_id.to_numpy(),ids)
            np.testing.assert_array_equal(old_coordinates.time_online.to_numpy(),t)
            np.testing.assert_array_equal(prediction,old_prediction)
            detail['exact_frozen_reference_prediction_parity'] = True
        scores.append(score)
        replicates = policy['bootstrap']['screen_replicates'] if scope == 'screen' else policy['bootstrap']['final_replicates']
        code,counts = draw_group_multiplicities(ids, replicates, policy['bootstrap']['seed']+fold)
        samples = PreparedAUC.create(y,prediction,t,code,counts.shape[1]).evaluate(counts)
        np.testing.assert_allclose(samples[0],score,rtol=0,atol=1e-14)
        bootstrap_scores.append(samples)
        if scope == 'screen':
            coords = pd.read_parquet(ROOT/'artifacts/next/screen_reference'/f'fold_{fold}_coordinates.parquet')
            np.testing.assert_array_equal(coords.dataset_id.to_numpy(),ids)
            np.testing.assert_array_equal(coords.time_online.to_numpy(),t)
            np.testing.assert_array_equal(coords.target.to_numpy(),y)
            base_prediction = np.load(ROOT/'artifacts/next/screen_reference'/f'fold_{fold}.npy',allow_pickle=False)
            bootstrap_reference.append(PreparedAUC.create(y,base_prediction,t,code,counts.shape[1]).evaluate(counts))
        else:
            bootstrap_reference.append(load_reference_samples(ROOT,'PRIMARY',fold)[:len(counts)])
            fracture_rows.extend(fractures({k:v[valid] for k,v in metadata.items()},prediction,data,eid,fold))
        cv_rows.append({'experiment_id':eid,'scope':scope,'fold':fold,'ts_auc':score,'validation_rows':len(valid),
                        'validation_series':len(np.unique(ids)), 'seal_rows':0})
        engineering.append(detail)
        print('NEXT',eid,scope,fold,score,round(time.perf_counter()-tick,1),'seconds',flush=True)
        del model, prediction
        gc.collect()
    boot = paired_summary(np.asarray(bootstrap_reference).T,np.asarray(bootstrap_scores).T)
    np.save(out/'bootstrap_scores.npy',np.asarray(bootstrap_scores).T,allow_pickle=False)
    if scope == 'full':
        summary = summarize(scores)
    else:
        summary = {'fold_scores':scores,'mean':float(np.mean(scores)),'median':float(np.median(scores)), 'worst_fold':float(min(scores))}
    result = {'status':'COMPLETE','scope':scope,**summary,'paired_bootstrap':boot,
              'cv_rows':cv_rows,'fractures':fracture_rows,'engineering':engineering,'feature_names':names,
              'feature_source_identity':sources,'seal_rows':0,'timestamp':utc_now()}
    write_json(out/'RESULTS.json',result)
    del design, metadata
    gc.collect()
    return result


def export_tables():
    records, cv, fracture = [], [], []
    for path in sorted((ROOT/'artifacts/next/experiments').glob('*/RESULTS.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        c = result['candidate']
        full, screen = result.get('full',{}), result.get('screen',{})
        bootstrap = full.get('paired_bootstrap',{})
        row = {k:c.get(k) for k in ['experiment_id','parent_id','generation','hypothesis','change','learner','params','phase','weighting']}
        row.update({'feature_version':result.get('feature_hash'), 'status':result['status'],'reason':result['reason'],
            'primary_cv_mean':full.get('mean'),'primary_cv_median':full.get('median'),'primary_cv_worst':full.get('worst_fold'),
            'delta_vs_primary':result.get('delta_vs_primary'), 'screen_mean':screen.get('mean'),
            'screen_delta':screen.get('paired_bootstrap',{}).get('estimate_mean_delta'),
            'bootstrap_probability':bootstrap.get('probability_delta_positive'),
            'bootstrap_ci_low':bootstrap.get('ci_2_5'),'bootstrap_ci_high':bootstrap.get('ci_97_5'),
            'alternate_split_results':result.get('alternate_split_results'), 'regime_worst':result.get('regime_worst'),
            'mechanism_worst':result.get('mechanism_worst'),'latency_us':result.get('latency_us'),
            'elapsed_seconds':result.get('elapsed_seconds'), 'seal_rows':0, 'reduced_new_usage':0})
        records.append(row)
        for part in [screen,full]:
            cv.extend(part.get('cv_rows',[])); fracture.extend(part.get('fractures',[]))
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT/'MODEL_RESULTS_NEXT.csv',index=False)
    prior = ROOT/'artifacts/next/prior'
    prior.mkdir(exist_ok=True)
    if (ROOT/'EXPERIMENT_LEDGER.csv').exists() and not (prior/'EXPERIMENT_LEDGER_PRE_NEXT.csv').exists():
        shutil.copyfile(ROOT/'EXPERIMENT_LEDGER.csv',prior/'EXPERIMENT_LEDGER_PRE_NEXT.csv')
    frame.to_csv(ROOT/'EXPERIMENT_LEDGER.csv',index=False)
    pd.DataFrame(cv).to_csv(ROOT/'CV_RESULTS_NEXT.csv',index=False)
    ff = pd.DataFrame(fracture, columns=['experiment_id','fold','kind','bin','ts_auc'])
    ff.loc[ff.kind=='online_time'].to_csv(ROOT/'TIME_FRACTURE_NEXT.csv',index=False)
    ff.loc[ff.kind=='break_age'].to_csv(ROOT/'BREAK_AGE_NEXT.csv',index=False)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate')
    parser.add_argument('--screen-reference-only',action='store_true')
    parser.add_argument('--screen-only',action='store_true')
    parser.add_argument('--queue',default='configs/next_initial_candidates.json')
    args = parser.parse_args()
    set_num_threads(3)
    data = prepare_dev(ROOT)
    verify_reference_sources(ROOT)
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    queue = json.loads((ROOT/args.queue).read_text(encoding='utf-8'))
    if queue['fitting_code_sha256'] != fitting_hash() or queue['predictor_code_sha256'] != predictor_hash():
        raise RuntimeError('Preregistered fitting/predictor implementation changed')
    reference = screen_reference(data,policy)
    if args.screen_reference_only:
        return
    restored = json.loads((ROOT/'artifacts/next/REFERENCE_RESTORATION.json').read_text(encoding='utf-8'))['references']['PRIMARY']
    for candidate in queue['candidates']:
        eid = candidate['experiment_id']
        if args.candidate and args.candidate != eid:
            continue
        out = ROOT/'artifacts/next/experiments'/eid
        out.mkdir(parents=True,exist_ok=True)
        result_path = out/'RESULTS.json'
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding='utf-8'))
            if prior['candidate'] != candidate:
                raise RuntimeError('Existing experiment identity changed')
            if prior['status'] not in ('BUG','SCREEN_PASS_FULL_PENDING'):
                print('NEXT ALREADY COMPLETE',eid,prior['status'],flush=True)
                continue
        started = time.perf_counter()
        result = {'candidate':candidate,'feature_hash':feature_hash(EngineConfig(**candidate['engine_config'])),
                  'status':'BUG','reason':'experiment did not finish','timestamp':utc_now()}
        if candidate.get('extension'):
            if candidate.get('extension_registration'):
                register_extension(candidate['extension'],candidate['extension_registration'])
            result['feature_hash']=hashlib.sha256((result['feature_hash']+extension_hash(candidate['extension'])).encode()).hexdigest()
        try:
            screen = run_scope(data,candidate,'screen',policy,reference)
            result['screen'] = screen
            delta = screen['paired_bootstrap']['estimate_mean_delta']
            if delta < policy['screen']['minimum_mean_delta_for_full']:
                result.update(status='SCREEN_FAIL',reason='Fixed screen delta below -0.002; no full-CV claim')
            elif args.screen_only:
                result.update(status='SCREEN_PASS_FULL_PENDING',reason='Screen passed; full primary CV pending')
            else:
                full = run_scope(data,candidate,'full',policy,reference)
                result['full'] = full
                delta = full['mean']-restored['mean']
                result['delta_vs_primary'] = delta
                passed = (delta >= policy['promotion']['mean_delta_min'] and
                    full['median'] >= restored['median'] and
                    full['worst_fold']-restored['worst'] >= policy['promotion']['worst_delta_min'] and
                    full['paired_bootstrap']['probability_delta_positive'] >= policy['promotion']['bootstrap_probability_min'])
                result.update(status='ROBUSTNESS_PENDING' if passed else 'FULL_FAIL',
                    reason='Primary gates passed; full robustness/latency gates pending' if passed else 'One or more predeclared full primary gates failed')
        except Exception as exc:
            result.update(status='BUG',reason=type(exc).__name__+': '+str(exc))
            (out/'FAILURE.log').write_text(traceback.format_exc(),encoding='utf-8')
            print('NEXT BUG',eid,type(exc).__name__,str(exc),flush=True)
        result['elapsed_seconds'] = time.perf_counter()-started
        write_json(result_path,result)
        frame = export_tables()
        eligible=frame.loc[frame.status!='BUG']
        best = max(float(eligible.primary_cv_mean.max()),restored['mean']) if 'primary_cv_mean' in eligible and eligible.primary_cv_mean.notna().any() else restored['mean']
        checkpoint(ROOT,eid,frame.experiment_id.tolist(),['remaining finite queue','matched feature diagnostics','ranking objective','finalist robustness'],
                   failures=[result['reason']] if result['status']=='BUG' else [],
                   artifacts=['MODEL_RESULTS_NEXT.csv','CV_RESULTS_NEXT.csv'],best=float(best))
        print('NEXT RESULT',eid,result['status'],result.get('full',{}).get('mean'),result['reason'],flush=True)


if __name__ == '__main__':
    main()

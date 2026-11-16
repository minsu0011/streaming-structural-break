"""Package the fixed blend, then promote only after complete production proofs."""
import argparse,json,sys,gc
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_cross_order_blend import COMPONENTS,EID
from scripts.run_real_research import RealResearch
from src.validation.adaptive import Candidate,ResearchBudget,promotion_decision
from src.models.representations import bundle_class
from src.models.cross_order import CrossOrderBlendBundle,MODEL_NAME
from src.features.cross_order import REPRESENTATION,feature_names,implementation_hash
from src.features.cross_order_cache import build_cross_order_cache
from src.utils.artifacts import write_json,sha256,ledger,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--promote',action='store_true');args=parser.parse_args()
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());control=json.loads((ROOT/'artifacts/real/CONTROL.json').read_text())
    if state.get('active_candidate') or not control['pause_after_candidate']:raise RuntimeError('Clean main-loop pause required')
    research=RealResearch(ROOT/'data/raw',ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json'))
    stored=next((r for r in research.state['results'] if r['experiment_id']==EID),None)
    if args.promote:
        if stored is None:raise RuntimeError('Package the completed candidate before production verification')
        from src.streaming.prepared import source_hashes
        model=CrossOrderBlendBundle.load(ROOT/'artifacts/models/real'/EID/'fold_0.joblib')
        fused=json.loads((ROOT/'artifacts/real/fused'/f'{EID}_fast_init.json').read_text())
        contract=json.loads((ROOT/'artifacts/real/contracts'/f'{EID}_fused_fast_init.json').read_text())
        cv=json.loads((ROOT/'artifacts/real/cv_reproduction/cross_order_integration'/EID/'RESULTS.json').read_text())
        if fused['status']!='PASS' or fused['total_rows']!=4014405 or any(fused.get(k)!=v for k,v in source_hashes(model,fast_initialization=True).items()):raise RuntimeError('Current all-DEV fused OOF proof required')
        if contract['status']!='PASS' or contract['inference_source_sha256']!=sha256(ROOT/'src/streaming/inference.py'):raise RuntimeError('Current live contract proof required')
        if cv['status']!='PASS' or cv['total_rows']!=4014405:raise RuntimeError('Independent complete CV refit proof required')
        primary=next(r for r in research.state['results'] if r['experiment_id']==research.state['primary_id'])
        decision=promotion_decision(stored,primary)
        if decision['decision']!='PROMOTE':raise RuntimeError('Candidate no longer passes the frozen promotion gates')
        stored['deployment_status']='PRODUCTION_VERIFIED';research.state['primary_id']=EID
        research.state['promotions'].append({'experiment_id':EID,'timestamp':utc_now(),**decision,'production_proofs':['full OOF fused replay','independent five-fold refit','guarded live API']})
        ledger(experiment_id=EID,model=MODEL_NAME,ts_auc=stored['mean'],worst_fold=stored['worst_fold'],latency=stored['latency_us'],status='REAL_DEV_FIXED_BLEND_PROMOTED',notes=json.dumps(decision))
        research.save();print('FIXED BLEND PROMOTED',EID,stored['mean'],stored['median'],stored['worst_fold'],flush=True);return
    if stored is not None:raise RuntimeError('Fixed blend integration is already recorded')
    report=json.loads((ROOT/'artifacts/real/cross_order_blend/RESULTS.json').read_text());result=report['results'][0]
    if report['status']!='COMPLETE' or result['versus_frozen_primary']['decision']!='PROMOTE':raise RuntimeError('Complete qualifying five-fold result required')
    anchors=[next(r for r in research.state['results'] if r['experiment_id']==eid) for eid in COMPONENTS]
    first=bundle_class(anchors[0]['representation']).load(ROOT/'artifacts/models/real'/COMPONENTS[0]/'fold_0.joblib');config=first.feature_config
    cache=build_cross_order_cache(ROOT/'data/raw',ROOT/'data/cache',research.split,config)
    for fold in range(5):
        children=tuple(bundle_class(r['representation']).load(ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib') for eid,r in zip(COMPONENTS,anchors))
        model=CrossOrderBlendBundle(children,config);path=ROOT/'artifacts/models/real'/EID/f'fold_{fold}.joblib';model.save(path);loaded=CrossOrderBlendBundle.load(path)
        matrix=pd.read_parquet(cache/f'fold_{fold}.parquet',columns=list(feature_names(config))).to_numpy(dtype=np.float32)
        original=np.load(ROOT/'data/processed/oof'/EID/f'fold_{fold}.npy',allow_pickle=False)
        np.testing.assert_array_equal(model.predict(matrix),original);np.testing.assert_array_equal(loaded.predict(matrix),original)
        print('FIXED BLEND PACKAGED',fold,len(original),'identical rows',flush=True);del children,model,loaded,matrix,original;gc.collect()
    candidate=Candidate(model=MODEL_NAME,families='ABCD',sampling='S3',params={},feature_changes=anchors[0]['candidate']['feature_changes'],
        representation=REPRESENTATION,phase='cross_order_fixed_average',hypothesis=report['policy']['hypothesis'])
    oof=ROOT/'data/processed/oof'/EID;previous=json.loads((oof/'MANIFEST.json').read_text());write_json(oof/'ORIGINAL_COMPONENT_COORDINATES.json',previous)
    write_json(oof/'MANIFEST.json',{**previous,'candidate':asdict(candidate),'cache_directory':str(cache.relative_to(ROOT)),'representation':REPRESENTATION,
        'feature_hash':implementation_hash(config),'component_coordinate_manifest_sha256':sha256(oof/'ORIGINAL_COMPONENT_COORDINATES.json')})
    imported={k:v for k,v in result.items() if k not in ('cv_rows','fractures','latencies')}
    imported.update(model=MODEL_NAME,candidate=asdict(candidate),feature_hash=implementation_hash(config),config_hash=config.digest(),representation=REPRESENTATION,
        save_load_max_abs_diff=0.,deployment_status='PENDING_PRODUCTION_VERIFICATION',model_bytes_max=max((ROOT/'artifacts/models/real'/EID/f'fold_{k}.joblib').stat().st_size for k in range(5)))
    imported['latency']={'throughput_median_us_per_point':result['latency_us'],'fold_benchmarks':result['latencies'],'ten_million_points_seconds':result['latency_us']*10,'includes':'two causal feature states and a fixed float32 child-probability average; no IPC'}
    fractures=pd.DataFrame(result['fractures']);imported['time_profile']=fractures.loc[fractures.kind=='online_time'].groupby('slice').ts_auc.mean().to_dict()
    imported['age_profile']=fractures.loc[fractures.kind=='evidence_age_with_negative_controls'].groupby('slice').ts_auc.mean().to_dict()
    research.state['results'].append(imported);research.state['attempted'].append(candidate.identity)
    research.state['cv_rows'].extend([{**r,'model':MODEL_NAME} for r in result['cv_rows']]);research.state['fractures'].extend(result['fractures'])
    ledger(experiment_id=EID,model=MODEL_NAME,ts_auc=imported['mean'],worst_fold=imported['worst_fold'],latency=imported['latency_us'],status='REAL_DEV_FIXED_BLEND_INTEGRATION_PENDING',notes='Frozen scores unchanged. Full production and refit verification required before promotion.')
    research.save();print('FIXED BLEND PACKAGED; PRIMARY UNCHANGED PENDING PROOFS',flush=True)


if __name__=='__main__':main()

"""Apply the precommitted DEV selection policy; freeze only in the final buffer."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.representations import bundle_class, ar_prefix
from src.streaming.fused import implementation_hash as fused_hash
from src.streaming.fused_ar import implementation_hash as ar_fused_hash
from src.streaming.fast_initialization import implementation_hash as initialization_hash
from src.validation.adaptive import ResearchBudget
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json, sha256, utc_now


def production_mode(result):
    return 'fused' if result['model'] in ('M3_histgb','M4_lightgbm','M5_xgboost','M7_cross_order_equal') else 'native'


def eligible(result):
    if result.get('status')!='COMPLETE' or not result.get('candidate'):return False
    if result.get('deployment_status')=='PENDING_PRODUCTION_VERIFICATION':return False
    if len(result.get('fold_scores',[]))!=5 or not np.isfinite(result['fold_scores']).all():return False
    if result.get('save_load_max_abs_diff')!=0:return False
    try:bundle_class(result['candidate'].get('representation','base'))
    except ValueError:return False
    return all((ROOT/'artifacts/models/real'/result['experiment_id']/f'fold_{k}.joblib').exists() for k in range(5))


def valid_fused_receipt(report,result):
    representation=result['candidate'].get('representation','base')
    if representation.startswith('ar8_arch1_'):
        from src.streaming.fused_arch import implementation_hash as arch_hash
        if report.get('arch_fused_source_hash')!=arch_hash():return False
    if representation=='cross_order_ar4_ar8_equal':
        from src.streaming.fused_cross_order import implementation_hash as cross_hash
        if report.get('cross_order_fused_source_hash')!=cross_hash() or report.get('ar_fused_source_hash')!=ar_fused_hash():return False
    return (report.get('status')=='PASS' and report.get('fused_source_hash')==fused_hash()
        and report.get('initialization_source_hash')==initialization_hash()
        and (not ar_prefix(representation) or report.get('ar_fused_source_hash')==ar_fused_hash()))


def best_existing_latency(result):
    values=[float(result['latency_us'])]
    for path in (ROOT/'artifacts/real/fused').glob(result['experiment_id']+'*.json'):
        report=json.loads(path.read_text())
        if valid_fused_receipt(report,result):values.append(report['fused_latency']['throughput_median_us_per_point'])
    for path in (ROOT/'artifacts/real/latency').glob('*/'+result['experiment_id']+'_*.json'):
        report=json.loads(path.read_text())
        if report.get('status')!='PASS' or not report.get('isolated_from_research_workers'):continue
        if report.get('model_sha256')!=sha256(ROOT/'artifacts/models/real'/result['experiment_id']/'fold_0.joblib'):continue
        if report.get('mode')=='fused' and not valid_fused_receipt(report,result):continue
        values.append(report['long_stream']['measured_us_per_point'])
    return min(values)


def selection_plan(state):
    results=[r for r in state['results'] if eligible(r)]
    primary=next(r for r in results if r['experiment_id']==state['primary_id'])
    others=[r for r in results if r['experiment_id']!=primary['experiment_id']]
    robust_pool=[r for r in others if r['mean']>=primary['mean']-.005 and r['median']>=primary['median']-.005]
    robust=max(robust_pool,key=lambda r:(r['worst_fold'],-r['std'])) if robust_pool else None
    excluded={primary['experiment_id']}|({robust['experiment_id']} if robust else set())
    fast_pool=[r for r in others if r['experiment_id'] not in excluded and r['mean']>=primary['mean']-.010 and r['worst_fold']>=primary['worst_fold']-.010]
    shortlist=sorted(fast_pool,key=lambda r:(best_existing_latency(r),r['experiment_id']))[:5]
    return primary,robust,shortlist


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--benchmark-tag',default='final_selection')
    parser.add_argument('--audit-phase-closure',action='store_true',help='Allow early DEV freeze only after an explicit irreversible model-research closure; reduced diagnostics keep the original deadline')
    args=parser.parse_args()
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());primary,robust,shortlist=selection_plan(state)
    policy_path=ROOT/'artifacts/real/FINAL_CANDIDATE_SELECTION_POLICY.json'
    plan={'status':'PLANNED_NOT_FROZEN','timestamp':utc_now(),'primary_id':primary['experiment_id'],
        'robust_id':robust['experiment_id'] if robust else None,'fast_shortlist':[r['experiment_id'] for r in shortlist],
        'benchmark_candidates':list(dict.fromkeys([primary['experiment_id'],*([robust['experiment_id']] if robust else []),*[r['experiment_id'] for r in shortlist]])),
        'benchmark_modes':{r['experiment_id']:production_mode(r) for r in [primary,*([robust] if robust else []),*shortlist]},
        'policy_sha256':sha256(policy_path),'selection_state_sha256':sha256(ROOT/'artifacts/real/RESEARCH_STATE.json'),
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_scores_consulted':False}
    if not args.freeze:
        write_json(ROOT/'artifacts/real/FINAL_SELECTION_PLAN.json',plan);print(json.dumps(plan,indent=2));return
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if budget.remaining_experiment_seconds()>0:
        closure_path=ROOT/'artifacts/real/MODEL_RESEARCH_CLOSURE.json'
        if not args.audit_phase_closure or not budget.state.get('model_research_closed') or not closure_path.exists():raise RuntimeError('Early freeze requires a recorded transition from model research to audit-only work')
        closure=json.loads(closure_path.read_text())
        if closure['status']!='CLOSED_FOR_AUDIT_ONLY' or closure['research_state_sha256']!=sha256(ROOT/'artifacts/real/RESEARCH_STATE.json') or closure['primary_id']!=primary['experiment_id']:raise RuntimeError('Research closure does not match the final selection state')
    control=json.loads((ROOT/'artifacts/real/CONTROL.json').read_text())
    if state.get('active_candidate') or not control.get('pause_after_candidate'):raise RuntimeError('A clean stopped research loop is required')
    frozen_path=ROOT/'DEV_CHAMPION_CANDIDATES.json'
    if frozen_path.exists():raise RuntimeError('Final selection is immutable once written')
    benchmark_reports={}
    lookup={r['experiment_id']:r for r in state['results']}
    for eid in plan['benchmark_candidates']:
        mode=production_mode(lookup[eid]);suffix=mode+('_fast_init' if mode=='fused' else '')
        path=ROOT/'artifacts/real/latency'/args.benchmark_tag/f'{eid}_{suffix}.json';report=json.loads(path.read_text())
        if report['status']!='PASS' or not report['isolated_from_research_workers']:raise RuntimeError('Isolated benchmark PASS required before selection')
        if report['model_sha256']!=sha256(ROOT/'artifacts/models/real'/eid/'fold_0.joblib'):raise RuntimeError('Benchmark model identity changed')
        if mode=='fused' and not valid_fused_receipt(report,lookup[eid]):raise RuntimeError('Production benchmark source validation is stale')
        benchmark_reports[eid]={'path':str(path.relative_to(ROOT)),'sha256':sha256(path),
            'us_per_point':report['long_stream']['measured_us_per_point'],'amortized_10m_seconds':report['real_history_initialization']['ten_million_amortized_seconds_at_observed_mean_length']}
    fast=min(shortlist,key=lambda r:benchmark_reports[r['experiment_id']]['us_per_point']) if shortlist else None
    selected={'PRIMARY_DEV_CANDIDATE':primary,'BACKUP_ROBUST_CANDIDATE':robust,'BACKUP_FAST_CANDIDATE':fast}
    split=FrozenSplit(ROOT/'artifacts/validation')
    report={**plan,'status':'FROZEN_FOR_FINAL_DIAGNOSTICS','freeze_timestamp':utc_now(),'split_sha256':split.digest,'candidates':{},
        'benchmark_evidence':benchmark_reports,'selection_uncertainty':'Adaptive DEV selection; scores and conditional OOF bootstrap are not independent or selection-adjusted estimates.'}
    for role,result in selected.items():
        report[role]=result['experiment_id'] if result else None
        if result is None:continue
        eid=result['experiment_id'];spec={'status':'FROZEN_PRIMARY' if role=='PRIMARY_DEV_CANDIDATE' else 'FROZEN_BACKUP','role':role,
            'experiment_id':eid,'candidate':result['candidate'],'development_ids':sorted(split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].astype(int).tolist()),
            'known_training_ids':sorted(split.mapping),'split_sha256':split.digest,'use_fused_inference':production_mode(result)=='fused',
            'use_fast_historical_init':production_mode(result)=='fused'}
        spec_path=ROOT/'configs'/('deployment.json' if role=='PRIMARY_DEV_CANDIDATE' else f'deployment_{role.lower()}.json')
        write_json(spec_path,spec)
        report['candidates'][role]={k:result[k] for k in ('experiment_id','model','candidate','feature_hash','fold_scores','mean','median','std','worst_fold','time_profile','age_profile')}
        report['candidates'][role].update(specification_path=str(spec_path.relative_to(ROOT)),specification_sha256=sha256(spec_path),
            model_directory='artifacts/deployment/final/'+role,benchmark=benchmark_reports[eid])
    write_json(frozen_path,report);print('FINAL CANDIDATES FROZEN',json.dumps({role:report[role] for role in selected}),flush=True)


if __name__=='__main__':main()

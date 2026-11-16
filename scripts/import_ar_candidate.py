"""Promote the completed AR input experiment only after native deployment guards."""
import json, sys
from dataclasses import asdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.adaptive import Candidate,ResearchBudget,promotion_decision
from src.utils.artifacts import ledger,sha256,utc_now
from scripts.run_real_research import RealResearch

EID='M4_ar_residual_input_ABCD_S3_9f68be316ea3'


def main():
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    control=json.loads((ROOT/'artifacts/real/CONTROL.json').read_text())
    if state.get('active_candidate') or not control.get('pause_after_candidate'):raise RuntimeError('Clean main-loop checkpoint required')
    if any(r['experiment_id']==EID for r in state['results']):raise RuntimeError('AR candidate already imported')
    report=json.loads((ROOT/'artifacts/real'/f'{EID}.json').read_text())
    contract=json.loads((ROOT/'artifacts/real/contracts'/f'{EID}_native.json').read_text())
    if report['status']!='COMPLETE' or len(report['fold_scores'])!=5 or contract['status']!='PASS':raise RuntimeError('Full CV and native contract PASS required')
    if contract['inference_source_sha256']!=sha256(ROOT/'src/streaming/inference.py'):raise RuntimeError('Native inference validation is stale')
    anchor=report['policy']['candidate']
    spec=Candidate(anchor['model'],families=anchor['families'],sampling=anchor['sampling'],params=anchor['params'],feature_changes=anchor['feature_changes'],
        representation=report['representation'],phase='ar_input_refinement',hypothesis='Historical-only AR(4) innovations preserve magnitude while removing predictable dependence before the unchanged calibrated causal feature bank.')
    result={k:v for k,v in report.items() if k not in ('cv_rows','fractures','latencies')}
    result.update(model=spec.model,candidate=asdict(spec),feature_hash=report['representation_hash'],
        latency={'throughput_median_us_per_point':report['latency_us'],'fold_benchmarks':report['latencies'],
                 'ten_million_points_seconds':report['latency_us']*10,'includes':'native AR update, calibrated features and one-row tree probability; no IPC'},
        model_bytes_max=max(r['model_bytes'] for r in report['latencies']),external_experiment_id_retained=True,
        canonical_candidate_identity=spec.identity,deployment_kind='ar_residual_calibrated',deployment_native_contract='PASS')
    frame=pd.DataFrame(report['fractures'])
    result['time_profile']=frame.loc[frame.kind=='online_time'].groupby('slice').ts_auc.mean().dropna().to_dict()
    result['age_profile']=frame.loc[frame.kind=='evidence_age_with_negative_controls'].groupby('slice').ts_auc.mean().dropna().to_dict()
    result['weakest_fold_index']=min(range(5),key=lambda k:result['fold_scores'][k])
    research=RealResearch(ROOT/'data/raw',ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json'))
    champion=next(r for r in research.state['results'] if r['experiment_id']==research.state['primary_id'])
    decision=promotion_decision(result,champion)
    research.state['results'].append(result);research.state['attempted'].append(spec.identity)
    research.state['cv_rows'].extend([{**r,'model':spec.model} for r in report['cv_rows']]);research.state['fractures'].extend(report['fractures'])
    research.state['promotions'].append({'experiment_id':EID,'timestamp':utc_now(),**decision})
    if decision['decision']=='PROMOTE':research.state['primary_id']=EID
    ledger(experiment_id=EID,model=spec.model,feature_version=result['feature_hash'],feature_families=spec.families,sampling_policy=spec.sampling,
        ts_auc=result['mean'],worst_fold=result['worst_fold'],latency=result['latency_us'],status='REAL_DEV_AR_INPUT_VALIDATED',notes=json.dumps(decision))
    research.save();print('AR IMPORT',decision,'mean',result['mean'],'median',result['median'],'worst',result['worst_fold'],flush=True)


if __name__=='__main__':main()

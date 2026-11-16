"""Register fixed ARCH trials; require production proofs before backup eligibility."""
import argparse,json,sys
from dataclasses import asdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.adaptive import Candidate,ResearchBudget,promotion_decision
from src.models.representations import bundle_class
from src.streaming.prepared import source_hashes
from src.utils.artifacts import write_json,sha256,ledger,utc_now
from scripts.run_real_research import RealResearch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    research=RealResearch(ROOT/'data/raw',ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json'))
    if research.state.get('active_candidate') or not json.loads((ROOT/'artifacts/real/CONTROL.json').read_text())['pause_after_candidate']:raise RuntimeError('Clean main-loop pause required')
    for path in sorted((ROOT/'artifacts/real').glob('M*_AR8_ARCH1_ABCD_S3_*.json')):
        if path.stem.endswith('_CHECKPOINT'):continue
        report=json.loads(path.read_text());eid=report['experiment_id']
        result=next((r for r in research.state['results'] if r['experiment_id']==eid),None)
        if args.verify:
            if result is None:raise RuntimeError('Register complete CV evidence before validating deployment')
            if result.get('deployment_status')=='PRODUCTION_VERIFIED':continue
            model=bundle_class(result['representation']).load(ROOT/'artifacts/models/real'/eid/'fold_0.joblib')
            fused=json.loads((ROOT/'artifacts/real/fused'/f'{eid}_fast_init.json').read_text())
            contract=json.loads((ROOT/'artifacts/real/contracts'/f'{eid}_fused_fast_init.json').read_text())
            refit=json.loads((ROOT/'artifacts/real/cv_reproduction/arch_integration'/eid/'RESULTS.json').read_text())
            if fused['status']!='PASS' or fused['total_rows']!=4014405 or any(fused.get(k)!=v for k,v in source_hashes(model,fast_initialization=True).items()):raise RuntimeError('Current full fused OOF proof required')
            if contract['status']!='PASS' or contract['inference_source_sha256']!=sha256(ROOT/'src/streaming/inference.py'):raise RuntimeError('Current live API proof required')
            if refit['status']!='PASS' or refit['total_rows']!=4014405:raise RuntimeError('Independent complete CV refit proof required')
            result['deployment_status']='PRODUCTION_VERIFIED'
            primary=next(r for r in research.state['results'] if r['experiment_id']==research.state['primary_id']);decision=promotion_decision(result,primary)
            research.state['promotions'].append({'experiment_id':eid,'timestamp':utc_now(),**decision})
            if decision['decision']=='PROMOTE':research.state['primary_id']=eid
            ledger(experiment_id=eid,model=result['model'],ts_auc=result['mean'],worst_fold=result['worst_fold'],latency=result['latency_us'],status='REAL_DEV_ARCH_PRODUCTION_VERIFIED',notes=json.dumps(decision))
            research.save();print('ARCH ELIGIBLE',eid,decision['decision'],flush=True);continue
        if result is not None:continue
        if report['status']!='COMPLETE' or len(report['fold_scores'])!=5:raise RuntimeError('Complete five-fold result required')
        spec=Candidate(model=report['learner_name'],families='ABCD',sampling='S3',params={},feature_changes=report['anchor_candidate']['feature_changes'],
            representation=report['representation'],phase=report['phase'],hypothesis=report['hypothesis'])
        result={k:v for k,v in report.items() if k not in ('cv_rows','fractures','latencies')}
        result.update(model=spec.model,candidate=asdict(spec),canonical_candidate_identity=spec.identity,external_experiment_id_retained=True,deployment_status='PENDING_PRODUCTION_VERIFICATION',
            latency={'throughput_median_us_per_point':report['latency_us'],'fold_benchmarks':report['latencies'],'ten_million_points_seconds':report['latency_us']*10,'includes':'native AR8, historical variance forecast, features and scalar learner; no IPC'},
            model_bytes_max=max(r['model_bytes'] for r in report['latencies']))
        fractures=pd.DataFrame(report['fractures']);result['time_profile']=fractures.loc[fractures.kind=='online_time'].groupby('slice').ts_auc.mean().to_dict()
        result['age_profile']=fractures.loc[fractures.kind=='evidence_age_with_negative_controls'].groupby('slice').ts_auc.mean().to_dict()
        research.state['results'].append(result);research.state['attempted'].append(spec.identity)
        research.state['cv_rows'].extend([{**r,'model':spec.model} for r in report['cv_rows']]);research.state['fractures'].extend(report['fractures'])
        ledger(experiment_id=eid,model=spec.model,ts_auc=result['mean'],worst_fold=result['worst_fold'],latency=result['latency_us'],status='REAL_DEV_ARCH_INTEGRATION_PENDING',notes='Independent CV refit, all-DEV streaming replay and live API proof required before final-role eligibility.')
        research.save();print('ARCH REGISTERED',eid,'mean',result['mean'],'worst',result['worst_fold'],flush=True)


if __name__=='__main__':main()

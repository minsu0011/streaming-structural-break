"""Import complete external pairwise evaluations only at a clean main-loop pause."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.validation.adaptive import ResearchBudget, promotion_decision
from src.utils.artifacts import ledger, utc_now
from scripts.run_real_research import RealResearch


def main():
    control = json.loads((ROOT / 'artifacts/real/CONTROL.json').read_text())
    state = json.loads((ROOT / 'artifacts/real/RESEARCH_STATE.json').read_text())
    if not control.get('pause_after_candidate') or state.get('active_candidate'):
        raise RuntimeError('Main research must pause after a complete candidate before importing results')
    research = RealResearch(ROOT / 'data/raw', ResearchBudget(ROOT / 'artifacts/validation/REAL_SESSION.json'))
    primary = next(r for r in research.state['results'] if r['experiment_id'] == research.state['primary_id'])
    for path in sorted((ROOT / 'artifacts/real').glob('M6_pairwise_*_ABCD_*.json')):
        if path.stem.endswith('_CHECKPOINT'): continue
        report = json.loads(path.read_text()); eid = report['experiment_id']
        if any(r['experiment_id'] == eid for r in research.state['results']): continue
        if report['status'] != 'COMPLETE':
            ledger(experiment_id=eid, model='M6_pairwise', status='EXTERNAL_PAIRWISE_FAILED', notes=f'Incomplete five-fold result: {path.name}')
            continue
        result = {k: v for k, v in report.items() if k not in ('cv_rows', 'fractures', 'latencies', 'fold_metadata')}
        result['phase'] = 'pairwise_objective'
        result['latency'] = {'throughput_median_us_per_point': report['latency_us'], 'fold_benchmarks': report['latencies'],
                             'ten_million_points_seconds': report['latency_us'] * 10, 'includes': 'causal feature update plus single-row probability; no IPC'}
        result['model_bytes_max'] = max(r['model_bytes'] for r in report['latencies'])
        result['hypothesis'] = 'Replace independent row loss with a fixed sampled within-time positive-negative pairwise loss, at the explicitly frozen representation anchor.'
        result['deployment_status'] = 'RESEARCH_ONLY_NOT_ADOPTED'
        fractures = pd.DataFrame(report['fractures'])
        result['time_profile'] = fractures.loc[fractures.kind == 'online_time'].groupby('slice').ts_auc.mean().to_dict()
        result['age_profile'] = fractures.loc[fractures.kind == 'evidence_age_with_negative_controls'].groupby('slice').ts_auc.mean().to_dict()
        decision = promotion_decision(result, primary)
        if decision['decision'] == 'PROMOTE':
            raise RuntimeError('Pairwise candidate meets score gates; integrate frozen training dispatch before promotion')
        research.state['results'].append(result); research.state['cv_rows'].extend(report['cv_rows']); research.state['fractures'].extend(report['fractures'])
        research.state['promotions'].append({'experiment_id': eid, 'timestamp': utc_now(), **decision})
        ledger(experiment_id=eid, model='M6_pairwise', sampling_policy=report['sampling'], ts_auc=result['mean'], worst_fold=result['worst_fold'],
               latency=result['latency_us'], status='REAL_DEV_PAIRWISE_COMPLETE', notes=json.dumps({'pair_budget': report['pair_budget'], 'decision': decision, 'report': str(path.relative_to(ROOT))}))
        print('IMPORTED', eid, result['mean'], decision['decision'], flush=True)
    research.save()


if __name__ == '__main__': main()

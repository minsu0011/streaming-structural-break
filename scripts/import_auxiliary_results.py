"""Register complete, bounded auxiliary hypotheses at a clean main-loop pause."""
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
        raise RuntimeError('Pause the main worker after a complete candidate before importing')
    research = RealResearch(ROOT / 'data/raw', ResearchBudget(ROOT / 'artifacts/validation/REAL_SESSION.json'))
    primary = next(r for r in research.state['results'] if r['experiment_id'] == research.state['primary_id'])
    reports = []
    blend_path = ROOT / 'artifacts/real/fixed_model_blends/RESULTS.json'
    if blend_path.exists():
        blend = json.loads(blend_path.read_text())
        if blend['status'] == 'COMPLETE':
            for item in blend['results']:
                item['elapsed_seconds'] = None
                item['shared_run_elapsed_seconds'] = blend['elapsed_seconds']
                item['timing_note'] = 'The two fixed averages share component loading and validation work; individual elapsed runtime was not instrumented.'
                reports.append((item, blend_path))
    for path in (ROOT / 'artifacts/real').glob('M4_rank_gaussian_ABCD_S3_*.json'):
        if not path.stem.endswith('_CHECKPOINT'): reports.append((json.loads(path.read_text()), path))
    for path in (ROOT/'artifacts/real').glob('M4_context_*_S3_*.json'):
        if not path.stem.endswith('_CHECKPOINT'):
            report=json.loads(path.read_text());report['deployment_status']='RESEARCH_ONLY_NOT_ADOPTED';reports.append((report,path))
    for path in (ROOT/'artifacts/real').glob('M4_hist_quantile_*_ABCD_S3_*.json'):
        if not path.stem.endswith('_CHECKPOINT'):
            report=json.loads(path.read_text());report['deployment_status']='RESEARCH_ONLY_NOT_ADOPTED';reports.append((report,path))
    for path in (ROOT/'artifacts/real').glob('M4_AR8_magnitude_*_S3_*.json'):
        if not path.stem.endswith('_CHECKPOINT'):
            report=json.loads(path.read_text());report['deployment_status']='RESEARCH_ONLY_NOT_ADOPTED';reports.append((report,path))
    for report, path in reports:
        eid = report['experiment_id']
        if any(r['experiment_id'] == eid for r in research.state['results']): continue
        if report['status'] != 'COMPLETE':
            ledger(experiment_id=eid, model=report['model'], status='EXTERNAL_HYPOTHESIS_INCOMPLETE', notes=path.name)
            continue
        result = {k: v for k, v in report.items() if k not in ('cv_rows', 'fractures', 'latencies')}
        result['latency'] = {'throughput_median_us_per_point': report['latency_us'], 'fold_benchmarks': report['latencies'],
                             'ten_million_points_seconds': report['latency_us'] * 10, 'includes': 'causal feature update plus single-row prediction; no IPC'}
        result['model_bytes_max'] = max(r['model_bytes'] for r in report['latencies'])
        fractures = pd.DataFrame(report['fractures'])
        result['time_profile'] = fractures.loc[fractures.kind == 'online_time'].groupby('slice').ts_auc.mean().to_dict()
        result['age_profile'] = fractures.loc[fractures.kind == 'evidence_age_with_negative_controls'].groupby('slice').ts_auc.mean().to_dict()
        decision = promotion_decision(result, primary)
        if decision['decision'] == 'PROMOTE': raise RuntimeError('Auxiliary candidate meets primary gates; complete deployment integration before importing promotion')
        research.state['results'].append(result); research.state['cv_rows'].extend(report['cv_rows']); research.state['fractures'].extend(report['fractures'])
        research.state['promotions'].append({'experiment_id': eid, 'timestamp': utc_now(), **decision})
        ledger(experiment_id=eid, model=report['model'], sampling_policy=report['sampling'], ts_auc=result['mean'], worst_fold=result['worst_fold'], latency=result['latency_us'],
               status='REAL_DEV_AUXILIARY_COMPLETE', notes=json.dumps({'decision': decision, 'report': str(path.relative_to(ROOT)), 'elapsed_runtime': result['elapsed_seconds']}))
        print('IMPORTED', eid, result['mean'], decision['decision'], flush=True)
    research.save()


if __name__ == '__main__': main()

"""Immutable report-level synthesis of numerical gates and available engineering.

This does not freeze final roles, overwrite the live experiment ledger, create
a final handoff, or infer PASS from absent evidence.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.decision import evaluate_candidate
from src.next.research_clock import ResearchClock
from src.utils.artifacts import write_json, sha256, utc_now

PRIMARY_GATES = {'primary_mean_delta', 'primary_median_delta', 'primary_worst_fold_delta',
    'final_bootstrap_replicates', 'paired_bootstrap_probability'}
ROBUST_GATES = {'alternate_nonnegative_count', 'historical_worst_delta', 'supported_mechanism_worst_delta'}
LATENCY_GATES = {'point_latency_ratio', 'amortized_latency_ratio'}
ENGINEERING_CHECKS = {'full_primary_oof_reproduction', 'full_runtime_parity', 'full_inference_oof_parity',
    'single_pass_future_suffix_contract', 'cold_import_readonly', 'official_local_wire',
    'independent_raw_all_dev_match', 'final_artifact_latency', 'frozen_source_identity'}


def classify(original, decision, marginal=True):
    if original['status'] in ('BUG', 'SCREEN_FAIL'):
        return original['status']
    if original.get('full', {}).get('status') != 'COMPLETE':
        return 'FULL_CV_PENDING'
    failed = set(decision['failed_gates'])
    if failed & PRIMARY_GATES:
        return 'FULL_FAIL'
    if failed & ROBUST_GATES:
        return 'ROBUSTNESS_FAIL'
    if failed & LATENCY_GATES:
        return 'LATENCY_FAIL'
    if 'engineering' in failed:
        return 'BUG'
    if not marginal:
        return 'REDUNDANT'
    if decision['promoted']:
        return 'NEW_GENERATION' if decision['new_generation_seal_worthy'] else 'PROMOTED'
    return 'ROBUSTNESS_PENDING'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--verify-payloads', action='store_true')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Simple immutable snapshot tag required')
    guard = SealGuard(ROOT)
    out = ROOT/'artifacts/next/evidence_snapshots'/args.tag
    if out.exists():
        raise RuntimeError('An evidence snapshot must not be overwritten')
    out.mkdir(parents=True)
    sources = {}
    def read(relative, required=True):
        path = ROOT/relative
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        record = json.loads(path.read_text(encoding='utf-8'))
        sources[str(path.relative_to(ROOT))] = sha256(path)
        return record
    def csv(relative):
        path = ROOT/relative
        sources[str(path.relative_to(ROOT))] = sha256(path)
        return pd.read_csv(path)
    def payloads(folder, report):
        if args.verify_payloads:
            for name, digest in report.get('files_sha256', {}).items():
                if sha256(ROOT/folder/name) != digest:
                    raise RuntimeError('Evidence payload changed: '+str(Path(folder)/name))
    policy = read('ROBUSTNESS_SPLIT_LOCK.json')
    reference = read('artifacts/next/REFERENCE_RESTORATION.json')['references']['PRIMARY']
    regime_lock = read('artifacts/next/current_forensic/REGIME_ASSIGNMENT_LOCK.json')
    required_mechanisms = sorted(name for name, count in regime_lock['mechanism_counts'].items() if count >= 100 and name != 'NO_BREAK_CONTROL')
    reference_h = read('artifacts/next/robustness/reference/PRIMARY/LEAVE_ONE_H_REGIME_OUT/RESULTS.json')
    reference_alts = [read(f'artifacts/next/robustness/reference/PRIMARY/ALT_{j}/RESULTS.json') for j in (1, 2, 3)]
    latency_reference = read('artifacts/next/latency/research_v1/PRIMARY_reference.json')
    artifact_latency_reference = read('artifacts/next/latency/all_dev_v1/PRIMARY.json', required=False)
    records, rows = [], []
    for path in sorted((ROOT/'artifacts/next/experiments').glob('*/RESULTS.json')):
        original = read(path.relative_to(ROOT))
        candidate = original['candidate']
        eid = candidate['experiment_id']
        if eid != path.parent.name:
            raise RuntimeError('Experiment identity differs from its directory')
        full = original.get('full', {})
        valid_full = original['status'] != 'BUG' and full.get('status') == 'COMPLETE'
        evidence = {'required_supported_mechanisms': required_mechanisms}
        supplemental = {}
        if valid_full:
            scores = np.asarray(full['fold_scores'], dtype=float)
            if scores.shape != (5,) or not np.isfinite(scores).all():
                raise RuntimeError('An incomplete CV record claims completion')
            for name, value in [('mean', scores.mean()), ('median', np.median(scores)), ('worst_fold', scores.min())]:
                if abs(full[name]-value) > 1e-14:
                    raise RuntimeError('Stored fold aggregate does not match its scores')
            bootstrap = full['paired_bootstrap']
            forensic_folder = f'artifacts/next/forensic/{eid}'
            forensic = read(forensic_folder+'/RESULTS.json', required=False)
            if forensic:
                if forensic['status'] != 'PASS':
                    raise RuntimeError('An available forensic result failed')
                payloads(forensic_folder, forensic)
                bootstrap = forensic['all_time_paired_bootstrap']
                fractures = csv(forensic_folder+'/PAIRED_FRACTURES.csv')
                if len(fractures) != 24 or fractures.scope.duplicated().any():
                    raise RuntimeError('Incomplete or duplicated forensic scopes')
                mechanism = fractures.loc[fractures.scope.str.startswith('mechanism_')].copy()
                mechanism['name'] = mechanism.scope.str.removeprefix('mechanism_')
                if set(mechanism.loc[mechanism.supported, 'name']) != set(required_mechanisms):
                    raise RuntimeError('Candidate-dependent mechanism support is forbidden')
                evidence['mechanism_deltas'] = dict(zip(mechanism.name, mechanism.estimate_mean_delta))
                supplemental['mechanisms'] = mechanism.to_dict('records')
                supplemental['complementarity_vs_primary'] = forensic['complementarity']
            if abs(bootstrap['estimate_mean_delta']-(full['mean']-reference['mean'])) > 1e-14:
                raise RuntimeError('Paired-bootstrap point estimate differs from its CV mean')
            evidence.update(mean_delta=full['mean']-reference['mean'], median_delta=full['median']-reference['median'],
                worst_delta=full['worst_fold']-reference['worst'], bootstrap_replicates=bootstrap['replicates'],
                bootstrap_probability=bootstrap['probability_delta_positive'])
            supplemental['paired_bootstrap'] = bootstrap
            robustness = read(f'artifacts/next/robustness/candidates/{eid}/RESULTS.json', required=False)
            if robustness:
                if robustness['status'] != 'PASS' or len(robustness['summaries']) != 4:
                    raise RuntimeError('Incomplete stress summary')
                splits = {record['split_name']: record for record in robustness['summaries']}
                deltas = []
                for j in (1, 2, 3):
                    current = splits[f'ALT_{j}']
                    if len(current['fold_scores']) != 5:
                        raise RuntimeError('Incomplete alternate folds')
                    delta = float(np.mean(current['fold_scores'])-reference_alts[j-1]['mean'])
                    if abs(delta-current['mean_delta']) > 1e-14:
                        raise RuntimeError('Alternate delta does not match the frozen reference')
                    deltas.append(delta)
                h = splits['LEAVE_ONE_H_REGIME_OUT']
                evidence.update(alternate_mean_deltas=deltas, historical_worst_delta=min(h['fold_scores'])-min(reference_h['fold_scores']))
                supplemental['robustness_splits'] = robustness['summaries']
                supplemental['historical_worst'] = min(h['fold_scores'])
            matched = read(f'artifacts/next/matched_contrast/{eid}/RESULTS.json', required=False)
            if matched:
                contrast = csv(f'artifacts/next/matched_contrast/{eid}/PAIRED_FRACTURES.csv')
                supported = contrast.loc[contrast.supported & contrast.scope.str.startswith('mechanism_')]
                evidence.update(matched_statistic_mean_delta=matched['all_time']['estimate_mean_delta'],
                    matched_supported_mechanism_gain=float(supported.estimate_mean_delta.max()))
            variant = {'log_energy': 'log_energy_fast_h', 'serial_variance': 'serial_variance_fast_h'}.get(candidate.get('extension', {}).get('kind'), 'fast_h')
            latency = read(f'artifacts/next/latency/research_v1/{eid}_{variant}.json', required=False)
            if latency:
                if latency['status'] != 'PASS' or not latency['isolated_from_workspace_research_workers']:
                    raise RuntimeError('A latency measurement was not isolated and valid')
                if latency['identity']['model_sha256'] != sha256(path.parent/'full/fold_0.joblib'):
                    raise RuntimeError('Latency was measured on a different model')
                evidence.update(point_latency_ratio=latency['point_distribution_us']['median']/latency_reference['point_distribution_us']['median'],
                    amortized_latency_ratio=latency['real_history_initialization']['amortized_us_per_point']/latency_reference['real_history_initialization']['amortized_us_per_point'])
                supplemental['latency'] = latency
            # Prefer the exact complete-DEV artifact with its contemporaneous
            # complete-DEV PRIMARY baseline when both measurements exist.
            artifact_latency = read(f'artifacts/next/latency/all_dev_v1/{eid}.json', required=False)
            if artifact_latency is not None:
                if artifact_latency_reference is None:
                    raise RuntimeError('All-DEV artifact latency lacks its matching baseline')
                descriptor = read(f'artifacts/next/refits/{eid}/cache_research_v1/model.json', required=True)
                if artifact_latency['status'] != 'PASS' or not artifact_latency['isolated_from_workspace_research_workers'] or artifact_latency['identity']['model_sha256'] != descriptor['model_sha256']:
                    raise RuntimeError('All-DEV artifact latency identity differs')
                supplemental['cv_model_latency'] = supplemental.get('latency')
                supplemental['latency'] = artifact_latency
                supplemental['latency_model_kind'] = 'ALL_DEV_REFIT_ARTIFACT'
                evidence.update(point_latency_ratio=artifact_latency['point_distribution_us']['median']/artifact_latency_reference['point_distribution_us']['median'],
                    amortized_latency_ratio=artifact_latency['real_history_initialization']['amortized_us_per_point']/artifact_latency_reference['real_history_initialization']['amortized_us_per_point'])
            engineering = read(f'artifacts/next/final_engineering/{eid}/RESULTS.json', required=False)
            if engineering:
                for relative, digest in engineering['evidence_files_sha256'].items():
                    if sha256(ROOT/relative) != digest:
                        raise RuntimeError('Final engineering source evidence changed')
                if set(engineering['checks']) != ENGINEERING_CHECKS:
                    raise RuntimeError('Incomplete final engineering checklist')
                evidence['engineering_pass'] = engineering['status'] == 'PASS' and all(value is True for value in engineering['checks'].values())
        decision = evaluate_candidate(policy, evidence)
        marginal = original.get('marginal_gate_pass', True)
        status = classify(original, decision, marginal)
        statistical = PRIMARY_GATES|ROBUST_GATES
        numerical_complete = all(decision['gates'][name]['status'] == 'PASS' for name in statistical)
        record = {'candidate': candidate, 'original_status': original['status'], 'assessed_status': status,
            'evidence': evidence, 'decision': decision, 'supplemental': supplemental,
            'all_statistical_gates_passed': numerical_complete, 'marginal_complexity_gate_pass': marginal,
            'final_role_frozen': False, 'original_result_sha256': sha256(path)}
        records.append(record)
        rows.append({'experiment_id': eid, 'parent_id': candidate.get('parent_id'), 'generation': candidate.get('generation'),
            'phase': candidate.get('phase'), 'hypothesis': candidate.get('hypothesis'), 'change': candidate.get('change'),
            'learner': candidate.get('learner', 'fixed_late_blend'), 'weighting': candidate.get('weighting'),
            'params': json.dumps(candidate.get('params', {}), sort_keys=True), 'original_status': original['status'], 'status': status,
            'primary_cv_mean': full.get('mean') if valid_full else None, 'median': full.get('median') if valid_full else None,
            'worst_fold': full.get('worst_fold') if valid_full else None, 'delta_vs_primary': evidence.get('mean_delta'),
            'bootstrap_replicates': evidence.get('bootstrap_replicates'), 'bootstrap_p': evidence.get('bootstrap_probability'),
            'bootstrap_ci_low': supplemental.get('paired_bootstrap', {}).get('ci_2_5'), 'bootstrap_ci_high': supplemental.get('paired_bootstrap', {}).get('ci_97_5'),
            'alternate_mean_deltas': json.dumps(evidence.get('alternate_mean_deltas')), 'historical_worst': supplemental.get('historical_worst'),
            'supported_mechanism_worst_delta': min((evidence.get('mechanism_deltas', {}).get(name, float('inf')) for name in required_mechanisms), default=None) if 'mechanism_deltas' in evidence else None,
            'point_latency_us': supplemental.get('latency', {}).get('point_distribution_us', {}).get('median'),
            'all_statistical_gates_passed': numerical_complete, 'failed_gates': '|'.join(decision['failed_gates']),
            'pending_gates': '|'.join(decision['pending_gates']), 'marginal_complexity_gate_pass': marginal,
            'reason': original.get('reason'), 'elapsed_seconds': original.get('elapsed_seconds'), 'seal_rows': 0, 'reduced_new_uses': 0})
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'MODEL_EVIDENCE.csv', index=False)
    write_json(out/'CANDIDATE_EVIDENCE.json', records)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'tag': args.tag,
        'experiments': len(rows), 'status_counts': frame.status.value_counts().to_dict(),
        'statistical_gate_pass_candidates': frame.loc[frame.all_statistical_gates_passed, 'experiment_id'].tolist(),
        'source_reports_sha256': sources, 'script_sha256': sha256(Path(__file__)),
        'files_sha256': {name: sha256(out/name) for name in ['MODEL_EVIDENCE.csv', 'CANDIDATE_EVIDENCE.json']},
        'clock': ResearchClock(ROOT).snapshot(), 'seal_rows': 0, 'reduced_new_uses': 0,
        'payload_byte_checks_requested': args.verify_payloads,
        'authority': 'Read-only synthesis of completed evidence; no missing evidence is promoted to PASS. This snapshot neither selects final roles nor closes research or creates a handoff.'})
    print('EVIDENCE SNAPSHOT', args.tag, len(rows), frame.status.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()

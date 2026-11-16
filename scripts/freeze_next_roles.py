"""Freeze at most three DEV roles only during the additional final-audit phase."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.research_clock import ResearchClock
from src.next.active_research_time import verify_final_duration
from src.next.guard import SealGuard
from src.next.decision import evaluate_candidate
from src.utils.artifacts import write_json, sha256, utc_now


def select_roles(rows):
    eligible = [row for row in rows if row['decision']['promoted'] and row['marginal_complexity_gate_pass']]
    if not eligible:
        return {'PRIMARY_DEV_CANDIDATE': 'FROZEN_PRIMARY_REFERENCE', 'ROBUST_DEV_CANDIDATE': None, 'FAST_DEV_CANDIDATE': None}
    primary = sorted(eligible, key=lambda row: (-row['mean'], -row['worst'], -row['median'], row['id']))[0]
    remaining = [row for row in eligible if row['id'] != primary['id']]
    robust = [row for row in remaining if row['mean'] >= primary['mean']-.005 and row['median'] >= primary['median']-.005]
    robust = sorted(robust, key=lambda row: (-row['worst'], row['std'], row['id']))
    robust = robust[0] if robust else None
    fast = [row for row in remaining if (robust is None or row['id'] != robust['id']) and row['mean'] >= primary['mean']-.010 and row['worst'] >= primary['worst']-.010]
    fast = sorted(fast, key=lambda row: (row['point_us'], row['amortized_us'], -row['mean'], row['id']))[:5]
    return {'PRIMARY_DEV_CANDIDATE': primary['id'], 'ROBUST_DEV_CANDIDATE': robust['id'] if robust else None,
        'FAST_DEV_CANDIDATE': fast[0]['id'] if fast else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    clock = ResearchClock(ROOT).snapshot()
    if not clock['minimum_research_met'] or not clock['final_audit_started']:
        raise RuntimeError('Final roles can only be frozen after ten research hours and explicit additional-audit start')
    substantive_duration = verify_final_duration(ROOT)
    out = ROOT/'artifacts/next/final_selection'
    if (out/'RESULTS.json').exists():
        raise RuntimeError('Final roles are already frozen')
    guard = SealGuard(ROOT)
    snapshot_dir = ROOT/args.snapshot
    snapshot = json.loads((snapshot_dir/'RESULTS.json').read_text(encoding='utf-8'))
    if snapshot['status'] != 'PASS' or not snapshot['payload_byte_checks_requested']:
        raise RuntimeError('A verified-payload evidence snapshot is required')
    for relative, digest in snapshot['source_reports_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Source evidence changed after the final snapshot')
    for relative, digest in snapshot['files_sha256'].items():
        if sha256(snapshot_dir/relative) != digest:
            raise RuntimeError('Evidence snapshot payload changed')
    evidence = json.loads((snapshot_dir/'CANDIDATE_EVIDENCE.json').read_text(encoding='utf-8'))
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    role_policy_path = ROOT/'configs/next_final_role_policy.json'
    role_policy = json.loads(role_policy_path.read_text(encoding='utf-8'))
    if role_policy['status'] != 'LOCKED' or role_policy['promotion_policy_sha256'] != sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'):
        raise RuntimeError('Frozen final-role policy changed')
    rows = []
    for record in evidence:
        eid = record['candidate']['experiment_id']
        original = json.loads((ROOT/'artifacts/next/experiments'/eid/'RESULTS.json').read_text(encoding='utf-8'))
        if original.get('full', {}).get('status') != 'COMPLETE' or original['status'] == 'BUG':
            continue
        decision = evaluate_candidate(policy, record['evidence'])
        if decision != record['decision']:
            raise RuntimeError('Evidence decision cannot be independently reconstructed')
        full = original['full']
        latency = record['supplemental'].get('latency', {})
        rows.append({'id': eid, 'mean': full['mean'], 'median': full['median'], 'worst': full['worst_fold'],
            'std': float(np.std(full['fold_scores'])), 'fold_scores': full['fold_scores'],
            'point_us': latency.get('point_distribution_us', {}).get('median', float('inf')),
            'amortized_us': latency.get('real_history_initialization', {}).get('amortized_us_per_point', float('inf')),
            'decision': decision, 'marginal_complexity_gate_pass': record['marginal_complexity_gate_pass'],
            'statistical_gates_passed': record['all_statistical_gates_passed'], 'candidate': record['candidate']})
    roles = select_roles(rows)
    lookup = {row['id']: row for row in rows}
    if roles['PRIMARY_DEV_CANDIDATE'] == 'FROZEN_PRIMARY_REFERENCE':
        raise RuntimeError('No new candidate qualifies; retain the existing PRIMARY and use the explicit old-artifact fallback packager')
    primary = lookup[roles['PRIMARY_DEV_CANDIDATE']]
    robust = lookup[roles['ROBUST_DEV_CANDIDATE']] if roles['ROBUST_DEV_CANDIDATE'] else None
    unresolved = []
    for row in rows:
        if not row['statistical_gates_passed'] or not row['marginal_complexity_gate_pass'] or row['decision']['promoted']:
            continue
        could_win_primary = (-row['mean'], -row['worst'], -row['median'], row['id']) < (-primary['mean'], -primary['worst'], -primary['median'], primary['id'])
        could_win_robust = (row['id'] != primary['id'] and row['mean'] >= primary['mean']-.005 and row['median'] >= primary['median']-.005 and
            (robust is None or (-row['worst'], row['std'], row['id']) < (-robust['worst'], robust['std'], robust['id'])))
        # A prospective reduced-tree candidate with passing numerical gates
        # cannot be omitted merely because its speed audit has not run yet.
        fast_nominee = row['candidate'].get('params', {}).get('n_estimators', 100) <= 50
        if could_win_primary or could_win_robust or fast_nominee:
            unresolved.append(row['id'])
    if unresolved:
        raise RuntimeError('Finish higher-priority finalist engineering before freezing roles: '+', '.join(unresolved))
    out.mkdir(parents=True, exist_ok=True)
    files, selected = {}, []
    for role, eid in roles.items():
        if eid is None:
            continue
        row = lookup[eid]
        engineering_path = ROOT/'artifacts/next/final_engineering'/eid/'RESULTS.json'
        engineering = json.loads(engineering_path.read_text(encoding='utf-8'))
        if engineering['status'] != 'PASS' or not all(engineering['checks'].values()):
            raise RuntimeError('Selected engineering receipt incomplete')
        for relative, digest in engineering['evidence_files_sha256'].items():
            if sha256(ROOT/relative) != digest:
                raise RuntimeError('Selected engineering evidence changed')
        kind = row['candidate'].get('extension', {}).get('kind')
        if kind not in ('conditional_variance', 'log_energy', 'serial_variance'):
            raise RuntimeError('Selected family still requires a reviewed final deployment adapter')
        spec = {'status': 'FROZEN_NEXT_DEV_PRIMARY' if role == 'PRIMARY_DEV_CANDIDATE' else 'FROZEN_NEXT_DEV_BACKUP',
            'role': role, 'candidate': row['candidate'], 'development_ids': sorted(guard.dev_ids),
            'known_training_ids': sorted(guard.split.mapping), 'split_sha256': guard.split.digest,
            'seal_lock_sha256': guard.lock_sha, 'selection_policy_sha256': sha256(role_policy_path),
            'evidence_snapshot_sha256': sha256(snapshot_dir/'RESULTS.json'),
            'expected_model_sha256': engineering['model_sha256'], 'entrypoint': engineering['entrypoint']}
        path = ROOT/'configs/next_roles'/(role+'.json')
        if path.exists():
            raise RuntimeError('A role specification already exists; preserve it')
        write_json(path, spec)
        files[path.relative_to(ROOT).as_posix()] = sha256(path)
        # The family entry's default train() uses that family's highest-priority
        # selected role. Other roles use their explicit role specifications.
        default_path = ROOT/{'log_energy': 'configs/next_registered_deployment.json',
            'serial_variance': 'configs/next_serial_deployment.json', 'conditional_variance': 'configs/next_deployment.json'}[kind]
        if role == 'PRIMARY_DEV_CANDIDATE' or not default_path.exists():
            if default_path.exists():
                raise RuntimeError('Default NEXT deployment spec is already frozen')
            write_json(default_path, spec)
            files[default_path.relative_to(ROOT).as_posix()] = sha256(default_path)
        selected.append({'role': role, 'experiment_id': eid, 'candidate': row['candidate'],
            'primary_cv_mean': row['mean'], 'median': row['median'], 'worst_fold': row['worst'], 'fold_scores': row['fold_scores'],
            'new_generation_seal_worthy': row['decision']['new_generation_seal_worthy'],
            'model_sha256': engineering['model_sha256'], 'descriptor_sha256': engineering['descriptor_sha256'],
            'source_model_directory': f'artifacts/next/refits/{eid}/cache_research_v1',
            'entrypoint': engineering['entrypoint'], 'runtime': engineering['runtime'],
            'specification': path.relative_to(ROOT).as_posix(), 'engineering_report_sha256': sha256(engineering_path)})
    result = {'status': 'FROZEN_DEV_SELECTION', 'created_utc': utc_now(), 'clock': clock, 'roles': roles, 'candidates': selected,
        'substantive_research_seconds': substantive_duration['substantive_research_seconds'],
        'substantive_research_report_sha256': sha256(ROOT/'artifacts/next/scheduler/SUBSTANTIVE_RESEARCH_DURATION.json'),
        'maximum_final_candidates': 3, 'role_policy_sha256': sha256(role_policy_path),
        'evidence_snapshot': str(snapshot_dir.relative_to(ROOT)), 'evidence_snapshot_sha256': sha256(snapshot_dir/'RESULTS.json'),
        'specification_files_sha256': files, 'selector_sha256': sha256(Path(__file__)),
        'seal_opened': {letter: False for letter in 'ABCD'}, 'reduced_new_uses': 0, 'public_submissions': 0,
        'selection_uncertainty': 'Conditional on adaptively searched DEV; no independent seal evaluation or post-selection coverage claim.'}
    write_json(out/'RESULTS.json', result)
    write_json(ROOT/'DEV_CANDIDATES_NEXT.json', result)
    print('FROZEN NEXT DEV ROLES', roles, flush=True)


if __name__ == '__main__':
    main()

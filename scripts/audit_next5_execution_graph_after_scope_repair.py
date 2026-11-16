"""Snapshot real dependency chronology and verify frozen execution source pins."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import csv
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--final-audit', action='store_true')
    parser.add_argument('--reviewed-recovery-probe', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+', args.tag)
    if args.final_audit:
        assert read_json(BASE / 'SESSION.json').get('audit_started_utc')
    phase = 'FINAL_AUDIT' if args.final_audit else 'RESEARCH'
    paths = {p.stem: p for p in (BASE / 'queue').glob('*.json')}
    jobs = {key: read_json(path) for key, path in paths.items()}
    repair_path = BASE / 'maintenance/COHORT_PROVENANCE_REPAIR/REPAIR.json'
    repair = read_json(repair_path) if repair_path.exists() else None
    if repair:
        assert digest(paths[repair['job']]) == repair['repaired_queue_sha256']
        assert digest(repair_path.parent / 'QUEUE_BEFORE.json') == repair['original_queue_sha256']
        assert digest(repair_path.parent / 'POLICY_LOCK.json') == repair['policy_sha256']
    allowed_late_dependencies = {repair['job']: {repair['new_success_dependency']}} if repair else {}
    api_repair_path = BASE / 'maintenance/READONLY_IMPORT_REPAIR/REPAIR.json'
    api_repair = read_json(api_repair_path) if api_repair_path.exists() else None
    if api_repair:
        assert digest(api_repair_path.parent / 'POLICY_LOCK.json') == api_repair['policy_sha256']
        changed = api_repair['changed_queue']
        assert digest(ROOT / changed['before_path']) == changed['before_sha256']
        assert digest(paths[changed['job']]) == changed['after_sha256']
        before = read_json(ROOT / changed['before_path'])
        expected = [api_repair['replacement_jobs'][name] for name in before['depends_on_success']] + ['API_IMPORT_REPAIR_RESOLUTIONS']
        assert jobs[changed['job']]['depends_on_success'] == expected
        allowed_late_dependencies[changed['job']] = set(expected)
        for original_id in api_repair['replacement_jobs']:
            original_path = api_repair_path.parent / 'queue_before' / (original_id + '.json')
            assert digest(paths[original_id]) == digest(original_path)
    from src.next5.final_test_repair_evidence import verify_execution_changes
    verify_execution_changes(jobs, paths, allowed_late_dependencies)
    from src.next5.final_graph_repair_evidence import verify_execution_changes as verify_scope_repair
    from src.next5.final_graph_repair_evidence import production_policy_paths, verify_fixture_inventory, verify_reviewed_graph_failure
    verify_scope_repair(jobs, paths, allowed_late_dependencies)
    states, inputs, edges, rows = {}, {}, {}, []
    pending_failure_resolutions = []
    for identifier, job in jobs.items():
        assert identifier == job['id']
        assert not set(job).intersection(('depends_on', 'dependencies', 'requires'))
        deps = job.get('depends_on_success', [])
        assert len(deps) == len(set(deps)) and identifier not in deps and set(deps) <= set(jobs)
        edges[identifier] = deps
        inputs[paths[identifier].relative_to(ROOT).as_posix()] = digest(paths[identifier])
        for dep in deps:
            if datetime.fromisoformat(jobs[dep]['registered_utc']) > datetime.fromisoformat(job['registered_utc']):
                assert dep in allowed_late_dependencies.get(identifier, set())
                assert datetime.fromisoformat(jobs[dep]['registered_utc']) <= datetime.fromisoformat(job['execution_repair_utc'])
        path = BASE / 'jobs' / identifier / 'STATE.json'
        if path.exists():
            states[identifier] = read_json(path)
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    active, visited = set(), set()
    def visit(identifier):
        if identifier in active:
            raise RuntimeError('Dependency cycle: ' + identifier)
        if identifier in visited:
            return
        active.add(identifier)
        for dep in edges[identifier]:
            visit(dep)
        active.remove(identifier)
        visited.add(identifier)
    for identifier in jobs:
        visit(identifier)
    for identifier, job in sorted(jobs.items()):
        state = states.get(identifier, {'status': 'PENDING'})
        checked = []
        if 'started_utc' in state:
            assert datetime.fromisoformat(job['registered_utc']) <= datetime.fromisoformat(state['started_utc'])
            if 'execution_repair_utc' in job:
                assert datetime.fromisoformat(job['execution_repair_utc']) <= datetime.fromisoformat(state['started_utc'])
            for dep in edges[identifier]:
                upstream = states[dep]
                assert upstream['status'] == 'COMPLETE' and upstream['exit_code'] == 0, (identifier, dep)
                assert datetime.fromisoformat(upstream['finished_utc']) <= datetime.fromisoformat(state['started_utc'])
                checked.append(dep)
        if state['status'] == 'COMPLETE':
            assert state['exit_code'] == 0
            result_path = ROOT / state['attempt_directory'] / 'RESULT.json'
            result = read_json(result_path)
            assert result['status'] == 'COMPLETE' and result['exit_code'] == 0
            assert result['started_utc'] == state['started_utc'] and result['finished_utc'] == state['finished_utc']
            assert digest(ROOT / state['attempt_directory'] / 'RUN.log') == state['log_sha256'] == result['log_sha256']
            inputs[result_path.relative_to(ROOT).as_posix()] = digest(result_path)
        elif state['status'] == 'QUARANTINED':
            path = BASE / 'failure_resolutions' / (identifier + '.json')
            if path.exists():
                resolution = read_json(path)
                assert resolution['status'] == 'RESOLVED_WITH_PRESERVED_FAILURE'
                assert resolution['quarantined_state_sha256'] == digest(BASE / 'jobs' / identifier / 'STATE.json')
                for name, sha in resolution['evidence_sha256'].items():
                    assert digest(ROOT / name) == sha
                inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            elif args.reviewed_recovery_probe and identifier == 'FA008_EXECUTION_GRAPH':
                verify_reviewed_graph_failure()
                pending_failure_resolutions.append({'original_job': identifier, 'replacement_job': 'FA008R_GRAPH_CHECK',
                    'resolution_claimed': False, 'reviewed_policy_scope_repair': True})
            else:
                assert not args.final_audit and api_repair and identifier in api_repair['replacement_jobs'], 'Unreviewed or unresolved final failure: ' + identifier
                replacement = api_repair['replacement_jobs'][identifier]
                assert replacement in jobs
                if state.get('started_utc'):
                    assert state['exit_code'] == 1
                    assert digest(ROOT / state['attempt_directory'] / 'RUN.log') == state['log_sha256']
                else:
                    assert state['resolved_by'] == replacement and state['no_inference_started']
                pending_failure_resolutions.append({'original_job': identifier, 'replacement_job': replacement,
                    'replacement_status': states.get(replacement, {'status': 'PENDING'})['status'],
                    'resolution_claimed': False})
        elif args.final_audit and job['kind'] not in ('final_audit', 'delivery'):
            raise RuntimeError('Unfinished pre-audit work: ' + identifier)
        rows.append({'job': identifier, 'kind': job['kind'], 'status': state['status'],
            'registered_utc': job['registered_utc'], 'started_utc': state.get('started_utc', ''),
            'finished_utc': state.get('finished_utc', ''), 'prerequisites': ';'.join(edges[identifier]),
            'successful_prerequisites_verified_before_actual_start': ';'.join(checked),
            'preserved_metadata_execution_repair': identifier in allowed_late_dependencies,
            'scientific_recipe_unchanged': (digest(BASE / 'registrations' / (job['candidate_id'] + '.json')) == job['registration_sha256']) if job['kind'] == 'research' else ''})
        assert rows[-1]['scientific_recipe_unchanged'] is not False
    # Explicit .py path maps only: never reinterpret raw-data hashes as source files.
    pins = {}
    policy_paths = list((ROOT / 'configs/next5').glob('*.json')) + list((BASE / 'registrations').glob('*.json'))
    policy_paths += production_policy_paths()
    for path in sorted(set(policy_paths)):
        pending = [read_json(path)]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                for name, sha in value.items():
                    if isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) and name.endswith('.py'):
                        source = (ROOT / name).resolve()
                        assert source.is_relative_to(ROOT.resolve()) and source.is_file()
                        assert digest(source) == sha, (path, name)
                        if name in pins:
                            assert pins[name] == sha
                        pins[name] = sha
                    elif isinstance(sha, (dict, list)):
                        pending.append(sha)
            elif isinstance(value, list):
                pending.extend(value)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    supervisor = read_json(BASE / 'runtime/SUPERVISOR.json')
    assert digest(ROOT / 'src/next5/scheduler.py') == supervisor['source_sha256']
    out = BASE / ('final_audit' if args.final_audit else 'integrity_audits') / args.tag / 'execution_graph'
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'JOB_CHRONOLOGY.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'registered_jobs': len(jobs), 'status_counts': dict(Counter(r['status'] for r in rows)),
        'acyclic_graph': True, 'actual_successful_dependencies_precede_starts': True,
        'completed_worker_logs_hash_verified': True, 'frozen_python_source_pins': pins,
        'preserved_unstarted_metadata_repair': repair,
        'reviewed_readonly_import_repair': api_repair,
        'pending_preserved_failure_resolutions': pending_failure_resolutions,
        'all_preserved_failures_resolved': not pending_failure_resolutions,
        'reviewed_fixture_policy_scope': verify_fixture_inventory(),
        'running_scheduler_source_unchanged': True, 'snapshot_inputs_sha256': inputs,
        'table_sha256': digest(out / 'JOB_CHRONOLOGY.csv'), 'source_sha256': digest(Path(__file__)),
        'new_fits': 0, 'real_data_read': False,
        'scope': 'Snapshot only. PENDING/RUNNING states may advance; no pending job is claimed complete.'}, immutable=True)
    progress('NEXT5 execution graph audited', str(len(jobs)) + ' jobs and ' + str(len(pins)) + ' frozen source paths', phase=phase)
    print('NEXT5 EXECUTION GRAPH PASS', len(jobs), 'jobs', len(pins), 'source paths', flush=True)


if __name__ == '__main__':
    main()

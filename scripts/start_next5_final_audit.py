"""Register the separate audit only after observed research and completed work."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.finalization import begin_audit
from src.next5.scheduler import register_job


def main():
    session = read_json(BASE / 'SESSION.json')
    account = read_json(BASE / 'runtime/ACCOUNTING.json')
    assert session['status'] == 'ACTIVE_RESEARCH' and not session.get('audit_started_utc')
    assert 0 <= time.time() - account['wall_timestamp'] <= 60
    assert account['productive_research_seconds'] >= 36000, 'No audit registration before the actual research minimum'
    assert read_json(ROOT / 'DEV_CANDIDATES_NEXT5.json')['status'] == 'FROZEN_NEXT5_FINAL_ROLES'
    for path in (BASE / 'queue').glob('*.json'):
        job = read_json(path)
        if job['kind'] in ('final_audit', 'delivery'):
            continue
        state_path = BASE / 'jobs' / job['id'] / 'STATE.json'
        assert state_path.exists(), 'Pending research work: ' + job['id']
        state = read_json(state_path)
        if state['status'] == 'QUARANTINED':
            resolution = read_json(BASE / 'failure_resolutions' / (job['id'] + '.json'))
            assert resolution['quarantined_state_sha256'] == digest(state_path)
        else:
            assert state['status'] == 'COMPLETE' and state['exit_code'] == 0, 'Unfinished work: ' + job['id']
    jobs = (
        ('FA001_FIT_INVENTORY', 'scripts/audit_next5_fit_inventory.py', ['--tag', 'final_v1', '--final-audit'], 'Reconcile every primitive learner fit and original RP group/source/model receipt'),
        ('FA002_COMPLETE_TESTS', 'scripts/run_next5_full_tests.py', ['--tag', 'final_v1'], 'Complete project suite under explicit raw/old per-series data access guard'),
        ('FA003_FROZEN_STATISTICS', 'scripts/audit_next5_frozen_statistics.py', ['--tag', 'final_v1'], 'Fresh numerical reproduction of 2470 candidate fold/scopes and 95 actual official scorer comparisons'),
        ('FA004_SYNTHETIC_BYTES', 'scripts/audit_next5_synthetic_receipts.py', ['--tag', 'final_v1'], 'Byte-hash all 3536 synthetic cohort files and exact fitted-model/reference lineage'),
        ('FA005_APPEND_LEDGER', 'scripts/append_next5_final_ledger.py', ['--tag', 'final_v1'], 'Append 315 NEXT5 rows while preserving the original 276 rows and all original bytes'),
        ('FA006_FROZEN_BOUNDARY', 'scripts/audit_next5_frozen_boundary.py', ['--tag', 'final_v1'], 'Recheck raw/source/old-delivery/staged bytes, partitions, environment and unopened guards'),
        ('FA007_SCALAR_TABLES', 'scripts/build_next5_scalar_tables.py', ['--tag', 'final_v1', '--complete'], 'Traceable final tables from complete PRIMARY, ALT/H and all 97 fitted stress artifacts'),
        ('FA008_EXECUTION_GRAPH', 'scripts/audit_next5_execution_graph.py', ['--tag', 'final_v1', '--final-audit'], 'Actual dependency chronology, completion logs, preserved failures and frozen Python source pins'),
        ('FA009_PORTABLE_REPLAY', 'scripts/verify_next5_portable_payload.py', ['--tag', 'final_v1'], 'Copy the raw-free payload and replay original BL80/B2/D2 fingerprints in fresh processes'),
    )
    assert not (BASE / 'final_audit/PLAN.json').exists()
    sources = {script: digest(ROOT / script) for _, script, _, _ in jobs}
    # Freeze the helpers and report/delivery tools as well as entry scripts.
    # No audit-time code edit may silently change the interpretation or payload.
    preparation_sources = [*(ROOT / 'src/next5').glob('*.py'),
        *(ROOT / 'scripts').glob('*next5*.py'), *(ROOT / 'scripts').glob('*next5*.ps1'),
        *(ROOT / 'tests').glob('*next5*.py')]
    for path in preparation_sources:
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    previous = None
    for offset, (identifier, script, options, description) in enumerate(jobs):
        job = {'id': identifier, 'kind': 'final_audit', 'priority': 200 + offset,
            'argv': [script, *options], 'description': description, 'retry_safe': False}
        if previous:
            job['depends_on_success'] = [previous]
        register_job(job)
        previous = identifier
    atomic_json(BASE / 'final_audit/PLAN.json', {'created_utc': utc_now(), 'status': 'REGISTERED_SEPARATE_AUDIT',
        'jobs': [r[0] for r in jobs], 'source_sha256': sources,
        'research_minimum_seconds': 36000, 'separate_observed_audit_minimum_seconds': 3000,
        'audit_target_maximum_seconds': 3600,
        'fresh_new_role_raw_training': 'NOT_REQUIRED_NO_NEW_NEXT5_ROLE',
        'no_idle_padding': True, 'source_script_sha256': digest(Path(__file__))}, immutable=True)
    session = begin_audit()
    progress('Separate NEXT5 final audit started', session['audit_started_utc'], phase='FINAL_AUDIT')
    print('NEXT5 FINAL AUDIT STARTED', session['audit_started_utc'], flush=True)


if __name__ == '__main__':
    main()

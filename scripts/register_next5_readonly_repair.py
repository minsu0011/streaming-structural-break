"""Preserve two blocked imports and register fresh read-only API audits."""
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.scheduler import register_job


def main():
    parent = BASE / 'maintenance/READONLY_IMPORT_REPAIR'
    pause = read_json(parent / 'PAUSE.json')
    owner = pause['owner']
    process = psutil.Process(owner['pid'])
    assert process.create_time() == owner['process_create_time'] and process.status() == psutil.STATUS_STOPPED
    cold = read_json(parent / 'COLD_PROBE.json')
    assert cold['status'] == 'PASS' and cold['persistent_filesystem_write_attempts'] == 0
    assert cold['batch_probability_max_abs_difference'] <= 1e-15
    derivation = read_json(parent / 'DERIVATION.json')
    assert digest(ROOT / 'scripts/verify_next5_streaming.py') == derivation['original_sha256']
    assert digest(ROOT / 'scripts/verify_next5_streaming_readonly_core.py') == derivation['derived_sha256']
    api_policy = read_json(BASE / 'runtime_validation/API_POLICY_LOCK.json')
    candidates = list(api_policy['candidates'])
    assert len(candidates) == 7
    failed = ['API_E2_SPARSE_START_MIXTURE', 'API_M1_ROBUST_LOCATION_LIKELIHOOD']
    sources = ('src/next5/readonly_streaming.py', 'src/next/readonly_bootstrap.py',
        'scripts/verify_next5_streaming.py', 'scripts/verify_next5_streaming_readonly_core.py',
        'scripts/verify_next5_streaming_repaired.py', 'scripts/prepare_next5_readonly_core.py',
        'scripts/probe_next5_readonly_entry.py', 'scripts/register_next5_readonly_repair.py',
        'scripts/resolve_next5_readonly_repair.py', 'tests/test_next5_readonly_entry.py')
    atomic_json(parent / 'POLICY_LOCK.json', {'created_utc': utc_now(),
        'sources_sha256': {name: digest(ROOT / name) for name in sources},
        'candidates': candidates, 'actual_failed_jobs': failed,
        'original_API_policy_sha256': digest(BASE / 'runtime_validation/API_POLICY_LOCK.json'),
        'derivation_sha256': digest(parent / 'DERIVATION.json'), 'cold_probe_sha256': digest(parent / 'COLD_PROBE.json'),
        'reason': 'Original core import initializes disk caches before the existing disable-loaded-caches call; cold write audit correctly blocks it',
        'repair': 'Use the existing pinned NEXT4 read-only bootstrap in an additive entry; preserve core, feature, model, descriptor, scoring, prefix and write guards',
        'fresh_output_directory': 'artifacts/next5/runtime_validation/api_readonly_v1',
        'old_audits_restarted_in_place': False, 'new_fits': 0, 'new_candidate_or_seed': False,
        'actual_persistent_writes_permitted': False}, immutable=True)
    register_job({'id': 'READONLY_IMPORT_CONTRACTS', 'kind': 'validation', 'priority': 35,
        'argv': ['-m', 'pytest', 'tests/test_next5_readonly_entry.py', '-q',
            '--junitxml=artifacts/next5/maintenance/READONLY_IMPORT_REPAIR/CONTRACTS.xml'],
        'description': 'Cold public import is read-only, original batch output matches, only import/output audit locations changed',
        'retry_safe': False})
    before = parent / 'queue_before'
    before.mkdir(exist_ok=False)
    replacements = {}
    for cid in candidates:
        old_id, new_id = 'API_' + cid, 'API_RO_' + cid
        old_queue = BASE / 'queue' / (old_id + '.json')
        original = read_json(old_queue)
        shutil.copyfile(old_queue, before / old_queue.name)
        state_path = BASE / 'jobs' / old_id / 'STATE.json'
        if old_id in failed:
            state = read_json(state_path)
            assert state['status'] in ('QUARANTINED', 'RUNNING', 'FAILED')
            log = ROOT / state['attempt_directory'] / 'RUN.log'
            assert 'Research inference attempted persistent filesystem write: os.mkdir' in log.read_text(encoding='utf-8')
            shutil.copyfile(log, parent / (old_id + '_FAILED.log'))
            atomic_json(parent / (old_id + '_STATE_AT_REPAIR.json'), state, immutable=True)
        else:
            assert not state_path.exists(), 'Unstarted original audit required'
            atomic_json(state_path, {**original, 'status': 'QUARANTINED', 'quarantined_utc': utc_now(),
                'reason': 'Unstarted audit superseded after the shared import-time cache bug was proven',
                'resolved_by': new_id, 'no_inference_started': True}, immutable=True)
        register_job({'id': new_id, 'kind': 'validation', 'priority': 86,
            'depends_on_success': [*original['depends_on_success'], 'READONLY_IMPORT_CONTRACTS'],
            'argv': ['scripts/verify_next5_streaming_repaired.py', '--candidate', cid],
            'description': 'Reviewed import-only repair: complete original all-RP streaming/prefix audit in a fresh directory for ' + cid,
            'retry_safe': False})
        replacements[old_id] = new_id
    register_job({'id': 'API_IMPORT_REPAIR_RESOLUTIONS', 'kind': 'validation', 'priority': 87,
        'depends_on_success': list(replacements.values()), 'argv': ['scripts/resolve_next5_readonly_repair.py'],
        'description': 'Resolve seven preserved original audit records only after every replacement API audit passes',
        'retry_safe': False})
    latency_path = BASE / 'queue/LATENCY_FIXED_ARTIFACT_BLOCKS.json'
    assert not (BASE / 'jobs/LATENCY_FIXED_ARTIFACT_BLOCKS/STATE.json').exists()
    shutil.copyfile(latency_path, before / latency_path.name)
    latency = read_json(latency_path)
    original_sha = digest(latency_path)
    assert set(latency['depends_on_success']) == set(replacements)
    stamp = utc_now()
    revised = {**latency, 'depends_on_success': [replacements[name] for name in latency['depends_on_success']] + ['API_IMPORT_REPAIR_RESOLUTIONS'],
        'execution_repair_utc': stamp, 'execution_repair': 'Require successful fresh read-only API replacements; original timing implementation and fixed blocks unchanged'}
    atomic_json(latency_path, revised)
    atomic_json(parent / 'REPAIR.json', {'created_utc': stamp, 'status': 'REGISTERED_READONLY_IMPORT_REPAIR',
        'policy_sha256': digest(parent / 'POLICY_LOCK.json'), 'replacement_jobs': replacements,
        'actual_failed_original_API_jobs': failed, 'unstarted_original_API_jobs': [name for name in replacements if name not in failed],
        'changed_queue': {'job': latency['id'], 'before_path': (before / latency_path.name).relative_to(ROOT).as_posix(),
            'before_sha256': original_sha, 'after_sha256': digest(latency_path)},
        'original_API_queues_unchanged': True, 'original_numeric_sources_unchanged': True,
        'resolution_requires_all_seven_successful_fresh_audits': True, 'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Read-only API repair registered', 'Two blocked imports and five unstarted audits preserved; fresh seven API runs plus cold contract and success-only resolution')
    print('NEXT5 READONLY REPAIR REGISTERED: seven fresh audits, no new fits', flush=True)


if __name__ == '__main__':
    main()

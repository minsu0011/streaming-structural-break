"""Submit the frozen diagnostic with the queue runner's arguments schema."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, note


def main():
    lock = read_json(BASE / 'CONDITIONAL_IMPLEMENTATION_LOCK.json')
    assert all(digest(ROOT / name) == sha for name, sha in lock['source_sha256'].items())
    assert read_json(BASE / 'CONDITIONAL_DIAGONAL_GATE.json')['status'] == 'PASS'
    path = BASE / 'runtime/TASK_QUEUE.json'
    items = read_json(path) if path.exists() else []
    ids = {item['id'] for item in items}
    new = []
    for case in ('N0', 'N1', 'N2', 'N3'):
        for length in (1000, 2048, 5000):
            task = {'id': f'conditional_{case}_H{length}', 'status': 'READY',
                'script': 'scripts/next7/conditional_null_grid.py',
                'source_sha256': digest(ROOT / 'scripts/next7/conditional_null_grid.py'),
                'arguments': ['--mode', 'run', '--case', case, '--H-length', str(length)],
                'gates': ['PREFLIGHT_NEXT7.json', 'audit/FAST_CONDITIONAL_TEST_GATE.json',
                    'CONDITIONAL_DIAGONAL_GATE.json'],
                'hypothesis': 'F4_NULL_GROWTH_CORRECTION'}
            assert task['id'] not in ids, 'This submission is immutable and runs once'
            assert not (BASE / 'runtime/queue' / task['id']).exists()
            new.append(task)
    save('CONDITIONAL_QUEUE_SCHEMA_ERRATUM_001.json', {
        'status': 'REPAIRED_BEFORE_DISPATCH', 'created_utc': utc_now(),
        'issue': 'Frozen grid driver queue() uses args, but queue_runner consumes arguments.',
        'scope': 'Unused submission helper only; prepare/run and all scientific sources unchanged.',
        'grid_queue_function_never_invoked': True, 'grid_jobs_launched_before_repair': 0,
        'remedy': 'Separate schema-correct submission wrapper; frozen implementation remains byte-identical.',
        'wrapper_sha256': digest(__file__),
        'runner_sha256': digest(ROOT / 'scripts/next7/queue_runner.py'),
        'implementation_lock_sha256': digest(BASE / 'CONDITIONAL_IMPLEMENTATION_LOCK.json'),
        'new_tasks': new})
    save('runtime/TASK_QUEUE.json', items + new, False)
    note('Conditional null grid queued after submission schema check',
        '12 fixed diagnostic tasks; scientific implementation unchanged; max two workers',
        'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

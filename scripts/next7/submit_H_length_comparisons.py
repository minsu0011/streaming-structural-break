"""Register four complete-background diagnostic summaries without changing fits."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, note


def main():
    path = BASE / 'runtime/TASK_QUEUE.json'
    previous = read_json(path)
    assert read_json(BASE / 'CROSSED_H_LENGTH_COMPARISON_LOCK.json')['status'] == 'FROZEN'
    script = 'scripts/next7/compare_crossed_H_lengths.py'
    tasks = {}
    for case in ('N0', 'N1', 'N2', 'N3'):
        tasks[f'support_partition_{case}_H5000'] = {
            'id': f'crossed_H_length_comparison_{case}', 'status': 'READY', 'script': script,
            'source_sha256': digest(ROOT / script), 'arguments': ['--case', case],
            'gates': ['PREFLIGHT_NEXT7.json', 'CROSSED_H_LENGTH_COMPARISON_LOCK.json',
                *[f'{family}/{case}/H_{length}/DONE.json' for length in (1000, 2048, 5000)
                  for family in ('conditional_statistics', 'support_partition')]],
            'hypothesis': 'FIXED_H_LENGTH_DIAGNOSTIC', 'new_scientific_candidates': 0}
    assert not ({item['id'] for item in previous} & {item['id'] for item in tasks.values()})
    added = list(tasks.values())
    ordered = []
    for item in previous:
        ordered.append(item)
        if item['id'] in tasks:
            ordered.append(tasks.pop(item['id']))
    assert not tasks and len(ordered) == len(previous) + 4
    save('H_LENGTH_COMPARISON_QUEUE_SUBMISSION.json', {'status': 'QUEUED', 'created_utc': utc_now(),
        'old_queue_sha256': digest(path), 'old_queue': previous, 'new_tasks': added,
        'capacity_unchanged': 2, 'only_complete_backgrounds_admitted': True})
    save('runtime/TASK_QUEUE.json', ordered, False)
    note('Matched H length comparisons queued', 'All three fixed contrasts in each background; no new fit, cutoff, or random sample', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

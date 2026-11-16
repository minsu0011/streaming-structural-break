"""Gate immutable summaries on complete cohorts, never partial result peeking."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, note
from src.next6_extension.synthetic import CASES


def main():
    path = BASE / 'runtime/TASK_QUEUE.json'
    previous = read_json(path)
    added = []
    after = {}
    def task(name, script, arguments, gates):
        result = {'id': name, 'status': 'READY', 'script': script, 'source_sha256': digest(ROOT / script),
            'arguments': arguments, 'gates': ['PREFLIGHT_NEXT7.json', *gates],
            'hypothesis': 'FIXED_RESULT_POSTPROCESSING', 'new_scientific_candidates': 0}
        added.append(result)
        return result
    for case in ('N0', 'N1', 'N2', 'N3'):
        for length in (1000, 2048, 5000):
            parent_id = f'conditional_{case}_H{length}'
            args = ['--case', case, '--H-length', str(length)]
            cohort = f'conditional_null/grid/{case}/H_{length}/DONE.json'
            statistics = f'conditional_statistics/{case}/H_{length}/DONE.json'
            items = []
            if not (BASE / statistics).exists():
                items.append(task(f'conditional_statistics_{case}_H{length}', 'scripts/next7/aggregate_conditional_grid.py',
                    args, [cohort, 'CROSSED_AGGREGATOR_IMPLEMENTATION_LOCK.json']))
            if not (BASE / f'support_partition/{case}/H_{length}/DONE.json').exists():
                items.append(task(f'support_partition_{case}_H{length}', 'scripts/next7/decompose_bootstrap_support.py',
                    args, [cohort, statistics, f'bootstrap_support/{case}/H_{length}/DONE.json',
                        'SUPPORT_PARTITION_IMPLEMENTATION_LOCK.json']))
            after[parent_id] = items
    for case in CASES:
        after['classifier_evaluation_' + case] = [task('classifier_statistics_' + case,
            'scripts/next7/summarize_classifier_stress.py', ['--case', case],
            [f'classifier_stress/evaluation/{case}/DONE.json', 'CLASSIFIER_UNCERTAINTY_IMPLEMENTATION_LOCK.json'])]
    after['classifier_calibration_bom_repair'] = [task('classifier_calibration_repair_closure',
        'scripts/next7/close_classifier_bom_repair.py', [], ['CLASSIFIER_CALIBRATION_FROZEN.json',
            'runtime/queue/classifier_calibration_bom_repair/TERMINAL.json'])]
    assert not ({i['id'] for i in previous} & {i['id'] for i in added})
    ordered = []
    for item in previous:
        ordered.append(item)
        ordered += after.pop(item['id'], [])
    assert not any(after.values())
    assert len(ordered) == len(previous) + len(added)
    save('POSTPROCESSING_QUEUE_SUBMISSION.json', {'status': 'QUEUED', 'created_utc': utc_now(),
        'old_queue_sha256': digest(path), 'old_queue': previous, 'new_tasks': added,
        'capacity_unchanged': 2, 'only_complete_cohorts_admitted': True})
    save('runtime/TASK_QUEUE.json', ordered, False)
    note('Complete-cohort summaries queued', f'{len(added)} exact and paired summaries or repair closure; no new candidate or random sample', 'AUTONOMOUS_MAINTENANCE')


if __name__ == '__main__':
    main()

"""Queue H-only reconstruction and complete-cohort reference diagnostics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, note


def main():
    path = BASE / 'runtime/TASK_QUEUE.json'
    previous = read_json(path)
    assert read_json(BASE / 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json')['status'] == 'FROZEN'
    script = 'scripts/next7/diagnose_normalizer_transport.py'
    shared = {'status': 'READY', 'script': script, 'source_sha256': digest(ROOT / script),
              'hypothesis': 'FIXED_H_REFERENCE_TRANSPORT_FORENSIC', 'new_scientific_candidates': 0}
    added = [{**shared, 'id': 'normalizer_transport_H_parameters', 'arguments': ['--prepare-H'],
        'gates': ['PREFLIGHT_NEXT7.json', 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json', 'BOOTSTRAP_SUPPORT_DONE.json']}]
    for case in ('N0', 'N1', 'N2', 'N3'):
        for length in (1000, 2048, 5000):
            added.append({**shared, 'id': f'normalizer_transport_correlation_{case}_H{length}',
                'arguments': ['--case', case, '--H-length', str(length)],
                'gates': ['PREFLIGHT_NEXT7.json', 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json',
                    'NORMALIZER_H_PARAMETERS_DONE.json', f'conditional_statistics/{case}/H_{length}/DONE.json']})
    assert not ({item['id'] for item in previous} & {item['id'] for item in added})
    # The scheduler reserves all live owners first. New short diagnostics can
    # use the next available slot without interrupting either running cohort.
    ordered = added + previous
    save('NORMALIZER_TRANSPORT_QUEUE_SUBMISSION.json', {'status': 'QUEUED', 'created_utc': utc_now(),
        'old_queue_sha256': digest(path), 'old_queue': previous, 'new_tasks': added,
        'capacity_unchanged': 2, 'live_work_not_interrupted': True,
        'new_scientific_candidates': 0, 'new_random_draws': 0})
    save('runtime/TASK_QUEUE.json', ordered, False)
    note('Normalizer transport forensic queued', 'Original H references only, then complete-cohort associations; existing models and thresholds fixed', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

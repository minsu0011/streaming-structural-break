"""Admit a BOM-only authority CSV decoding repair before any new calibration."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next7.common import BASE, read_json, save, digest, utc_now, note


def main():
    begun = read_json(BASE / 'audit/CLASSIFIER_BOM_REPAIR_STARTED.json')
    script = 'scripts/next7/run_classifier_stress.py'
    old_source = (BASE / 'audit/sources/CLASSIFIER_CALIBRATION_PRE_BOM_REPAIR.py').read_text(encoding='utf-8')
    new_source = (ROOT / script).read_text(encoding='utf-8')
    assert old_source.replace("(parent / 'MAXIMA.csv').open(newline='', encoding='utf-8')",
        "(parent / 'MAXIMA.csv').open(newline='', encoding='utf-8-sig')") == new_source
    assert not (BASE / 'classifier_stress/calibration').exists()
    assert not (BASE / 'classifier_stress/evaluation').exists()
    checks = []
    for fold in range(5):
        parent = ROOT / f'artifacts/next6_extension/calibration/fold_{fold}'
        receipt = read_json(parent / 'DONE.json')
        path = parent / 'MAXIMA.csv'
        assert path.read_bytes().startswith(b'\xef\xbb\xbf')
        with path.open(newline='', encoding='utf-8-sig') as f:
            records = list(csv.DictReader(f))
        assert len(records) == 2048
        expected = {(c, i, m) for c in ('N0', 'N1', 'N2', 'N3') for i in range(256) for m in ('M0', 'M3')}
        actual = {(r['background'], int(r['seed']), r['method']) for r in records}
        assert actual == expected
        for method in ('M0', 'M3'):
            maxima = np.array([float(r['maximum']) for r in records if r['method'] == method])
            assert np.isfinite(maxima).all() and len(maxima) == 1024
            assert np.sort(maxima)[973] == receipt['thresholds'][method]
        checks.append({'fold': fold, 'records': len(records), 'all_keys_present_once': True,
            'both_existing_cutoffs_reconstructed_exact': True, 'authority_file_sha256': digest(path)})
    implementation_path = BASE / 'CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json'
    assert digest(implementation_path) == begun['old_implementation_sha256']
    implementation = read_json(implementation_path)
    assert implementation['source_sha256'][script] == begun['old_source_sha256']
    implementation['source_sha256'][script] = digest(ROOT / script)
    implementation['engineering_repair'] = {'created_utc': utc_now(), 'previous_implementation_sha256': begun['old_implementation_sha256'],
        'previous_copy': 'audit/sources/CLASSIFIER_IMPLEMENTATION_PRE_BOM_REPAIR.json',
        'repair': 'UTF8 BOM decoding only, before any classifier calibration or new evaluation',
        'model_feature_cutoff_definitions_unchanged': True}
    assert all(digest(ROOT / name) == sha for name, sha in implementation['source_sha256'].items())
    save(implementation_path.name, implementation, False)
    queue_path = BASE / 'runtime/TASK_QUEUE.json'
    previous = read_json(queue_path)
    old_queue_sha = digest(queue_path)
    old_task = next(r for r in previous if r['id'] == 'classifier_calibration')
    assert read_json(BASE / 'runtime/queue/classifier_calibration/TERMINAL.json')['status'] == 'QUARANTINE'
    replacement = {**old_task, 'id': 'classifier_calibration_bom_repair', 'source_sha256': digest(ROOT / script),
        'gates': old_task['gates'] + ['CLASSIFIER_CALIBRATION_BOM_REPAIR_001.json'],
        'replaces_software_failure': 'classifier_calibration'}
    ordered = []
    for item in previous:
        updated = dict(item)
        if item['id'].startswith('classifier_evaluation_'):
            assert not (BASE / 'runtime/queue' / item['id']).exists()
            updated['source_sha256'] = digest(ROOT / script)
            updated['gates'] = item['gates'] + ['CLASSIFIER_CALIBRATION_BOM_REPAIR_001.json']
        ordered.append(updated)
        if item['id'] == 'classifier_calibration':
            ordered.append(replacement)
    save('CLASSIFIER_CALIBRATION_BOM_REPAIR_001.json', {'status': 'PASS', 'created_utc': utc_now(),
        **begun, 'completed_utc': utc_now(), 'checks': checks, 'new_source_sha256': digest(ROOT / script),
        'new_implementation_sha256': digest(implementation_path), 'old_queue_sha256': old_queue_sha,
        'old_queue': previous, 'replacement_task': replacement, 'source_change_exactly_BOM_decoder': True,
        'new_models': 0, 'new_cutoff_search': 0, 'scientific_outcome_released_before_repair': False,
        'classification': 'SOFTWARE_FILE_ENCODING_BEFORE_CALIBRATION', 'safe_repaired_retry_count': 1})
    save('runtime/queue/classifier_calibration/RESOLUTION.json', {'status': 'RESOLVED_SOFTWARE_ERROR',
        'created_utc': utc_now(), 'replacement_id': replacement['id'],
        'repair_receipt_sha256': digest(BASE / 'CLASSIFIER_CALIBRATION_BOM_REPAIR_001.json')})
    save('runtime/TASK_QUEUE.json', ordered, False)
    note('Classifier calibration BOM repair admitted', 'All five authority files reconstruct original cutoffs exactly; no calibration outcomes preceded repair', 'AUTONOMOUS_MAINTENANCE')


if __name__ == '__main__':
    main()

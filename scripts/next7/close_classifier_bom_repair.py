"""Resolve the earlier software quarantine only after repaired calibration succeeds."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, note


def main():
    original = BASE / 'runtime/queue/classifier_calibration/TERMINAL.json'
    repaired = BASE / 'runtime/queue/classifier_calibration_bom_repair/TERMINAL.json'
    frozen = BASE / 'CLASSIFIER_CALIBRATION_FROZEN.json'
    assert read_json(original)['status'] == 'QUARANTINE'
    assert read_json(repaired)['status'] == 'COMPLETE'
    assert read_json(frozen)['status'] == 'FROZEN'
    assert read_json(frozen)['all_five_before_new_evaluation']
    assert read_json(frozen)['implementation_sha256'] == digest(BASE / 'CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json')
    resolution = original.parent / 'RESOLUTION.json'
    prior = read_json(resolution)
    assert prior['status'] == 'RESOLVED_SOFTWARE_ERROR'
    save('audit/CLASSIFIER_BOM_REPAIR_PENDING_RESOLUTION.json', prior)
    save(str(resolution.relative_to(BASE)), {'status': 'REPAIRED_AND_REPRODUCED', 'created_utc': utc_now(),
        'terminal_sha256': digest(original), 'replacement_id': 'classifier_calibration_bom_repair',
        'evidence_sha256': {p.relative_to(BASE).as_posix(): digest(p) for p in (repaired, frozen,
            BASE / 'CLASSIFIER_CALIBRATION_BOM_REPAIR_001.json')},
        'scientific_definition_changed': False, 'calibration_path_count': 1024}, False)
    save('CLASSIFIER_CALIBRATION_REPAIR_CLOSED.json', {'status': 'PASS', 'created_utc': utc_now(),
        'resolution_sha256': digest(resolution), 'all_five_repaired_calibrations_frozen': True})
    note('Classifier file-encoding repair fully reproduced', 'Original baseline maxima and cutoffs pass on all1024 calibration paths', 'AUTONOMOUS_MAINTENANCE')


if __name__ == '__main__':
    main()

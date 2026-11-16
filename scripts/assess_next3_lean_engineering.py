"""Apply the fixed E003 engineering gates; final raw reproduction remains required."""
from pathlib import Path
import sys
import shutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from scripts.verify_next3_lean_shared_legacy_metadata import checked_implementation
from src.next3.lean_shared_runtime import source_hash


def main():
    core_sha = checked_implementation(); base = BASE/'engineering/lean_shared'
    paths = {'all_dev': base/'all_dev/RESULTS.json', 'oof': base/'heldout_metadata_corrected_v1/oof/RESULTS.json',
        'alternate': base/'heldout_metadata_corrected_v1/alternate/RESULTS.json',
        'cold': base/'cold/RESULTS.json', 'direct': base/'direct/RESULTS.json',
        'public_api': BASE/'lean_public_api_all_dev/RESULTS.json'}
    reports = {key: read_json(path) for key, path in paths.items()}
    assert all(record['status'] == 'PASS' for record in reports.values())
    for name in ('all_dev', 'oof', 'alternate'):
        assert reports[name]['identity']['runtime_sha256'] == source_hash()
        assert reports[name]['identity']['implementation_lock_sha256'] == core_sha
    assert reports['public_api']['identity']['variant'] == 'LEAN_PACKED_SHARED'
    assert reports['all_dev']['variants'][0]['prediction_fingerprint_sha256'] == reports['public_api']['output_fingerprint_sha256']
    cost_pass = reports['cold']['cold_cost_gate_pass'] and reports['direct']['direct_cost_gate_pass']
    current_path = ROOT/'configs/next3/LEAN_RUNTIME_SELECTION.json'; previous = read_json(current_path)
    assert previous['status'] == 'VALIDATION_ONLY' and previous['runtime_sha256'] == source_hash()
    backup = base/'SELECTION_VALIDATION_ONLY.json'; assert not backup.exists(); shutil.copyfile(current_path, backup)
    selected = {**previous, 'status': 'PROVISIONAL_ENGINEERING_ELIGIBLE' if cost_pass else 'REJECTED_COST',
        'reviewed_utc': utc_now(), 'core_implementation_lock_sha256': core_sha,
        'validation_harness_correction_sha256': digest(base/'VALIDATION_HARNESS_CORRECTION_002.json'),
        'proofs_sha256': {name: digest(path) for name, path in paths.items()},
        'median_postwarm_process_RSS_ratio_vs_E002': reports['cold']['median_RSS_ratio'],
        'median_first_none_ratio_vs_E002': reports['cold']['median_first_none_ratio'],
        'median_direct_10M_ratio_vs_E002': reports['direct']['median_direct_10M_ratio'],
        'actual_fresh_cold_trials': 6, 'actual_direct_10M_trials': 6,
        'scientific_quality_change': False, 'scientific_role': 'FROZEN_PERFORMANCE',
        'remaining_final_requirements': ['fresh_pinned_raw_training_reproduction', 'final_audit', 'verified_archive_replay']}
    atomic_json(base/'ENGINEERING_GATE_DECISION.json', selected, immutable=True)
    atomic_json(current_path, selected)
    progress('E003 fixed engineering decision recorded', selected['status']+'; final raw reproduction and audit remain', experiment='E003_LEAN_SHARED')
    print('NEXT3 LEAN ENGINEERING', selected['status'], flush=True)


if __name__ == '__main__': main()

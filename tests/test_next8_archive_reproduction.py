import copy
import pytest
from src.next8.archive_reproduction import validate_reproduction


def record_fixture():
    checks = [{'scope': 'outer_' + str(i // 66), 'method': 'M' + str(i % 6), 'case': str(i),
               'points': 1024, 'fresh_scalar_chunk_bitwise': True, 'pickle_resume_at73_bitwise': True,
               'future_suffix_prefix73_bitwise': True, 'only_H_and_current_scalar_given_to_state': True}
              for i in range(660)]
    canonical = [{'points': 1024, 'historical_NEXT7_fixture_bitwise_equal': True, 'public_infer_generator': True}
                 for _ in range(9)]
    fixture = {'status': 'FROZEN', 'runtime_manifest_sha256': 'a', 'implementation_lock_sha256': 'b',
               'synthetic_inputs': {}, 'checks': checks, 'canonical_public_entry_checks': canonical}
    record = {**copy.deepcopy(fixture), 'status': 'PASS', 'replay_new_process_fixture_bitwise': True,
              'original_workspace_access_blocked': True, 'new_fits': 0, 'new_quality_scores': 0,
              'competition_or_RP_cache_access': 0, 'denied_read_attempts': 0, 'research_paths': 660,
              'canonical_paths': 9, 'scalar_prediction_points': 669 * 1024}
    return record, fixture


def test_archive_receipt_requires_closed_original_access_and_actual_checks():
    r, f = record_fixture()
    assert validate_reproduction(r, f)['status'] == 'PASS'
    for key, value in [('original_workspace_access_blocked', False), ('denied_read_attempts', 1),
                       ('new_fits', 1), ('replay_new_process_fixture_bitwise', False)]:
        changed = copy.deepcopy(r); changed[key] = value
        with pytest.raises(AssertionError):
            validate_reproduction(changed, f)
    changed = copy.deepcopy(r)
    changed['checks'][0]['pickle_resume_at73_bitwise'] = False
    with pytest.raises(AssertionError):
        validate_reproduction(changed, f)


def test_repeated_case_or_different_fixture_cannot_replace_coverage():
    r, f = record_fixture()
    r['checks'][1] = copy.deepcopy(r['checks'][0])
    f['checks'] = copy.deepcopy(r['checks'])
    with pytest.raises(AssertionError):
        validate_reproduction(r, f)
    r, f = record_fixture()
    f['synthetic_inputs'] = {'changed_recipe': {}}
    with pytest.raises(AssertionError):
        validate_reproduction(r, f)

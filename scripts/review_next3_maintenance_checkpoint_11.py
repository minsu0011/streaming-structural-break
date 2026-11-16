"""Close chronology, copy-entry and long-horizon compatibility maintenance."""
from pathlib import Path
import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.bottlenecks import record_board
from src.next3.final_report_facts import maintenance_facts
from src.next3.final_validation import successful_receipt, tests_proof


def main():
    references = {
        'chronology': (BASE/'maintenance/registration_chronology/v1/RESULTS.json', 'PASS_RECORDED_REGISTRATION_CHRONOLOGY'),
        'copy_smoke': (BASE/'maintenance/raw_input_copy_smoke_v2_entry_first/RESULTS.json', 'PASS_ZERO_FIT_COPIED_INPUT_SMOKE'),
        'long_parity': (BASE/'maintenance/lean_long_horizon_parity_v1/RESULTS.json', 'PASS'),
        'horizon_logic': (BASE/'maintenance/horizon_calibration_logic_v1/RESULTS.json', 'PASS_EXPLANATORY_RANK_ALGEBRA')}
    records = {key: successful_receipt(path, statuses=(status,)) for key, (path, status) in references.items()}
    assert records['chronology']['event_chains_checked'] == 172 and records['chronology']['science_fits_accounted'] == 129
    assert records['copy_smoke']['copied_files'] == 244 and records['copy_smoke']['project_modules_loaded_from_copy'] == 74
    assert records['long_parity']['public_API_prediction_points'] == 34603008 and records['long_parity']['all_original_float32_predictions_exact']
    assert records['horizon_logic']['new_fits'] == records['copy_smoke']['new_fits'] == records['long_parity']['new_fits'] == 0
    tests = {}
    for name, count, classname, job in [('CHRONOLOGY_PRIMITIVE_TESTS', 22, 'tests.test_next3_chronology', 'M646_CHRONOLOGY_PRIMITIVE_TESTS'),
        ('CORE_REGISTRATION_SCHEMA_TESTS', 31, 'tests.test_next3_final_core_contract', 'M648_CORE_REGISTRATION_SCHEMA_TESTS')]:
        path = BASE/'maintenance'/(name+'.xml')
        state = read_json(BASE/'jobs'/job/'STATE.json')
        assert state['status'] == 'COMPLETE' and state['exit_code'] == 0
        tests[name] = tests_proof(path, minimum_count=count, required_classnames={classname},
            not_before_timestamp=datetime.datetime.fromisoformat(state['started_utc']).timestamp())
        assert tests[name]['tests'] == count
    maintenance = maintenance_facts(); assert maintenance['failed_child_attempts'] == 12
    board = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    for item in board:
        if item['id'] == 'B4':
            item.update(status='ACTIVE', next_test='Review remaining statistical/report assumptions. Recorded UTC chronology covers all 129 fits; fixed-horizon calibration is explicitly marginal and cannot be extended to long horizons without additional premises.')
        elif item['id'] == 'B8':
            item.update(status='ACTIVE', next_test='Prepare final source/environment/reproduction/distribution checks. Copied public entry-first admission smoke and 34,603,008 original long-horizon synthetic outputs pass; fresh final raw training, tests, 30M and actual ZIP remain required.')
        else:
            assert item['status'] != 'ACTIVE'; continue
        for path, _ in references.values(): item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
    record_board(board, '172 recorded UTC chains and all 129 fits accounted. Copied-input smoke corrected its import order after one preserved zero-fit failure; 244 files and 74 copied project modules pass. Frozen E003 public output matches all 34,603,008 original long-horizon synthetic points. Fixed-horizon calibration explanation verified; no new scientific result or fit.')
    atomic_json(BASE/'maintenance/CHECKPOINT_11_COMPLETION.json', {'status':'PASS', 'created_utc':utc_now(),
        'scientific_fits':129, 'new_fits_in_this_maintenance_batch':0, 'new_scientific_promotions':0,
        'recorded_UTC_chains':172, 'copied_input_files':244, 'copied_project_modules':74,
        'long_horizon_public_points_exact':34603008, 'targeted_tests':tests,
        'latest_whole_suite_historical_count':919, 'current_final_whole_suite_still_required':True,
        'maintenance_counts':maintenance, 'actual_final_audit_started':False,
        'fresh_final_raw_training_completed':False, 'actual_final_Desktop_archive_created':False,
        'source_sha256':digest(Path(__file__)),
        'evidence_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path,_ in references.values()}},immutable=True)
    progress('NEXT3 checkpoint 11','Chronology, copied-entry admission, complete original long-horizon parity and fixed-horizon interpretation verified; 12 failed child attempts preserved; no new fits')
    print('NEXT3 CHECKPOINT 11 PASS',flush=True)


if __name__ == '__main__': main()

"""Review prepared evidence contracts while leaving final audit and training pending."""
from pathlib import Path
import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.bottlenecks import record_board
from src.next3.final_validation import prepared_table_proof, tests_proof


def main():
    tables = prepared_table_proof(ROOT, BASE/'final_tables/prepared_v2')
    started = datetime.datetime.fromisoformat(read_json(BASE/'SESSION.json')['started_utc']).timestamp()
    tests = tests_proof(BASE/'maintenance/FINAL_AUDIT_PRIMITIVE_TESTS.xml', minimum_count=31,
        required_classnames={'tests.test_next3_final_validation', 'tests.test_next3_result_tables'},
        not_before_timestamp=started)
    official_path = BASE/'maintenance/official_contract_review_v1/RESULTS.json'
    official = read_json(official_path)
    assert official['status'] == 'PASS_LOCAL_CONTRACT_REVIEW'
    assert official['public_source_comparison_complete']
    assert all(record['byte_identical'] for record in official['current_official_source_comparison'].values())
    board = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    for item in board:
        if item['id'] == 'B4':
            item.update(status='ACTIVE', next_test='Complete final claim checks and fresh-audit evidence publication. Closed science tables distinguish 13 recipes, two frozen roles and two unexecuted menu items; inherited ALT/H are not new generalization evidence.')
            for path in (BASE/'final_tables/prepared_v2/RESULTS.json', BASE/'final_tables/prepared_v2/INPUT_RETENTION.json',
                         BASE/'maintenance/FINAL_AUDIT_PRIMITIVE_TESTS.xml'):
                item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
        elif item['id'] == 'B8':
            item.update(status='ACTIVE', next_test='Finish final reproduction/report execution paths and verify the actual final ZIP. Official runner/scorer bytes are unchanged; pinned-wheel cache is operational preparation, not a fourth runtime family.')
            item['evidence'].append({'path': official_path.relative_to(ROOT).as_posix(), 'sha256': digest(official_path)})
            path = BASE/'maintenance/pinned_wheel_cache_v1/RESULTS.json'
            if path.exists():
                cache = read_json(path); assert cache['status'] == 'PASS' and cache['pin_count'] == 63
                item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
        else:
            assert item['status'] != 'ACTIVE'
    record_board(board, 'Closed-table preview preserves all 213 historical ledger rows and adds 29 records accounting for 129 fits; 31 audit/table tests pass. Current official runner/scorer are byte-identical to frozen snapshots. No scientific or runtime family reopened.')
    result = {'status': 'PASS', 'created_utc': utc_now(), 'new_scientific_fits': 0, 'scientific_promotions': 0,
        'prepared_scientific_rows': tables['scientific_table_rows'], 'prepared_ledger_rows': tables['files']['EXPERIMENT_LEDGER.csv']['rows'],
        'targeted_tests': tests, 'last_full_test_suite_passes': 753,
        'full_suite_not_repeated_for_only_pure_report_helpers': True,
        'actual_final_audit_performed': False, 'raw_reproduction_performed': False,
        'pending': ['Final claim/audit/report builders', 'Scheduled fresh pinned raw reproduction',
            'Final fixed runtime verification', 'Final full tests', 'Verified actual Desktop ZIP and replay'],
        'sources_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in
            (BASE/'final_tables/prepared_v2/RESULTS.json', BASE/'final_tables/prepared_v2/INPUT_RETENTION.json', official_path)},
        'script_sha256': digest(Path(__file__))}
    atomic_json(BASE/'maintenance/CHECKPOINT_08_COMPLETION.json', result, immutable=True)
    progress('NEXT3 checkpoint 08', 'Scientific menu remains frozen; strict table/clock/XML evidence tested and actual final training/ZIP requirements remain pending')
    print('NEXT3 CHECKPOINT 08 PASS', tests['tests'], 'targeted tests; science unchanged', flush=True)


if __name__ == '__main__': main()

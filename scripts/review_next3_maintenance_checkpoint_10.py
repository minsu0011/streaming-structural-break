"""Review integrated delivery maintenance against a newly completed full suite."""
from pathlib import Path
import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.bottlenecks import record_board
from src.next3.final_source_contract import assert_source_snapshot
from src.next3.final_validation import tests_proof, successful_receipt


def main():
    inputs = read_json(BASE/'maintenance/FULL_SUITE_CHECKPOINT_07_INPUTS.json')
    assert_source_snapshot(ROOT, inputs['source_snapshot'])
    tests_path = BASE/'maintenance/FULL_TESTS_CHECKPOINT_7.xml'
    tests = tests_proof(tests_path, minimum_count=919,
        required_classnames={'tests.test_ts_auc', 'tests.test_scorer_official_checks',
            'tests.test_next3_final_core_contract', 'tests.test_next3_distribution_validation',
            'tests.test_next3_final_source_contract', 'tests.test_next3_claim_checks',
            'tests.test_next3_portable_report_contract'},
        not_before_timestamp=datetime.datetime.fromisoformat(inputs['created_utc']).timestamp())
    claims_path = BASE/'maintenance/report_previews/review_v1/claims_v2/RESULTS.json'
    claims = successful_receipt(claims_path, statuses=('PREVIEW_CLAIMS_PASS_NOT_FINAL_AUDIT',))
    assert claims['numeric_cross_checks'] == 245 and claims['numbered_Korean_answers'] == 44
    board = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    proof_paths = [tests_path, claims_path, ROOT/'docs/NEXT3_FINAL_AUDIT_EXECUTION_PROTOCOL.md']
    for item in board:
        if item['id'] == 'B4':
            item.update(status='ACTIVE', next_test='Review remaining evidence/schema and final report coverage. All 44 draft answers and 245 independent numeric joins pass; no new independent generalization evidence is claimed.')
        elif item['id'] == 'B8':
            item.update(status='ACTIVE', next_test='Review integrated raw reproduction, final source/test/API/core-audit and actual archive completion chain. Final execution remains time-gated and unrun; no fourth runtime family.')
        else:
            assert item['status'] != 'ACTIVE'; continue
        for path in proof_paths:
            item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
    record_board(board, f'Integrated maintenance full suite passed {tests["tests"]} cases against an unchanged captured source set. Korean preview has 44 answers and 245 numeric joins. Actual final audit, raw reproduction and Desktop ZIP are still pending behind the original time gate.')
    atomic_json(BASE/'maintenance/CHECKPOINT_10_COMPLETION.json', {'status':'PASS', 'created_utc':utc_now(),
        'full_tests': tests, 'current_source_set_unchanged_since_full_test_registration': True,
        'source_snapshot_sha256':digest(BASE/'maintenance/FULL_SUITE_CHECKPOINT_07_INPUTS.json'),
        'numeric_preview_claims':245, 'numbered_Korean_answers':44,
        'scientific_fits':129, 'new_scientific_promotions':0, 'new_runtime_families':0,
        'actual_final_audit_completed':False, 'fresh_final_raw_training_completed':False,
        'actual_final_Desktop_archive_created':False,
        'script_sha256':digest(Path(__file__)),
        'evidence_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path in proof_paths}},immutable=True)
    progress('NEXT3 checkpoint 10', str(tests['tests'])+' full-suite tests and 245 draft numeric joins passed; original scientific roles and final time boundaries preserved')
    print('NEXT3 CHECKPOINT 10 PASS',tests['tests'],'FULL TESTS',flush=True)


if __name__ == '__main__': main()

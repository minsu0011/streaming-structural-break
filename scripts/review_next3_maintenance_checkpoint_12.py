"""Record completed portable-copy and actual-tree validation with honest failures."""
from pathlib import Path
import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.bottlenecks import record_board
from src.next3.final_report_facts import maintenance_facts
from src.next3.final_validation import successful_receipt, tests_proof
from src.next3.maintenance import resource_snapshot


def main():
    paths = [BASE/'maintenance/long_horizon_copy_replay_v1/RESULTS.json',
        BASE/'maintenance/old_desktop_preservation_maintenance_v1/RESULTS.json',
        BASE/'maintenance/evidence_navigation_v1/RESULTS.json',
        BASE/'maintenance/FULL_SUITE_CHECKPOINT_08_RESULT.json',
        BASE/'maintenance/tree_structural_maintenance_v2/RESULTS.json']
    expected = ['PASS','PASS','PASS_EXISTING_EVIDENCE_NAVIGATION','PASS','PASS_TREE_STRUCTURE_AND_ARTIFICIAL_BOUNDARIES']
    records = [successful_receipt(path,statuses=(status,)) for path,status in zip(paths,expected)]
    assert records[0]['public_API_prediction_points'] == 34603008 and records[0]['copied_files'] == 249
    assert records[1]['file_count'] == 20 and records[2]['evidence_records'] == 25
    assert records[3]['tests']['tests'] == 971 and records[3]['source']['file_count'] == 892
    tree = records[4]
    assert tree['unique_model_nodes_checked'] == 5800 and tree['artificial_full_forest_margin_comparisons'] == 38732
    assert tree['new_fits'] == tree['new_quality_scores'] == tree['new_runtime_families'] == 0
    for name,sha in tree['source_sha256'].items(): assert digest(ROOT/name) == sha
    for name,sha in tree['files_sha256'].items(): assert digest(paths[-1].parent/name) == sha
    state = read_json(BASE/'jobs/M668_TREE_ORACLE_PRECISION_REGRESSION/STATE.json')
    assert state['status'] == 'COMPLETE' and state['exit_code'] == 0
    tests = tests_proof(BASE/'maintenance/TREE_CERTIFICATE_PRIMITIVE_TESTS_CORRECTED.xml',minimum_count=19,
        required_classnames={'tests.test_next3_tree_certificate'},
        not_before_timestamp=datetime.datetime.fromisoformat(state['started_utc']).timestamp())
    assert tests['tests'] == 19
    maintenance = maintenance_facts()
    assert maintenance['failed_child_attempts'] == 13 and maintenance['superseded_without_execution'] == 4
    items = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    for item in items:
        if item['id'] == 'B4':
            item.update(status='ACTIVE',next_test='Make existing frozen aggregate statistical evidence reviewable without real observations or saved point predictions. Preserve conditional inference scope, no new resampling or selection. Review final claim/source dependencies.')
        elif item['id'] == 'B8':
            item.update(status='ACTIVE',next_test='Maintain final reproduction/distribution readiness. Whole suite 971 passes; corrected actual-tree structural certificate covers 5800 nodes and 38732 artificial margins. Fresh raw models will receive the same certificate during the actual final audit.')
        else:
            assert item['status'] != 'ACTIVE'; continue
        item['evidence'].extend({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)} for path in paths)
    board = record_board(items,'Optional raw-free copied long replay, prior Desktop byte preservation, 25-record evidence map and full 971-test suite pass. Actual tree validation exposed a Python oracle precision failure; original attempt preserved, 19 corrected primitive tests and 5800-node/38732-margin audit pass. No new scientific fits, qualities or runtime family.')
    atomic_json(BASE/'maintenance/CHECKPOINT_12_COMPLETION.json',{'status':'PASS','created_utc':utc_now(),
        'scientific_fits':129,'new_fits_in_this_maintenance_batch':0,'new_scientific_promotions':0,
        'latest_executed_whole_suite':records[3]['tests'],'current_additional_tree_tests':tests,
        'whole_suite_scope':'971 was executed before the 19 new tree-certificate cases; final current collection and full suite remain required.',
        'current_tree_node_checks':5800,'artificial_tree_margin_comparisons':38732,
        'maintenance_counts':maintenance,'resources':resource_snapshot(),
        'actual_final_audit_started':False,'fresh_final_raw_training_completed':False,
        'actual_final_Desktop_archive_created':False,'source_sha256':digest(Path(__file__)),
        'evidence_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path in paths},
        'bottleneck_review_utc':board['created_utc']},immutable=True)
    progress('NEXT3 checkpoint 12','Portable-copy/tree proof and old Desktop preservation complete; 13 failed children honestly retained; B4/B8 remain active; no new science')
    print('NEXT3 CHECKPOINT 12 PASS',flush=True)


if __name__ == '__main__': main()

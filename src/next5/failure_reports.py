"""Final classification from preserved original scientific and execution records."""
from collections import Counter
import csv
from src.next5.artifacts import ROOT, BASE, digest


def completed_failure_rows(read):
    original_dir = BASE / 'failure_classification/research_v1'
    original = read(original_dir / 'RESULTS.json', 'PASS')
    original_table = original_dir / 'FAILURE_CLASSIFICATION.csv'
    assert digest(original_table) == original['table_sha256']
    with original_table.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    for row in rows:
        assert digest(ROOT / row['source_file']) == row['source_sha256']
        read(ROOT / row['source_file'])
        row['actual_failed_execution'] = False
        if row['record_type'] == 'infrastructure':
            resolution = read(BASE / 'failure_resolutions' / (row['id'] + '.json'), 'RESOLVED_WITH_PRESERVED_FAILURE')
            assert resolution['quarantined_state_sha256'] == row['source_sha256']
            row['actual_failed_execution'] = resolution['actual_failed_execution']
            row['successful_recovery_verified'] = True
    repair_dir = BASE / 'maintenance/READONLY_IMPORT_REPAIR'
    repair = read(repair_dir / 'RESOLVED.json', 'PASS')
    repair_table = repair_dir / 'FAILURE_CLASSIFICATION.csv'
    assert digest(repair_table) == repair['table_sha256']
    with repair_table.open(encoding='utf-8-sig', newline='') as stream:
        api_rows = list(csv.DictReader(stream))
    assert len(api_rows) == 7
    for row in api_rows:
        resolution = read(BASE / 'failure_resolutions' / (row['id'] + '.json'), 'RESOLVED_WITH_PRESERVED_FAILURE')
        state_path = BASE / 'jobs' / row['id'] / 'STATE.json'
        state = read(state_path, 'QUARANTINED')
        assert digest(state_path) == resolution['quarantined_state_sha256'] == row['quarantined_state_sha256']
        assert row['replacement_status'] == 'COMPLETE'
        rows.append({'record_type': 'API_import_infrastructure', 'id': row['id'],
            'classification': row['classification'], 'execution_status': state['status'],
            'resolution': resolution['resolution'],
            'source_file': state_path.relative_to(ROOT).as_posix(), 'source_sha256': digest(state_path),
            'actual_failed_execution': resolution['actual_failed_execution'],
            'successful_recovery_verified': True, 'replacement_job': row['replacement_job'],
            'verified_API_result_sha256': row['verified_API_result_sha256']})
    native_path = BASE / 'maintenance/NATIVE_REVIEW_PUBLICATION_REPAIR/RESOLVED.json'
    native = read(native_path, 'PASS')
    assert native['actual_failed_execution'] and native['successful_recovery_verified']
    assert native['all_prior_scientific_row_values_and_best_result_preserved']
    assert native['Python_processes_started'] == native['new_fits'] == native['new_predictions'] == native['new_random_draws'] == 0
    for name, sha in native['inputs_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(ROOT / native['replacement_review']) == native['replacement_review_sha256']
    rows.append({'record_type': 'native_metadata_publication', 'id': native['id'],
        'classification': native['classification'], 'execution_status': 'FAILED_THEN_RECOVERED',
        'resolution': 'Explicit PowerShell NullString backup argument; native fixture and versioned review 013 verified; partial review 012 preserved',
        'source_file': native_path.relative_to(ROOT).as_posix(), 'source_sha256': digest(native_path),
        'actual_failed_execution': True, 'successful_recovery_verified': True,
        'replacement_review': native['replacement_review']})
    counts = dict(Counter(row['classification'] for row in rows))
    assert counts == {'BUG': 4, 'QUARANTINED': 8, 'SCREEN_FAIL': 1, 'FULL_FAIL': 19}
    assert len({r['id'] for r in rows}) == len(rows) == 32
    assert sum(r['actual_failed_execution'] for r in rows) == 4
    return rows, {'status': 'PASS_COMPLETE_FAILURE_CLASSIFICATION', 'records': 32,
        'classification_counts': counts, 'actual_execution_failures': 4,
        'supervised_Python_execution_failures': 3, 'native_metadata_publication_failures': 1,
        'unstarted_superseded_jobs': 8, 'scientific_screen_failures': 1,
        'scientific_promotion_failures': 19, 'all_11_preserved_quarantines_resolved': True,
        'new_fits': 0, 'source_sha256': digest(ROOT / 'src/next5/failure_reports.py'),
        'scope': 'Final union of the original 24 records, seven reviewed API import records and one native metadata publication failure; inspection-command and development-only edits are separate maintenance history.'}

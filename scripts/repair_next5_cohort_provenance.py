"""Preserve an unstarted queue revision fixing omitted alarm cohort identity."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.scheduler import register_job
from src.next5.cohort_provenance import annotate_paired_alarms, rows


def main():
    identifier = 'SYN_ALL_FITTED_SUMMARY'
    queue = BASE / 'queue' / (identifier + '.json')
    assert not (BASE / 'jobs' / identifier / 'STATE.json').exists(), 'Only an unstarted summary can receive this metadata repair'
    original = read_json(queue)
    assert original['argv'] == ['scripts/summarize_next5_fitted_stress.py']
    out = BASE / 'maintenance/COHORT_PROVENANCE_REPAIR'
    out.mkdir(parents=True, exist_ok=False)
    before = out / 'QUEUE_BEFORE.json'
    before.write_bytes(queue.read_bytes())
    sources = ('src/next5/cohort_provenance.py', 'scripts/summarize_next5_fitted_stress_with_provenance.py',
        'scripts/repair_next5_cohort_provenance.py', 'tests/test_next5_cohort_provenance.py',
        'scripts/summarize_next5_fitted_stress.py', 'src/next5/fitted_fold_summary.py',
        'scripts/analyze_next5_synthetic.py', 'src/next5/synthetic.py')
    evidence = []
    for path in sorted((BASE / 'synthetic/candidates').glob('*/full/fold_0/ANALYSIS/PAIRED_ALARMS.csv')):
        original_rows = rows(path)
        annotated = annotate_paired_alarms(path, original_rows)
        evidence.append({'table': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
            'rows': len(original_rows), 'annotated_rows': len(annotated), 'all_numeric_and_original_cells_unchanged': True})
    assert len(evidence) == 19
    atomic_json(out / 'POLICY_LOCK.json', {'created_utc': utc_now(), 'status': 'FROZEN_BEFORE_REPAIRED_AGGREGATION',
        'classification': 'BUG_PREVENTED_BEFORE_SUMMARY_EXECUTION',
        'finding': 'PAIRED_ALARMS omits cohort kind, so N0-N3 horizon1024 have two distinct populations under one old group key',
        'repair': 'Recover exact kind from frozen loop order, separate scalar source tables and hash-bound cohort receipts; append cohort_kind to fold aggregation keys',
        'original_frozen_sources_and_per_fold_results_modified': False,
        'original_queue_sha256': digest(before), 'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'all_19_completed_fold0_schemas_verified': evidence,
        'sources_sha256': {name: digest(ROOT / name) for name in sources},
        'no_promotion_rule_or_candidate_change': True}, immutable=True)
    register_job({'id': 'COHORT_PROVENANCE_CONTRACTS', 'kind': 'validation', 'priority': 35,
        'argv': ['-m', 'pytest', 'tests/test_next5_cohort_provenance.py', '-q',
            '--junitxml=artifacts/next5/maintenance/COHORT_PROVENANCE_REPAIR/CONTRACTS.xml'],
        'description': 'Verify actual fixed alarm rows retain values and distinguish null populations; reject reordered attribution',
        'retry_safe': False})
    repaired = {**original, 'argv': ['scripts/summarize_next5_fitted_stress_with_provenance.py'],
        'depends_on_success': [*original['depends_on_success'], 'COHORT_PROVENANCE_CONTRACTS'],
        'execution_repair_utc': utc_now(), 'original_queue_sha256': digest(before),
        'execution_repair_policy_sha256': digest(out / 'POLICY_LOCK.json'),
        'description': original['description'] + '; explicit cohort metadata repair, original statistics unchanged'}
    assert not (BASE / 'jobs' / identifier / 'STATE.json').exists()
    atomic_json(queue, repaired)
    atomic_json(out / 'REPAIR.json', {'created_utc': utc_now(), 'status': 'REPAIRED_UNSTARTED_METADATA_AGGREGATION',
        'job': identifier, 'original_queue_sha256': digest(before), 'repaired_queue_sha256': digest(queue),
        'policy_sha256': digest(out / 'POLICY_LOCK.json'), 'new_success_dependency': 'COHORT_PROVENANCE_CONTRACTS',
        'failed_execution_created': False, 'original_science_sources_changed': False,
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0}, immutable=True)
    progress('Unstarted summary metadata repair registered', '19 original folded tables checked; exact cells preserved and distinct null cohorts separated')
    print('NEXT5 COHORT PROVENANCE REPAIR: 19 original fold tables verified; queue revision preserved', flush=True)


if __name__ == '__main__':
    main()

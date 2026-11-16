"""Final byte-level audit of every frozen synthetic cohort and its lineage."""
from pathlib import Path
import argparse
import csv
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.synthetic import cohort_menu, cohort_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+', args.tag)
    assert read_json(BASE / 'SESSION.json').get('audit_started_utc')
    menu = read_json(BASE / 'synthetic/MENU_LOCK.json')
    execution = read_json(BASE / 'synthetic/EXECUTION_LOCK.json')
    execution_sha = digest(BASE / 'synthetic/EXECUTION_LOCK.json')
    for name, sha in execution['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json') == menu['original_generator_policy_sha256']
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    cohorts = cohort_menu(menu, lock)
    assert len(cohorts) == 34
    coverage = read_json(BASE / 'synthetic/FITTED_COVERAGE_LOCK.json')['fitted_artifacts']
    assert len(coverage) == 97
    references, rows, files = {}, [], {}
    out = BASE / 'final_audit' / args.tag / 'synthetic_receipts'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'source_sha256': digest(Path(__file__)),
        'prediction_arrays_decoded': False, 'scope': 'All 97 candidate and seven matched-reference fitted-fold cohort files'}, immutable=True)

    def verify(directory, spec, scope, fold, candidate=None):
        path = directory / 'RESULTS.json'
        receipt = read_json(path)
        assert receipt['status'] == 'COMPLETE'
        identity = receipt['identity']
        assert identity['cohort'] == spec and identity['scope'] == scope and identity['fold'] == fold
        assert identity['execution_sha256'] == execution_sha and identity['menu_sha256'] == digest(BASE / 'synthetic/MENU_LOCK.json')
        assert receipt['new_fits'] == receipt['real_Shadow_rows'] == receipt['LOCAL_SEAL_rows'] == 0
        if candidate is not None:
            assert identity['candidate'] == candidate
            ref = references[(scope, fold, cohort_key(spec))]
            assert receipt['input_sha256'] == ref['input_sha256']
            assert identity['reference_prediction_sha256'] == ref['prediction_file_sha256']
            original_fit = BASE / 'experiments' / candidate / scope / ('fold_' + str(fold)) / 'RESULTS.json'
            fitted = read_json(original_fit)
            assert identity['model_sources']['model_sha256'] == fitted['model_sha256']
            assert identity['model_sources']['result_sha256'] == digest(original_fit)
            assert identity['model_sources']['registration_sha256'] == fitted['registration_sha256']
        else:
            for kind, proof in identity['model_sources'].items():
                original_fit = BASE / 'references/fits/PRIMARY' / scope / kind / ('fold_' + str(fold)) / 'RESULTS.json'
                fitted = read_json(original_fit)
                assert proof['model_sha256'] == fitted['model_sha256'] and proof['result_sha256'] == digest(original_fit)
        prediction_path = (ROOT / receipt['prediction_file']).resolve()
        assert prediction_path.is_relative_to(directory.resolve()) and prediction_path.name == 'PREDICTIONS.npz'
        assert digest(prediction_path) == receipt['prediction_file_sha256']
        assert receipt['horizon'] == spec['horizon'] and receipt['curves'] == spec['seeds'] * (2 if spec['paired'] else 1)
        files[receipt['prediction_file']] = receipt['prediction_file_sha256']
        rows.append({'candidate': candidate or 'MATCHED_REFERENCES', 'scope': scope, 'fold': fold,
            'cohort': cohort_key(spec), 'curves': receipt['curves'], 'horizon': receipt['horizon'],
            'input_sha256': receipt['input_sha256'], 'prediction_file': receipt['prediction_file'],
            'prediction_file_sha256': receipt['prediction_file_sha256'], 'bytes': prediction_path.stat().st_size,
            'cohort_receipt_sha256': digest(path)})
        if len(rows) % 10 == 0:
            progress('Complete synthetic cache byte audit', str(len(rows)) + '/3536 cohort files', phase='FINAL_AUDIT')
            print('NEXT5 SYNTHETIC BYTE AUDIT', len(rows), flush=True)
        return receipt

    for scope, folds in (('full', range(5)), ('screen', (0, 4))):
        for fold in folds:
            for spec in cohorts:
                key = cohort_key(spec)
                directory = BASE / 'synthetic/references' / scope / ('fold_' + str(fold)) / key
                references[(scope, fold, key)] = verify(directory, spec, scope, fold)
    for record in coverage:
        cid, scope, fold = record['candidate'], record['scope'], record['fold']
        for spec in cohorts:
            directory = BASE / 'synthetic/candidates' / cid / scope / ('fold_' + str(fold)) / cohort_key(spec)
            verify(directory, spec, scope, fold, cid)
    assert len(references) == 7 * 34 and len(rows) == 104 * 34 == 3536
    assert len(files) == len(rows)
    with (out / 'COHORT_BYTE_AUDIT.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'candidate_artifacts': 97, 'matched_reference_artifacts': 7, 'cohorts_per_artifact': 34,
        'cohort_files': len(rows), 'bytes_verified': sum(r['bytes'] for r in rows),
        'all_prediction_file_sha256_verified': True, 'all_candidate_reference_inputs_matched': True,
        'all_fitted_model_lineage_matched': True, 'prediction_arrays_decoded': False,
        'new_fits': 0, 'new_draws': 0, 'new_predictions': 0, 'real_data_read': False,
        'table_sha256': digest(out / 'COHORT_BYTE_AUDIT.csv'),
        'source_sha256': digest(Path(__file__)), 'execution_lock_sha256': execution_sha}, immutable=True)
    progress('All synthetic receipt bytes verified', '3536 immutable cohort files and exact fitted-model/reference lineage', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()

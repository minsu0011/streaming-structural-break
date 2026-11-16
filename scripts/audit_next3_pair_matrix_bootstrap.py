"""Reconstruct the original conditional bootstrap through independent pair matrices."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import psutil
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.evaluation import load_full_oof
from src.next3.pair_matrix_audit import dense_series, enumerate_pair_matrices, evaluate_pair_matrices
from src.next.data import prepare_dev
from src.validation.group_bootstrap import draw_group_multiplicities

REFERENCE = 'BL2_C205_DEPENDENCE_80'


def existing_fold_vector(candidate, fold):
    directory = ROOT/'artifacts/next2/forensic'/candidate if candidate == REFERENCE else BASE/'forensic'/candidate
    receipt_path = directory/'RESULTS.json'
    if not receipt_path.is_file(): return None, None
    receipt = read_json(receipt_path); assert receipt['status'] == 'PASS'
    name = f'fold_{fold}_all_time.npy'
    assert digest(directory/name) == receipt['files_sha256'][name]
    values = np.load(directory/name, allow_pickle=False); assert values.shape == (2001,)
    return values, {'path': (directory/name).relative_to(ROOT).as_posix(), 'sha256': digest(directory/name)}


def main():
    set_num_threads(2)
    registration = BASE/'hypotheses/D007_PAIR_MATRIX_BOOTSTRAP_AUDIT.json'
    hypothesis = read_json(registration)
    original = BASE/'selection_uncertainty'
    assert digest(original/'POLICY.json') == hypothesis['pool_policy_sha256']
    assert digest(original/'RESULTS.json') == hypothesis['existing_results_sha256']
    source_result = read_json(original/'RESULTS.json'); assert source_result['status'] == 'PASS'
    for name, sha in source_result['files_sha256'].items(): assert digest(original/name) == sha
    unit = BASE/'maintenance/PAIR_MATRIX_UNIT_TESTS.xml'
    suites = ET.parse(unit).getroot().findall('.//testsuite')
    assert suites and sum(int(s.attrib['tests']) for s in suites) == 15
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) ==
               int(s.attrib.get('skipped', 0)) == 0 for s in suites)
    implementation_path = BASE/'pair_matrix_audit/IMPLEMENTATION_LOCK.json'
    implementation = read_json(implementation_path)
    for name, sha in implementation['source_sha256'].items(): assert digest(ROOT/name) == sha
    candidates = read_json(original/'POLICY.json')['candidates']; assert len(candidates) == 12
    seed_base = read_json(BASE/'hypotheses/D002_JOINT_CONDITIONAL_UNCERTAINTY.json')['seed_base']
    with np.load(original/'AGGREGATE_CONTRAST_DRAWS.npz', allow_pickle=False) as arrays:
        expected_contrasts = arrays['contrasts'].copy()
        np.testing.assert_array_equal(arrays['candidates'], candidates)
        expected_maximum = arrays['max_abs_standardized_error'].copy()
    out = BASE/'pair_matrix_audit/run_v1'; out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'hypothesis_sha256': digest(registration),
        'implementation_lock_sha256': digest(implementation_path), 'unit_report_sha256': digest(unit),
        'candidate_order': [REFERENCE, *candidates], 'original_seed_base': seed_base,
        'replicates': 2000, 'tolerance_absolute': 1e-14, 'new_fits': 0,
        'matrix_formula': '(w.T @ integer_twice_wins @ w) / (2 * w.T @ integer_pairs @ w)',
        'existing_rank_kernel_used_for_reconstruction': False, 'matrix_product_chunk_size': 64}, immutable=True)
    began = time.perf_counter(); data = prepare_dev(ROOT); meta = data.rows()
    common_folds = []
    for fold in range(5):
        rows = np.flatnonzero(meta['fold'] == fold)
        code, counts = draw_group_multiplicities(meta['dataset_id'][rows], 2000, seed_base+fold)
        assert np.all(counts.sum(axis=1) == counts.shape[1]) and np.all(counts[0] == 1)
        common_folds.append((rows, code, counts))
    scores = {}; records = []; sources = {}; peak_rss = psutil.Process().memory_info().rss
    compared_existing = 0; max_existing_difference = 0.; total_pair_time_comparisons = 0
    for candidate in [REFERENCE, *candidates]:
        _, receipt, current_meta, prediction, source = load_full_oof(data, candidate)
        for name in meta: np.testing.assert_array_equal(meta[name], current_meta[name])
        sources[candidate] = source; fold_vectors = []
        for fold, (rows, code, counts) in enumerate(common_folds):
            tick = time.perf_counter()
            print('NEXT3 PAIR MATRIX START', candidate, fold, flush=True)
            dense, lengths, onset = dense_series(meta['target'][rows], prediction[rows], meta['time_online'][rows], code)
            numerator, denominator = enumerate_pair_matrices(dense, lengths, onset)
            comparisons = int(denominator.sum(dtype=np.int64)); total_pair_time_comparisons += comparisons
            assert comparisons > 0 and not np.diag(denominator).any()
            values = evaluate_pair_matrices(numerator, denominator, counts, chunk_size=64)
            assert values.shape == (2001,) and np.isfinite(values).all()
            expected, binding = existing_fold_vector(candidate, fold)
            difference = None
            if expected is not None:
                difference = float(np.max(np.abs(values-expected)))
                np.testing.assert_allclose(values, expected, rtol=0, atol=1e-14)
                compared_existing += len(values); max_existing_difference = max(max_existing_difference, difference)
            if candidate != REFERENCE:
                np.testing.assert_allclose(values[0], receipt['fold_scores'][fold], rtol=0, atol=1e-14)
            memory = psutil.Process().memory_info().rss; peak_rss = max(peak_rss, memory)
            records.append({'candidate': candidate, 'fold': fold, 'series': len(lengths),
                'online_rows': len(rows), 'ordered_positive_negative_time_pairs': comparisons,
                'observed_score': float(values[0]), 'replicates_including_observed': len(values),
                'existing_fold_array_compared': expected is not None, 'existing_fold_max_abs_difference': difference,
                'pair_matrix_bytes': numerator.nbytes+denominator.nbytes, 'process_RSS_bytes_after_fold': memory,
                'elapsed_seconds': time.perf_counter()-tick, 'existing_vector_binding': str(binding) if binding else ''})
            fold_vectors.append(values)
            del dense, numerator, denominator
            progress('D007 independent pair-matrix reconstruction',
                     f'{candidate} fold {fold}: 2001 fixed draws reconstructed; existing differences {difference}',
                     experiment='D007_PAIR_MATRIX_BOOTSTRAP_AUDIT')
            print('NEXT3 PAIR MATRIX PASS', candidate, fold, round(time.perf_counter()-tick, 3), flush=True)
        scores[candidate] = np.asarray(fold_vectors).T
        pd.DataFrame(records).to_csv(out/'FOLD_AUDIT_PROGRESS.csv', index=False)
        if candidate != REFERENCE:
            actual = scores[candidate].mean(axis=1)-scores[REFERENCE].mean(axis=1)
            column = candidates.index(candidate)
            np.testing.assert_allclose(actual, expected_contrasts[:, column], rtol=0, atol=1e-14)
    contrasts = np.column_stack([scores[name].mean(axis=1)-scores[REFERENCE].mean(axis=1) for name in candidates])
    max_contrast_difference = float(np.max(np.abs(contrasts-expected_contrasts)))
    np.testing.assert_allclose(contrasts, expected_contrasts, rtol=0, atol=1e-14)
    se = contrasts[1:].std(axis=0, ddof=1)
    maximum = np.max(np.abs((contrasts[1:]-contrasts[0])/se), axis=1)
    np.testing.assert_allclose(maximum, expected_maximum, rtol=0, atol=1e-10)
    critical = float(np.quantile(maximum, .95))
    np.testing.assert_allclose(critical, source_result['critical_value'], rtol=0, atol=1e-10)
    expected_table = pd.read_csv(original/'CONDITIONAL_POOL_ENVELOPE.csv', float_precision='round_trip').set_index('candidate')
    summaries = []
    for i, candidate in enumerate(candidates):
        vector = contrasts[:, i]; row = expected_table.loc[candidate]
        lower, upper = np.quantile(vector[1:], [.025, .975]); point = float(vector[0])
        bounds = [point-critical*se[i], point+critical*se[i]]
        np.testing.assert_allclose([point, lower, upper, se[i], *bounds],
            [row.estimate_delta, row.conditional_percentile_ci_2_5, row.conditional_percentile_ci_97_5,
             row.conditional_standard_error, row.joint_conditional_envelope_lower, row.joint_conditional_envelope_upper],
            rtol=0, atol=1e-14)
        summaries.append({'candidate': candidate, 'observed_mean_delta': point,
            'conditional_ci_low': float(lower), 'conditional_ci_high': float(upper),
            'probability_delta_positive': float(np.mean(vector[1:] > 0)),
            'joint_conditional_envelope_lower': float(bounds[0]), 'joint_conditional_envelope_upper': float(bounds[1]),
            'maximum_contrast_abs_difference': float(np.abs(vector-expected_contrasts[:, i]).max()),
            'selection_adjusted': False, 'independent_validation_of_generalization': False})
    pd.DataFrame(records).to_csv(out/'FOLD_AUDIT.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'CONDITIONAL_SUMMARY_RECONSTRUCTED.csv', index=False)
    np.savez_compressed(out/'RECONSTRUCTED_AGGREGATE_SCORES.npz', candidates=np.asarray([REFERENCE, *candidates]),
                        scores=np.stack([scores[name] for name in [REFERENCE, *candidates]]), contrasts=contrasts)
    for name, sha in implementation['source_sha256'].items(): assert digest(ROOT/name) == sha
    data.guard.check_state()
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'policy_sha256': digest(out/'POLICY.json'), 'candidates_including_reference': len(scores),
        'folds_reconstructed': len(records), 'total_bootstrap_scores_including_observed': len(records)*2001,
        'scores_compared_to_existing_fold_arrays': compared_existing,
        'max_existing_fold_abs_difference': max_existing_difference,
        'contrast_entries_compared': int(contrasts.size), 'max_contrast_abs_difference': max_contrast_difference,
        'reconstructed_joint_critical_value': critical, 'ordered_point_pair_comparisons': total_pair_time_comparisons,
        'peak_sampled_process_RSS_bytes': peak_rss, 'elapsed_seconds': time.perf_counter()-began,
        'new_fits': 0, 'new_scientific_candidates': 0, 'new_bootstrap_seed_choices': 0, 'seal_rows': 0,
        'selection_adjusted': False, 'sources': sources,
        'limitations': 'Independent numerical algorithm, not independent validation data. Existing adaptive DEV and fixed-fitted-pool conditional limitations remain.',
        'files_sha256': {name: digest(out/name) for name in ('FOLD_AUDIT.csv', 'CONDITIONAL_SUMMARY_RECONSTRUCTED.csv', 'RECONSTRUCTED_AGGREGATE_SCORES.npz')}}, immutable=True)
    print('NEXT3 INDEPENDENT PAIR MATRIX AUDIT PASS', len(records), max_contrast_difference, flush=True)


if __name__ == '__main__':
    main()

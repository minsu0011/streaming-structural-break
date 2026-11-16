"""Seven registered legacy substates, using only the unchanged original O1 cohort."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import time
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.state_transfer import arrays, clone_arguments, trace_components, trace_scores, transferred_arguments
from src.next3.legacy_state_transfer import LEGACY_BLOCKS, transfer_legacy_arguments
from src.next2.canonical_model import load
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next.fused_runtime import unique_array_bytes
from src.next2.counterexamples import policy, generate
from src.validation.group_bootstrap import draw_group_multiplicities
from scripts.forensic_next3_outlier_state_transfer import checked_inputs, evaluate_windows, ANCHORS, DIRECTIONS


def main():
    lock_path = BASE/'legacy_state_transfer/IMPLEMENTATION_LOCK.json'
    implementation = read_json(lock_path)
    for name, sha in implementation['source_sha256'].items(): assert digest(ROOT/name) == sha
    unit = BASE/'maintenance/LEGACY_TRANSFER_UNIT_TESTS.xml'
    suites = ET.parse(unit).getroot().findall('.//testsuite')
    assert suites and sum(int(s.attrib['tests']) for s in suites) == 19
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) ==
               int(s.attrib.get('skipped', 0)) == 0 for s in suites)
    _, expected, sources, thresholds = checked_inputs()
    original = BASE/'outlier_state_transfer/run_v1'
    old_result = read_json(original/'RESULTS.json'); assert old_result['status'] == 'PASS'
    for name, sha in old_result['files_sha256'].items(): assert digest(original/name) == sha
    old_curves = pd.read_csv(original/'MEAN_COUNTERFACTUAL_CURVES.csv', float_precision='round_trip')
    old_windows = pd.read_csv(original/'WINDOW_COUNTERFACTUAL_INTERVALS.csv', float_precision='round_trip')
    hypothesis_path = BASE/'hypotheses/D006_LEGACY_STATE_REFINEMENT.json'; hyp = read_json(hypothesis_path)
    out = BASE/'legacy_state_transfer/run_v1'; out.mkdir(parents=True, exist_ok=False)
    model_path = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    expected_model_sha = read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256'][model_path.relative_to(ROOT).as_posix()]
    assert digest(model_path) == expected_model_sha
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'hypothesis_sha256': digest(hypothesis_path),
        'implementation_lock_sha256': digest(lock_path), 'unit_report_sha256': digest(unit),
        'original_D005_receipt_sha256': digest(original/'RESULTS.json'), 'original_source_bindings': sources,
        'model_sha256': expected_model_sha, 'legacy_blocks': list(LEGACY_BLOCKS),
        'anchors': ANCHORS, 'directions': DIRECTIONS, 'thresholds': thresholds,
        'bootstrap_seed': hyp['bootstrap_seed'], 'bootstrap_replicates': hyp['bootstrap_replicates'],
        'new_fits': 0, 'new_candidates': 0}, immutable=True)
    suite = policy(); assert suite['evaluation_series_per_scenario'] == 512 and suite['online_length'] == 1024
    selected = [i for i in range(512) if suite['tau_cycle'][i%3] <= 256]
    assert len(selected) == 342
    local = {i: j for j, i in enumerate(selected)}
    cohorts = {a: 'ALL_512_COMMON_OFFSETS_1_511' if a < 511 else 'ORIGINAL_ONSET_128_256_ONLY_342' for a in ANCHORS}
    control_arrays = {a: np.empty((342 if a == 511 else 512, 256 if a == 511 else 511-a), dtype=np.float32) for a in ANCHORS}
    outlier_arrays = {a: np.empty_like(value) for a, value in control_arrays.items()}
    hybrids = {(a, b, d): np.empty_like(control_arrays[a]) for a in ANCHORS for b in LEGACY_BLOCKS for d in DIRECTIONS}
    coarse_arrays = {(a, d): np.empty_like(control_arrays[a]) for a in ANCHORS for d in DIRECTIONS}
    engine = PreparedSharedBlendPredictor(load(model_path), resident=False)
    compared = clone_checks = fine_coarse_checks = hybrid_points = 0
    began = time.perf_counter(); frozen_tree_hash = None
    for i in range(512):
        hc, oc, tc = generate('O1', i, control=True, lock=suite)
        ho, oo, to = generate('O1', i, control=False, lock=suite)
        onset = suite['tau_cycle'][i%3]
        assert tc == to == -1
        np.testing.assert_array_equal(hc, ho)
        np.testing.assert_array_equal(np.flatnonzero(oc != oo), [onset])
        left = engine.new_state(hc).call_args; right = engine.new_state(ho).call_args
        tree_hash = hashlib.sha256(b''.join(a.tobytes() for a in arrays(left[-1]).values())).hexdigest()
        if frozen_tree_hash is None: frozen_tree_hash = tree_hash
        assert tree_hash == frozen_tree_hash
        c = np.empty((1024, 3), dtype=np.float32); p = np.empty_like(c)
        snapshots = {}; cursor = 0
        for anchor in [a for a in ANCHORS if a < 511 or i in local]:
            end = onset+anchor+1
            c[cursor:end] = trace_components(left, oc[cursor:end]); p[cursor:end] = trace_components(right, oo[cursor:end])
            snapshots[anchor] = (clone_arguments(left), clone_arguments(right)); cursor = end
        c[cursor:] = trace_components(left, oc[cursor:]); p[cursor:] = trace_components(right, oo[cursor:])
        np.testing.assert_array_equal(c[:onset], p[:onset])
        for j, reference in enumerate(expected):
            np.testing.assert_array_equal(c[:, j], reference[i]); np.testing.assert_array_equal(p[:, j], reference[512+i])
            compared += 2048
        for anchor, (cargs, pargs) in snapshots.items():
            index = local[i] if anchor == 511 else i; stop = 768 if anchor == 511 else 512
            start = onset+anchor+1; end = onset+stop; future = oc[start:end]
            np.testing.assert_array_equal(future, oo[start:end])
            cv, pv = c[start:end, 2], p[start:end, 2]
            control_arrays[anchor][index] = cv; outlier_arrays[anchor][index] = pv
            for args, values in ((cargs, cv), (pargs, pv)):
                np.testing.assert_array_equal(trace_scores(clone_arguments(args), future), values); clone_checks += 1
            for direction in DIRECTIONS:
                destination, donor = (pargs, cargs) if direction == DIRECTIONS[0] else (cargs, pargs)
                fine = transfer_legacy_arguments(destination, donor, LEGACY_BLOCKS)
                coarse = transferred_arguments(destination, donor, ['LEGACY_ARCH_ABCD'])
                expected_coarse = trace_scores(coarse, future)
                np.testing.assert_array_equal(trace_scores(fine, future), expected_coarse); fine_coarse_checks += 1
                coarse_arrays[anchor, direction][index] = expected_coarse
                for block in LEGACY_BLOCKS:
                    hybrid = transfer_legacy_arguments(destination, donor, [block])
                    state_bytes = unique_array_bytes(hybrid[:-1])
                    assert state_bytes == unique_array_bytes(destination[:-1])
                    values = trace_scores(hybrid, future)
                    assert unique_array_bytes(hybrid[:-1]) == state_bytes
                    assert np.isfinite(values).all() and np.all((values >= 0)&(values <= 1))
                    hybrids[anchor, block, direction][index] = values; hybrid_points += len(values)
            assert hashlib.sha256(b''.join(a.tobytes() for a in arrays(pargs[-1]).values())).hexdigest() == frozen_tree_hash
        if (i+1)%32 == 0:
            progress('D006 legacy state refinement', f'{i+1}/512 original pairs; original predictions and fine/coarse transfer exact', experiment='D006_LEGACY_STATE_REFINEMENT')
            print('NEXT3 LEGACY TRANSFER', i+1, round(time.perf_counter()-began, 1), flush=True)
    count_cache = {}
    for n in (512, 342):
        _, counts = draw_group_multiplicities(np.arange(n), hyp['bootstrap_replicates'], hyp['bootstrap_seed'])
        count_cache[n] = counts[1:].astype(np.float64)
    coarse_curve_points = coarse_window_rows = 0
    numeric_columns = ['control_mean', 'outlier_mean', 'hybrid_mean', 'delta_from_destination',
        'delta_from_destination_ci_low', 'delta_from_destination_ci_high', 'delta_from_control',
        'delta_from_control_ci_low', 'delta_from_control_ci_high']
    for (anchor, direction), values in coarse_arrays.items():
        reference = old_curves[(old_curves.anchor_offset == anchor)&(old_curves.direction == direction)&(old_curves.block == 'LEGACY_ARCH_ABCD')]
        np.testing.assert_array_equal(values.mean(axis=0, dtype=np.float64), reference.hybrid_probability.to_numpy())
        coarse_curve_points += len(reference)
        rows, _ = evaluate_windows(cohorts[anchor], anchor, 'LEGACY_ARCH_ABCD', direction, values,
                                  control_arrays[anchor], outlier_arrays[anchor], thresholds, count_cache[len(values)])
        expected_windows = old_windows[(old_windows.anchor_offset == anchor)&(old_windows.direction == direction)&(old_windows.block == 'LEGACY_ARCH_ABCD')]
        actual = pd.DataFrame(rows)
        np.testing.assert_array_equal(actual[numeric_columns].to_numpy(), expected_windows[numeric_columns].to_numpy())
        coarse_window_rows += len(actual)
    curves = []; summaries = []; vectors = {}; vector_index = []
    for (anchor, block, direction), values in hybrids.items():
        cv, pv = control_arrays[anchor], outlier_arrays[anchor]
        destination = pv if direction == DIRECTIONS[0] else cv
        for offset, hm, cm, pm, dm in zip(range(anchor+1, 768 if anchor == 511 else 512),
            values.mean(axis=0, dtype=np.float64), cv.mean(axis=0, dtype=np.float64),
            pv.mean(axis=0, dtype=np.float64), destination.mean(axis=0, dtype=np.float64)):
            curves.append({'cohort': cohorts[anchor], 'seed_pairs': len(values), 'anchor_offset': anchor,
                'block': block, 'direction': direction, 'post_injection_offset': offset,
                'control_probability': cm, 'outlier_probability': pm, 'hybrid_probability': hm,
                'delta_from_destination': hm-dm, 'delta_from_control': hm-cm})
        rows, values_by_window = evaluate_windows(cohorts[anchor], anchor, block, direction, values, cv, pv, thresholds, count_cache[len(values)])
        for row, vector in zip(rows, values_by_window):
            key = f'window_{len(summaries):04d}'; vectors[key] = vector
            vector_index.append({'array': key, **{name: row[name] for name in ('cohort', 'anchor_offset', 'block', 'direction', 'window_start_inclusive', 'window_end_exclusive', 'metric', 'alpha')}})
            summaries.append(row)
    assert compared == 3145728 and len(summaries) == 630
    pd.DataFrame(curves).to_csv(out/'MEAN_COUNTERFACTUAL_CURVES.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'WINDOW_COUNTERFACTUAL_INTERVALS.csv', index=False)
    pd.DataFrame(vector_index).to_csv(out/'PAIR_WINDOW_ARRAY_INDEX.csv', index=False)
    np.savez_compressed(out/'PAIR_WINDOW_STATISTICS.npz', **vectors)
    for name, sha in implementation['source_sha256'].items(): assert digest(ROOT/name) == sha
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'policy_sha256': digest(out/'POLICY.json'), 'original_seed_pairs': 512, 'late_original_availability_subgroup': 342,
        'original_component_and_blend_predictions_compared': compared, 'original_prediction_max_abs_diff': 0.,
        'clone_replay_checks': clone_checks, 'all_seven_fine_to_coarse_legacy_transfer_checks': fine_coarse_checks,
        'D005_coarse_aggregate_curve_points_reconstructed_exact': coarse_curve_points,
        'D005_coarse_window_rows_reconstructed_exact': coarse_window_rows,
        'single_block_hybrid_prediction_points': hybrid_points, 'summary_rows': len(summaries),
        'fixed_state': True, 'model_arrays_unchanged': True, 'new_fits': 0, 'new_scientific_candidates': 0,
        'real_DEV_rows_accessed': 0, 'seal_rows': 0, 'null_TS_AUC_computed': False,
        'thresholds_recalibrated': False, 'suite_extended': False, 'elapsed_seconds': time.perf_counter()-began,
        'limitations': 'Same synthetic cohort and conditional diagnostic intervals; hybrid effects interact and may not correspond to reachable input histories. No remedy or scientific promotion inferred.',
        'files_sha256': {name: digest(out/name) for name in ('MEAN_COUNTERFACTUAL_CURVES.csv', 'WINDOW_COUNTERFACTUAL_INTERVALS.csv', 'PAIR_WINDOW_ARRAY_INDEX.csv', 'PAIR_WINDOW_STATISTICS.npz')}}, immutable=True)
    print('NEXT3 LEGACY TRANSFER PASS', len(summaries), flush=True)


if __name__ == '__main__':
    main()

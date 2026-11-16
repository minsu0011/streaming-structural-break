"""Fixed matched recurrent-state interventions, original O1 synthetic cohort only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
import sys
import hashlib
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.state_transfer import (BLOCKS, arrays, clone_arguments, transferred_arguments,
    trace_components, trace_scores, assert_historical_constants_equal)
from src.next2.canonical_model import load
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next.fused_runtime import unique_array_bytes
from src.next2.counterexamples import policy, generate, source_hash as generator_hash
from src.validation.group_bootstrap import draw_group_multiplicities

J = 'J102_SERIAL_VARIANCE_512'; C = 'C205_DEPENDENCE'; B = 'BL2_C205_DEPENDENCE_80'
ANCHORS = (0, 127, 255, 511)
BOUNDARIES = (1, 8, 32, 128, 256, 512, 768)
DIRECTIONS = ('CONTROL_INTO_OUTLIER', 'OUTLIER_INTO_CONTROL')


def checked_inputs():
    source_lock = read_json(BASE/'outlier_state_transfer/IMPLEMENTATION_LOCK.json')
    for name, sha in source_lock['source_sha256'].items(): assert digest(ROOT/name) == sha, name
    unit_path = BASE/'maintenance/STATE_TRANSFER_UNIT_TESTS.xml'
    suites = ET.parse(unit_path).getroot().findall('.//testsuite')
    assert suites and sum(int(s.attrib['tests']) for s in suites) == 7
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) == int(s.attrib.get('skipped', 0)) == 0 for s in suites)
    paths = [(ROOT/'artifacts/next2/counterexamples/counterexample/O1', J),
        (ROOT/'artifacts/next2/finalist_counterexamples/counterexample/O1', C),
        (ROOT/'artifacts/next2/finalist_counterexamples/counterexample/O1', B)]
    expected = []; sources = {}
    for folder, name in paths:
        receipt = read_json(folder/'RESULTS.json'); assert receipt['status'] == 'PASS'
        for filename in (name+'.npy', 'synthetic_tau.npy'):
            assert digest(folder/filename) == receipt['files_sha256'][filename]
            sources[(folder/filename).relative_to(ROOT).as_posix()] = digest(folder/filename)
        tau = np.load(folder/'synthetic_tau.npy', allow_pickle=False); assert tau.shape == (1024,) and np.all(tau == -1)
        values = np.load(folder/(name+'.npy'), mmap_mode='r', allow_pickle=False)
        assert values.shape == (1024, 1024); expected.append(values)
        sources[(folder/'RESULTS.json').relative_to(ROOT).as_posix()] = digest(folder/'RESULTS.json')
    table_path = ROOT/'artifacts/next2/finalist_counterexamples/COUNTEREXAMPLES.csv'
    table = pd.read_csv(table_path, float_precision='round_trip'); thresholds = {}
    for alpha in (.05, .01):
        row = table[(table.candidate == B)&(table.scenario == 'O1')&(table.alpha == alpha)]
        assert len(row) == 1; thresholds[alpha] = float(row.iloc[0].threshold)
    sources[table_path.relative_to(ROOT).as_posix()] = digest(table_path)
    return source_lock, expected, sources, thresholds


def evaluate_windows(cohort, anchor, block, direction, hybrids, control, outlier, thresholds, counts):
    start = anchor+1; stop = 768 if anchor == 511 else 512
    cuts = sorted({start, stop, *[x for x in BOUNDARIES if start < x < stop]})
    summaries = []; vectors = []
    destination = outlier if direction == 'CONTROL_INTO_OUTLIER' else control
    for begin, end in zip(cuts[:-1], cuts[1:]):
        take = slice(begin-start, end-start)
        metrics = [('probability_mean', None, None, [a[:, take].mean(axis=1, dtype=np.float64) for a in (control, outlier, destination, hybrids)])]
        for alpha, threshold in thresholds.items():
            crossing = [a[:, take] > threshold for a in (control, outlier, destination, hybrids)]
            metrics += [('threshold_occupancy', alpha, threshold, [a.mean(axis=1) for a in crossing]),
                        ('any_crossing', alpha, threshold, [a.any(axis=1).astype(np.float64) for a in crossing])]
        for metric, alpha, threshold, (c, o, d, h) in metrics:
            from_destination = h-d; from_control = h-c
            draws = counts@np.column_stack((from_destination, from_control))/len(h)
            interval = np.quantile(draws, [.025, .975], axis=0)
            summaries.append({'cohort': cohort, 'seed_pairs': len(h), 'anchor_offset': anchor,
                'block': block, 'direction': direction, 'window_start_inclusive': begin,
                'window_end_exclusive': end, 'metric': metric, 'alpha': alpha, 'threshold': threshold,
                'control_mean': float(c.mean()), 'outlier_mean': float(o.mean()), 'hybrid_mean': float(h.mean()),
                'delta_from_destination': float(from_destination.mean()),
                'delta_from_destination_ci_low': float(interval[0, 0]), 'delta_from_destination_ci_high': float(interval[1, 0]),
                'delta_from_control': float(from_control.mean()),
                'delta_from_control_ci_low': float(interval[0, 1]), 'delta_from_control_ci_high': float(interval[1, 1]),
                'interpretation': 'CONDITIONAL_SYNTHETIC_STATE_INTERVENTION_NOT_MODEL_SELECTION'})
            vectors.append(np.column_stack((c, o, h)))
    return summaries, vectors


def main():
    implementation, expected, sources, thresholds = checked_inputs()
    out = BASE/'outlier_state_transfer/run_v1'; out.mkdir(parents=True, exist_ok=False)
    suite = policy(); assert suite['evaluation_series_per_scenario'] == 512 and suite['online_length'] == 1024
    hypothesis = BASE/'hypotheses/D005_OUTLIER_STATE_TRANSFER.json'; hyp = read_json(hypothesis)
    model_path = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    reference_sha = read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256'][model_path.relative_to(ROOT).as_posix()]
    assert digest(model_path) == reference_sha
    engine = PreparedSharedBlendPredictor(load(model_path), resident=False)
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'hypothesis_sha256': digest(hypothesis),
        'implementation_lock_sha256': digest(BASE/'outlier_state_transfer/IMPLEMENTATION_LOCK.json'),
        'model_sha256': reference_sha, 'source_bindings_sha256': sources,
        'counterexample_suite_sha256': digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'), 'generator_sha256': generator_hash(),
        'thresholds': thresholds, 'blocks': BLOCKS, 'directions': DIRECTIONS,
        'anchors': ANCHORS, 'new_fit': False, 'new_candidate': False}, immutable=True)
    cohorts = {anchor: ('ALL_512_COMMON_OFFSETS_1_511' if anchor < 511 else 'ORIGINAL_ONSET_128_256_ONLY_342') for anchor in ANCHORS}
    selected = [i for i in range(512) if suite['tau_cycle'][i%3] <= 256]; assert len(selected) == 342
    local = {i: j for j, i in enumerate(selected)}
    control_arrays = {a: np.empty((342 if a == 511 else 512, 256 if a == 511 else 511-a), dtype=np.float32) for a in ANCHORS}
    outlier_arrays = {a: np.empty_like(value) for a, value in control_arrays.items()}
    hybrids = {(a, b, d): np.empty_like(control_arrays[a]) for a in ANCHORS for b in BLOCKS for d in DIRECTIONS}
    compared = 0; clone_checks = 0; full_transfer_checks = 0; hybrid_points = 0; began = time.perf_counter()
    frozen_tree_hash = None
    for i in range(512):
        hc, oc, tc = generate('O1', i, control=True, lock=suite)
        ho, oo, to = generate('O1', i, control=False, lock=suite)
        onset = suite['tau_cycle'][i%3]
        assert tc == to == -1; np.testing.assert_array_equal(hc, ho)
        np.testing.assert_array_equal(np.flatnonzero(oc != oo), [onset])
        left = engine.new_state(hc).call_args; right = engine.new_state(ho).call_args
        tree_hash = hashlib.sha256(b''.join(a.tobytes() for a in arrays(left[-1]).values())).hexdigest()
        if frozen_tree_hash is None: frozen_tree_hash = tree_hash
        assert tree_hash == frozen_tree_hash
        c = np.empty((1024, 3), dtype=np.float32); p = np.empty_like(c); snapshots = {}; cursor = 0
        available = [a for a in ANCHORS if a < 511 or i in local]
        for anchor in available:
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
            np.testing.assert_array_equal(future, oo[start:end]); assert_historical_constants_equal(cargs, pargs)
            cv, pv = c[start:end, 2], p[start:end, 2]
            control_arrays[anchor][index] = cv; outlier_arrays[anchor][index] = pv
            for source_args, reference in ((cargs, cv), (pargs, pv)):
                np.testing.assert_array_equal(trace_scores(clone_arguments(source_args), future), reference); clone_checks += 1
            for destination, donor, reference in ((pargs, cargs, cv), (cargs, pargs, pv)):
                whole = transferred_arguments(destination, donor, BLOCKS)
                np.testing.assert_array_equal(trace_scores(whole, future), reference); full_transfer_checks += 1
            for block in BLOCKS:
                for direction in DIRECTIONS:
                    destination, donor = (pargs, cargs) if direction == 'CONTROL_INTO_OUTLIER' else (cargs, pargs)
                    hybrid = transferred_arguments(destination, donor, [block]); before = unique_array_bytes(hybrid[:-1])
                    assert before == unique_array_bytes(destination[:-1])
                    value = trace_scores(hybrid, future); assert unique_array_bytes(hybrid[:-1]) == before
                    assert np.isfinite(value).all() and np.all((value >= 0)&(value <= 1))
                    hybrids[anchor, block, direction][index] = value; hybrid_points += len(value)
            assert hashlib.sha256(b''.join(a.tobytes() for a in arrays(pargs[-1]).values())).hexdigest() == frozen_tree_hash
        if (i+1)%32 == 0:
            progress('D005 matched-state counterexample characterization', f'{i+1}/512 pairs; original predictions and complete transfers exact', experiment='D005_OUTLIER_STATE_TRANSFER')
            print('NEXT3 STATE TRANSFER', i+1, round(time.perf_counter()-began, 1), flush=True)
    assert compared == 3*1024*1024
    curves = []; summaries = []; vector_arrays = {}; vector_index = []; count_cache = {}
    for n in (512, 342):
        _, counts = draw_group_multiplicities(np.arange(n), hyp['bootstrap_replicates'], hyp['bootstrap_seed'])
        count_cache[n] = counts[1:].astype(np.float64)
    for (anchor, block, direction), values in hybrids.items():
        cv, pv = control_arrays[anchor], outlier_arrays[anchor]
        destination = pv if direction == 'CONTROL_INTO_OUTLIER' else cv
        for offset, hmean, cmean, pmean, dmean in zip(range(anchor+1, 768 if anchor == 511 else 512),
                values.mean(axis=0, dtype=np.float64), cv.mean(axis=0, dtype=np.float64),
                pv.mean(axis=0, dtype=np.float64), destination.mean(axis=0, dtype=np.float64)):
            curves.append({'cohort': cohorts[anchor], 'seed_pairs': len(values), 'anchor_offset': anchor,
                'block': block, 'direction': direction, 'post_injection_offset': offset,
                'control_probability': cmean, 'outlier_probability': pmean, 'hybrid_probability': hmean,
                'delta_from_destination': hmean-dmean, 'delta_from_control': hmean-cmean})
        rows, vectors = evaluate_windows(cohorts[anchor], anchor, block, direction, values, cv, pv, thresholds, count_cache[len(values)])
        for row, vector in zip(rows, vectors):
            key = f'window_{len(summaries):04d}'; vector_arrays[key] = vector
            vector_index.append({'array': key, **{k: row[k] for k in ('cohort', 'anchor_offset', 'block', 'direction', 'window_start_inclusive', 'window_end_exclusive', 'metric', 'alpha')}})
            summaries.append(row)
    pd.DataFrame(curves).to_csv(out/'MEAN_COUNTERFACTUAL_CURVES.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'WINDOW_COUNTERFACTUAL_INTERVALS.csv', index=False)
    pd.DataFrame(vector_index).to_csv(out/'PAIR_WINDOW_ARRAY_INDEX.csv', index=False)
    np.savez_compressed(out/'PAIR_WINDOW_STATISTICS.npz', **vector_arrays)
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'policy_sha256': digest(out/'POLICY.json'), 'original_seed_pairs': 512, 'late_original_availability_subgroup': 342,
        'original_component_and_blend_predictions_compared': compared, 'original_prediction_max_abs_diff': 0.,
        'clone_replay_checks': clone_checks, 'full_six_block_transfer_replay_checks': full_transfer_checks,
        'single_block_hybrid_prediction_points': hybrid_points, 'fixed_state': True, 'model_arrays_unchanged': True,
        'new_fits': 0, 'new_scientific_candidates': 0, 'real_DEV_rows_accessed': 0, 'seal_rows': 0,
        'null_TS_AUC_computed': False, 'thresholds_recalibrated': False, 'suite_extended': False,
        'elapsed_seconds': time.perf_counter()-began, 'summary_rows': len(summaries),
        'files_sha256': {name: digest(out/name) for name in ('MEAN_COUNTERFACTUAL_CURVES.csv', 'WINDOW_COUNTERFACTUAL_INTERVALS.csv', 'PAIR_WINDOW_ARRAY_INDEX.csv', 'PAIR_WINDOW_STATISTICS.npz')},
        'limitations': 'Explicit synthetic state interventions may be unreachable under ordinary history; effects interact and are not additive. Conditional paired intervals do not account for adaptive diagnostic choice or multiplicity; no remedy or model promotion inferred.'}, immutable=True)
    progress('D005 state-transfer characterization complete', 'Every original point, clone and full transfer exact; single-block effects are diagnostic only', experiment='D005_OUTLIER_STATE_TRANSFER')


if __name__ == '__main__': main()

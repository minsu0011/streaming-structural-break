"""Paired current-channel response on the already fixed cross-background seeds."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from numba import njit
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.counterexamples import policy
from src.next2.cross_background_scale import series, BACKGROUNDS, FACTORS
from src.next2.pruned_runtime import CompactPairState, _values
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


@njit(cache=False)
def replay(points, filter_args, parameters, conditional, mode, calibration, work):
    result = np.empty((len(points), 2), dtype=np.float32)
    for t in range(len(points)):
        current = _values(float(points[t]), filter_args, parameters, conditional, mode, calibration, work)
        for k in range(2): result[t, k] = min(max(current[k], -12.), 12.)
    return result


def trace(h, o, settings):
    state = CompactPairState(h, settings, 'serial')
    return replay(o, state.filter_args, state.parameters, state.conditional, state.mode, state.calibration, state.work)


def main():
    out = ROOT/'artifacts/next2/channel_response'; out.mkdir(parents=True, exist_ok=True)
    paths = ['COUNTEREXAMPLE_SUITE_LOCK.json', 'docs/NEXT2_CHANNEL_RESPONSE_PROTOCOL.md',
        'src/next2/cross_background_scale.py', 'src/next2/counterexamples.py', 'src/next2/pruned_runtime.py',
        'src/next/whitening.py', 'src/next/fast_variance_initialization.py', 'src/next/serial_channel_evidence.py',
        'scripts/analyze_next2_channel_response.py']
    identity = {'files_sha256': {p: sha256(ROOT/p) for p in paths}, 'new_learner_fits': 0, 'model_scores': 0}
    if (out/'POLICY.json').exists(): assert json.loads((out/'POLICY.json').read_text()) == identity
    else: write_json(out/'POLICY.json', identity)
    rng = np.random.default_rng(2026090973)
    indices = rng.integers(0, 512, (2000, 512))
    counts = np.asarray([np.bincount(row, minlength=512) for row in indices], dtype=np.float64)/512
    lock = policy(); settings = {'order': 8, 'normalization': 'arch1', 'max_age': 512}
    rows = []; vectors = []; all_checks = 0
    for bg in BACKGROUNDS:
        for factor in FACTORS:
            summaries = np.empty((512, 3, 2), dtype=np.float64)
            for seed in range(512):
                h0, o0, t0 = series(lock, bg, factor, seed, control=True)
                h1, o1, tau = series(lock, bg, factor, seed)
                assert t0 == -1 and tau in (128, 256, 512)
                np.testing.assert_array_equal(h0, h1)
                a, b = trace(h0, o0, settings), trace(h1, o1, settings)
                np.testing.assert_array_equal(a[:tau], b[:tau])
                if seed == 0:
                    state = CompactPairState(h1, settings, 'serial')
                    actual = np.asarray([state.update(float(x))[[0, 13]].copy() for x in o1], dtype=np.float32)
                    np.testing.assert_array_equal(actual, b); all_checks += 1
                for j, horizon in enumerate((32, 128, 512)):
                    summaries[seed, j] = b[tau:tau+horizon].astype(float).mean(axis=0)-a[tau:tau+horizon].astype(float).mean(axis=0)
            for j, horizon in enumerate((32, 128, 512)):
                for k, channel in enumerate(('raw_energy', 'log_variance_forecast')):
                    values = summaries[:, j, k]; estimate = float(values.mean()); draws = counts@values
                    rows.append({'background': bg, 'variance_factor': factor, 'channel': channel, 'post_window': horizon,
                        'seed_pairs': 512, 'mean_paired_clipped_current_delta': estimate,
                        'ci_2_5': float(np.quantile(draws, .025)), 'ci_97_5': float(np.quantile(draws, .975)),
                        'fraction_pairs_negative_delta': float(np.mean(values < 0))})
                    vectors.append(np.r_[estimate, draws])
            print('NEXT2 CURRENT CHANNEL RESPONSE', bg, factor, flush=True)
    pd.DataFrame(rows).to_csv(out/'PAIRED_CHANNEL_RESPONSE.csv', index=False)
    np.save(out/'BOOTSTRAP_VECTORS.npy', np.asarray(vectors), allow_pickle=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'identity': identity, 'created_utc': utc_now(),
        'scenarios': 16, 'seed_pairs_per_scenario': 512, 'channel_updater_exactness_checks': all_checks,
        'exact_preintervention_pair_checks': 8192, 'new_learner_fits': 0, 'model_predictions': 0,
        'files_sha256': {name: sha256(out/name) for name in ('PAIRED_CHANNEL_RESPONSE.csv', 'BOOTSTRAP_VECTORS.npy')},
        'scope': 'Signed clipped current channels only, before evidence summaries and trees. Paired generator-conditional uncertainty; post-hoc diagnostic without selection adjustments.'})
    checkpoint('Fixed paired energy and forecast response across backgrounds', ['Synthesize scale-down evidence at channel and predictor levels', 'Begin separate audit only after six research hours'])


if __name__ == '__main__': main()

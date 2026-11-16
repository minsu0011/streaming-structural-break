"""Fixed synthetic H-only energy calibration asymmetry; no model predictions."""
from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.counterexamples import generate, policy
from src.next.whitening import fit_historical
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next2.pruned_runtime import historical_pair, CompactPairState
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    out = ROOT/'artifacts/next2/h_calibration_asymmetry'; out.mkdir(parents=True, exist_ok=True)
    paths = ['COUNTEREXAMPLE_SUITE_LOCK.json', 'docs/NEXT2_H_CALIBRATION_ASYMMETRY_PROTOCOL.md',
        'src/next2/counterexamples.py', 'src/next/whitening.py', 'src/next/fast_variance_initialization.py',
        'src/next2/pruned_runtime.py', 'src/next/normalization.py', 'scripts/analyze_next2_h_calibration.py']
    identity = {'sources_sha256': {p: sha256(ROOT/p) for p in paths}, 'new_learner_fits': 0, 'prediction_scores': 0}
    if (out/'POLICY.json').exists(): assert json.loads((out/'POLICY.json').read_text()) == identity
    else: write_json(out/'POLICY.json', identity)
    settings = {'order': 8, 'normalization': 'arch1', 'max_age': 512}
    rows = []; lock = policy(); assert lock['calibration_null_series_per_regime'] == 1000
    for bg in ('N0', 'N1', 'N2', 'N3'):
        for seed in range(1000):
            h, _, tau = generate(bg, seed, partition=0, horizon=1, control=True, lock=lock); assert tau == -1
            fit = fit_historical(h, 8)
            parameters, conditional = fast_variance_parameters(fit['historical_innovations'], 'arch1')
            channels = historical_pair(fit['historical_innovations'], parameters, 3)
            channels = channels[min(160, len(channels)//4):]
            calibration = np.stack([channels.mean(axis=0), np.maximum(channels.std(axis=0), 1e-6)])
            if seed == 0:
                state = CompactPairState(h, settings, 'serial')
                np.testing.assert_array_equal(calibration, state.calibration)
                np.testing.assert_array_equal(parameters, state.parameters)
            centered = channels[:, 0]-calibration[0, 0]; standardized = centered/calibration[1, 0]
            squared = centered*centered; top = math.ceil(.01*len(squared))
            floor = (-1-calibration[0, 0])/calibration[1, 0]
            assert np.min(standardized) >= floor-1e-12
            rows.append({'background': bg, 'synthetic_partition': 0, 'seed_index': seed,
                'energy_H_mean': calibration[0, 0], 'energy_H_sd': calibration[1, 0],
                'forecast_H_mean': calibration[0, 1], 'forecast_H_sd': calibration[1, 1],
                'energy_negative_floor_before_clipping': floor, 'energy_negative_floor_after_clipping': max(-12., floor),
                'fitted_arch_alpha': parameters[3], 'maximum_standardized_H_energy': float(standardized.max()),
                'H_fraction_above_positive_clip12': float(np.mean(standardized > 12)),
                'top_one_percent_share_of_energy_centered_ss': float(np.partition(squared, len(squared)-top)[-top:].sum()/squared.sum())})
        print('NEXT2 H CALIBRATION', bg, 1000, flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(out/'SYNTHETIC_H_PARAMETERS.csv', index=False)
    aggregates = []
    for bg, group in frame.groupby('background'):
        for name in frame.columns[3:]:
            values = group[name].to_numpy(float)
            aggregates.append({'background': bg, 'quantity': name, 'synthetic_H_samples': len(values),
                **{'q'+str(q): float(np.quantile(values, q)) for q in (.01, .05, .5, .95, .99)}})
    pd.DataFrame(aggregates).to_csv(out/'QUANTILES.csv', index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'identity': identity, 'created_utc': utc_now(),
        'H_only_calibrations': 4000, 'exact_full_state_calibration_checks': 4, 'new_learner_fits': 0,
        'competition_rows': 0, 'model_predictions': 0, 'threshold_changes': 0,
        'files_sha256': {name: sha256(out/name) for name in ('SYNTHETIC_H_PARAMETERS.csv', 'QUANTILES.csv')},
        'scope': 'Post-hoc explanatory distribution of fixed H calibration and exact raw-energy lower bounds. Empirical across-seed quantiles are not uncertainty intervals or causal effects.'})
    checkpoint('Fixed H-only energy calibration asymmetry', ['Integrate mathematical and empirical counterexample interpretation', 'Finish six-hour research before separate audit'])


if __name__ == '__main__': main()

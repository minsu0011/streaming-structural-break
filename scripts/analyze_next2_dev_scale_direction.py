"""Describe signed coverage of existing DEV mechanism windows, without scoring."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    data = prepare_dev(ROOT)
    base = ROOT/'artifacts/next/current_forensic'
    regime_lock = json.loads((base/'REGIME_ASSIGNMENT_LOCK.json').read_text())
    path = base/'DEV_SIGNATURES.parquet'
    assert sha256(path) == regime_lock['files_sha256'][path.name]
    frame = pd.read_parquet(path)
    assert set(data.guard.admit(frame.dataset_id.to_numpy(), purpose='Existing DEV scale-direction diagnostic')) == set(data.guard.dev_ids)
    fixed = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text())['mechanism']
    out = ROOT/'artifacts/next2/dev_scale_direction'; out.mkdir(parents=True, exist_ok=True)
    identity = {'script_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(ROOT/'docs/NEXT2_DEV_SCALE_DIRECTION_PROTOCOL.md'),
        'mechanism_source_sha256': sha256(ROOT/'src/next/regimes.py'), 'signature_sha256': sha256(path),
        'fixed_policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'), 'mechanism_policy': fixed,
        'new_fits': 0, 'predictions_accessed': 0, 'quality_scores_computed': 0}
    if (out/'POLICY.json').exists(): assert json.loads((out/'POLICY.json').read_text()) == identity
    else: write_json(out/'POLICY.json', identity)
    rows = []; maximum_error = 0.
    for r in frame.itertuples():
        direction = 'UNDEFINED_NO_BREAK' if r.tau < 0 else 'UNDEFINED_SHORT_POST'
        effect = None
        if r.tau >= 0:
            h, o, tau = data.get(int(r.dataset_id)); assert tau == r.tau
            pre = np.concatenate([np.asarray(h, dtype=np.float64), np.asarray(o[:tau], dtype=np.float64)])[-fixed['window']:]
            post = np.asarray(o[tau:tau+fixed['window']], dtype=np.float64)
            if min(len(pre), len(post)) >= fixed['minimum_each_side']:
                effect = float(np.log(max(float(post.std()), 1e-8)/max(float(pre.std()), 1e-8)))
                error = abs(abs(effect)-float(r.scale_effect)); assert error <= 1e-12
                maximum_error = max(maximum_error, error)
                direction = 'OBSERVED_SD_DECREASE' if effect < 0 else 'OBSERVED_SD_INCREASE' if effect > 0 else 'OBSERVED_SD_EQUAL'
        rows.append({'signature': r.signature, 'historical_regime': int(r.historical_regime), 'fold': int(r.fold),
            'break': r.tau >= 0, 'direction': direction, 'signed_log_sd_ratio': effect})
    values = pd.DataFrame(rows); counts = []; quantiles = []
    for scope, keys in (('all_DEV', []), ('existing_signature', ['signature']),
                        ('existing_H_regime', ['historical_regime']), ('original_fold', ['fold']),
                        ('signature_by_H_regime', ['signature', 'historical_regime'])):
        groups = [((), values)] if not keys else values.groupby(keys, dropna=False)
        for key, group in groups:
            if not isinstance(key, tuple): key = (key,)
            detail = dict(zip(keys, key)); common = {'scope': scope, **detail}
            for direction, part in group.groupby('direction'):
                counts.append({**common, 'direction': direction, 'series': len(part), 'break_series': int(part['break'].sum()),
                    'support_count_at_least_100_break_series': int(part['break'].sum()) >= 100})
            eligible = group.signed_log_sd_ratio.dropna().to_numpy(float)
            if len(eligible):
                quantiles.append({**common, 'eligible_break_series': len(eligible),
                    **{'signed_log_sd_ratio_q'+str(q): float(np.quantile(eligible, q)) for q in (.05, .25, .5, .75, .95)}})
    pd.DataFrame(counts).to_csv(out/'DIRECTION_COUNTS.csv', index=False)
    pd.DataFrame(quantiles).to_csv(out/'SIGNED_EFFECT_QUANTILES.csv', index=False)
    data.guard.check_state()
    write_json(out/'RESULTS.json', {'status': 'PASS', 'identity': identity, 'created_utc': utc_now(),
        'DEV_series': len(values), 'break_series': int(values['break'].sum()), 'seal_rows': 0,
        'maximum_absolute_effect_reconstruction_error': maximum_error,
        'old_scale_signature_log_sd_cutoff': fixed['absolute_log_sd_ratio_threshold'],
        'corresponding_ideal_variance_ratio_bounds': [float(np.exp(-2*fixed['absolute_log_sd_ratio_threshold'])), float(np.exp(2*fixed['absolute_log_sd_ratio_threshold']))],
        'files_sha256': {name: sha256(out/name) for name in ('DIRECTION_COUNTS.csv', 'SIGNED_EFFECT_QUANTILES.csv')},
        'new_fits': 0, 'quality_scores_computed': 0, 'per_series_output_written': False,
        'limitations': 'Observed raw-window SD direction is an evaluation-only proxy. It is not a known latent innovation variance factor, and the dominant-signature allocation also depends on mean, dependence and tail effects. No directional model gain is estimated here.'})
    checkpoint('Real DEV directional scale coverage without prediction scoring', ['Report limits of scale-down claims', 'Complete research synthesis before audit'])
    print('NEXT2 DEV SCALE COVERAGE PASS', len(values), maximum_error, flush=True)


if __name__ == '__main__': main()

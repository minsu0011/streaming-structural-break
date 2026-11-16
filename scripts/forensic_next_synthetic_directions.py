"""Post-hoc sign fractures of fixed synthetic futures, with paired scale CIs.

The observed heavy-tail scale weakness motivates this transparent follow-up.
No new futures, model fits, weights, thresholds or competition scores are used.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import gc
import json
import sys
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.bootstrap import PreparedAUC, paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    SealGuard(ROOT).check_state()
    set_num_threads(2)
    started = time.perf_counter()
    root = ROOT/'artifacts/next/synthetic_variance_stress'
    cache = ROOT/'data/next/synthetic_variance_stress_v1'
    out = root/'direction_fractures'
    out.mkdir(exist_ok=True)
    parent = json.loads((root/'RESULTS.json').read_text(encoding='utf-8'))
    if parent['status'] != 'PASS':
        raise RuntimeError('Parent controlled diagnostic is incomplete')
    policy_path = ROOT/'configs/next_synthetic_stress_policy.json'
    if sha256(policy_path) != parent['policy_sha256']:
        raise RuntimeError('Original synthetic policy changed')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    identity = {'parent_result_sha256': sha256(root/'RESULTS.json'), 'script_sha256': sha256(Path(__file__)),
        'bootstrap_code_sha256': sha256(ROOT/'src/next/bootstrap.py'), 'replicates': 2000, 'bootstrap_seed': 202609094,
        'nomination': 'Post-hoc after aggregate controlled results revealed V007 scale/persistence weaknesses in heavy-tail ARCH regimes.',
        'scope': 'Report both preexisting polarities for all five regimes and mean/scale/mixed scenarios. Bootstrap both scale directions in every regime, not only the most favorable result.',
        'sampling_unit': 'A generated series index, containing both its changed and paired stationary-null future; use the same multiplicity for both futures and every model.',
        'authority': 'Secondary synthetic counterexample diagnosis; no model-selection or production-threshold changes.'}
    if (out/'POLICY.json').exists() and json.loads((out/'POLICY.json').read_text(encoding='utf-8')) != identity:
        raise RuntimeError('Direction forensic policy changed')
    if not (out/'POLICY.json').exists():
        write_json(out/'POLICY.json', identity)
    if (out/'RESULTS.json').exists():
        report = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        if any(sha256(out/name) != digest for name, digest in report['files_sha256'].items()):
            raise RuntimeError('Completed synthetic direction report changed')
        print('SYNTHETIC DIRECTIONS retained PASS')
        return
    roles = list(policy['models'])
    rows, paired_rows, sources = [], [], {}
    length = policy['online_points']
    for regime, regime_policy in enumerate(policy['regimes']):
        blocks = []
        for first in range(0, policy['evaluation_series_per_regime'], 100):
            name = f'partition_1_regime_{regime}_block_{first:04d}'
            record = json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
            if record['status'] != 'PASS' or sha256(cache/(name+'.npy')) != record['prediction_sha256']:
                raise RuntimeError('Frozen controlled prediction block changed')
            sources[name] = record['prediction_sha256']
            blocks.append(np.load(cache/(name+'.npy'), allow_pickle=False))
        values = np.concatenate(blocks, axis=2)
        for polarity in (0, 1):
            chosen = np.arange(policy['evaluation_series_per_regime'])[polarity::2]
            times = np.tile(np.arange(length, dtype=np.int32), len(chosen))
            taus = np.array([policy['break_times'][index%len(policy['break_times'])] for index in chosen])
            target = (times >= np.repeat(taus, length)).astype(np.uint8)
            y, t = np.r_[target, np.zeros_like(target)], np.r_[times, times]
            groups = np.r_[np.repeat(chosen, length), np.repeat(chosen, length)]
            code, counts = draw_group_multiplicities(groups, identity['replicates'], identity['bootstrap_seed']+regime*10+polarity)
            for scenario in ('MEAN_SHIFT', 'SCALE_SHIFT', 'MIXED_SHIFT'):
                column = policy['scenarios'].index(scenario)
                samples = {}
                for r, role in enumerate(roles):
                    prediction = np.r_[values[r, column, chosen].ravel(), values[r, 0, chosen].ravel()]
                    score = ts_auc(y, prediction, t)
                    rows.append({'model': role, 'regime': regime_policy['name'], 'scenario': scenario,
                        'polarity': 'positive_mean_or_scale_up' if polarity == 0 else 'negative_mean_or_scale_down',
                        'changed_series': len(chosen), 'paired_null_series': len(chosen), 'synthetic_ts_auc': score})
                    if scenario == 'SCALE_SHIFT':
                        name = f'regime_{regime}_polarity_{polarity}_{role}_bootstrap.npy'
                        digest_path = out/(name+'.json')
                        if (out/name).exists() and digest_path.exists():
                            digest = json.loads(digest_path.read_text(encoding='utf-8'))
                            if sha256(out/name) != digest['sha256']:
                                raise RuntimeError('Completed direction bootstrap array changed')
                            sample = np.load(out/name, allow_pickle=False)
                        else:
                            sample = PreparedAUC.create(y, prediction, t, code, counts.shape[1]).evaluate(counts)
                            np.save(out/name, sample, allow_pickle=False)
                            write_json(digest_path, {'sha256': sha256(out/name), 'created_utc': utc_now()})
                        np.testing.assert_allclose(sample[0], score, rtol=0, atol=1e-14)
                        samples[role] = sample
                if samples:
                    for baseline in ('PRIMARY', 'HIGHEST_MEAN'):
                        paired_rows.append({'regime': regime_policy['name'], 'scenario': scenario,
                            'polarity': 'scale_up_1_5' if polarity == 0 else 'scale_down_1_over_1_5',
                            'candidate': 'V007_VARIANCE_FORECAST', 'reference': baseline,
                            'sampling_groups': len(chosen), 'folds': 'one generated cell, not competition CV',
                            **paired_summary(samples[baseline][:, None], samples['V007_VARIANCE_FORECAST'][:, None])})
                print('SYNTHETIC DIRECTION', regime, polarity, scenario, flush=True)
        pd.DataFrame(rows).to_csv(out/'DIRECTION_SCORES.csv', index=False)
        pd.DataFrame(paired_rows).to_csv(out/'SCALE_PAIRED_BOOTSTRAP.csv', index=False)
        del blocks, values
        gc.collect()
    files = list(out.glob('*.npy'))+[out/'DIRECTION_SCORES.csv', out/'SCALE_PAIRED_BOOTSTRAP.csv']
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'point_estimates': len(rows),
        'paired_scale_contrasts': len(paired_rows), 'replicates': identity['replicates'],
        'policy_sha256': sha256(out/'POLICY.json'), 'original_prediction_block_sha256': sources,
        'files_sha256': {path.name: sha256(path) for path in files}, 'elapsed_seconds': time.perf_counter()-started,
        'new_predictions': 0, 'new_model_fits': 0, 'seal_rows': 0, 'reduced_new_uses': 0,
        'limitations': 'Conditional paired synthetic uncertainty; comparisons were motivated by observed aggregate results and are not multiplicity- or selection-adjusted. Chosen simulations cannot certify competition or real-world performance.'})
    print('SYNTHETIC DIRECTIONS COMPLETE', flush=True)


if __name__ == '__main__':
    main()

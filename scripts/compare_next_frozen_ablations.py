"""Paired effect sizes for fixed interventions, reusing identical bootstrap draws.

No new fit, resampling seed, feature recipe, score transform or selection gate.
Contrasts are conditional post-hoc descriptions of the already frozen runs.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.bootstrap import paired_summary
from src.next.guard import SealGuard
from src.utils.artifacts import write_json, sha256, utc_now

PAIRS = (
    ('Y001_LONG_VARIANCE_ARCH1', 'V007_VARIANCE_FORECAST', 'Extend ages 128 to 512'),
    ('Y002_LONG_VARIANCE_ARCH2', 'Y001_LONG_VARIANCE_ARCH1', 'ARCH2 at fixed long ages'),
    ('Y003_LONG_VARIANCE_EWMA', 'Y001_LONG_VARIANCE_ARCH1', 'EWMA at fixed long ages'),
    ('Q002_VARIANCE_FORECAST_ONLY', 'V007_VARIANCE_FORECAST', 'Remove raw-energy channel at ages 128'),
    ('Q004_VARIANCE_NO_BAYES', 'V007_VARIANCE_FORECAST', 'Remove Bayes summaries at ages 128'),
    ('Q005_VARIANCE_NO_MEMORY', 'V007_VARIANCE_FORECAST', 'Remove evidence memory at ages 128'),
    ('Q009_VARIANCE_ARCH2_PAIR', 'V007_VARIANCE_FORECAST', 'ARCH2 at ages 128'),
    ('Q010_VARIANCE_EWMA_PAIR', 'V007_VARIANCE_FORECAST', 'EWMA at ages 128'),
    ('Z001_LOG_ENERGY_CURRENT', 'V007_VARIANCE_FORECAST', 'Replace lagged variance forecast by current log energy'),
    ('F101_VARIANCE_TREES50', 'V007_VARIANCE_FORECAST', '50 instead of 100 trees at ages 128'),
    ('B101_LONG_VARIANCE_NO_MEMORY', 'Y001_LONG_VARIANCE_ARCH1', 'Remove evidence memory at ages 512'),
    ('B102_LONG_VARIANCE_NO_BAYES', 'Y001_LONG_VARIANCE_ARCH1', 'Remove Bayes summaries at ages 512'),
    ('B104_LONG_VARIANCE_NO_GLR_MEMORY', 'Y001_LONG_VARIANCE_ARCH1', 'Remove explicit GLR and GLR-memory summaries at ages 512'),
    ('F201_LONG_VARIANCE_TREES50', 'Y001_LONG_VARIANCE_ARCH1', '50 instead of 100 trees at ages 512'),
    ('J101_SERIAL_VARIANCE_128', 'V007_VARIANCE_FORECAST', 'H-only AR1 sum-variance correction at ages 128'),
    ('J102_SERIAL_VARIANCE_512', 'Y001_LONG_VARIANCE_ARCH1', 'H-only AR1 sum-variance correction at ages 512'),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Invalid contrast tag')
    SealGuard(ROOT).check_state()
    out = ROOT/'artifacts/next/fixed_ablation_contrasts'/args.tag
    if out.exists():
        raise RuntimeError('Retain completed ablation contrast snapshots')
    lock_path = ROOT/'ROBUSTNESS_SPLIT_LOCK.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    inputs, rows, skipped = {}, [], []
    cache = {}
    def load(eid):
        if eid in cache:
            return cache[eid]
        folder = ROOT/'artifacts/next/experiments'/eid
        if not (folder/'RESULTS.json').exists():
            return None
        result = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
        if result['status'] == 'BUG' or result.get('full', {}).get('status') != 'COMPLETE':
            return None
        policy = json.loads((folder/'full/POLICY.json').read_text(encoding='utf-8'))
        if policy['policy_sha256'] != sha256(lock_path):
            raise RuntimeError('A paired intervention changed the frozen bootstrap or split policy')
        draws = np.load(folder/'full/bootstrap_scores.npy', allow_pickle=False)
        if draws.shape != (lock['bootstrap']['final_replicates']+1, 5) or not np.isfinite(draws).all():
            raise RuntimeError('Final paired ablation bootstrap matrix incomplete')
        np.testing.assert_allclose(draws[0], result['full']['fold_scores'], rtol=0, atol=1e-14)
        coordinates = {}
        for fold in range(5):
            record_path = folder/f'full/fold_{fold}.json'
            record = json.loads(record_path.read_text(encoding='utf-8'))
            for name, digest in record['files_sha256'].items():
                if sha256(folder/'full'/name) != digest:
                    raise RuntimeError('Frozen ablation model/predictions/coordinates changed')
            coordinates[fold] = record['files_sha256'][f'fold_{fold}_global_rows.npy']
        for path in (folder/'RESULTS.json', folder/'full/POLICY.json', folder/'full/bootstrap_scores.npy'):
            inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
        cache[eid] = result, draws, coordinates
        return cache[eid]
    for candidate, reference, intervention in PAIRS:
        left, right = load(candidate), load(reference)
        if left is None or right is None:
            skipped.append({'candidate': candidate, 'reference': reference, 'reason': 'No valid full CV; no effect size is fabricated from screen-only scores.'})
            continue
        lresult, ldraws, lcoords = left
        rresult, rdraws, rcoords = right
        if lcoords != rcoords:
            raise RuntimeError('Paired intervention has different original validation rows')
        summary = paired_summary(rdraws, ldraws)
        rows.append({'candidate': candidate, 'reference': reference, 'intervention': intervention,
            'candidate_mean': float(ldraws[0].mean()), 'reference_mean': float(rdraws[0].mean()),
            'candidate_worst_fold': float(ldraws[0].min()), 'reference_worst_fold': float(rdraws[0].min()),
            'worst_fold_delta': float(ldraws[0].min()-rdraws[0].min()),
            'positive_original_fold_deltas': int(np.sum(ldraws[0] > rdraws[0])),
            'original_fold_deltas': json.dumps((ldraws[0]-rdraws[0]).tolist()), **summary,
            'candidate_full_status': lresult['status'], 'conditional_post_hoc_contrast': True})
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out/'PAIRED_INTERVENTIONS.csv', index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'contrasts': len(rows), 'skipped': skipped,
        'policy_sha256': sha256(lock_path), 'source_files_sha256': inputs,
        'files_sha256': {'PAIRED_INTERVENTIONS.csv': sha256(out/'PAIRED_INTERVENTIONS.csv')},
        'script_sha256': sha256(Path(__file__)), 'new_fits': 0, 'new_bootstrap_draws': 0,
        'new_selection_rules': 0, 'seal_rows': 0, 'reduced_new_uses': 0,
        'interpretation': 'Fixed-OOF paired contrasts use the exact same 2000 whole-series bootstrap multiplicities within original folds. Interventions were defined in their earlier candidate queues. This synthesis is post-hoc and does not give selection-adjusted or simultaneous coverage. A negative mean delta can coexist with a better worst fold.'})
    print('FIXED ABLATION CONTRASTS', len(rows), 'skipped', len(skipped), flush=True)


if __name__ == '__main__':
    main()

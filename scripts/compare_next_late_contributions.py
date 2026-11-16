"""Conditional paired gains of the three frozen V/Q averages, without refits.

Observed blend means motivate this descriptive comparison. No weights, seeds,
prediction transforms or selection thresholds are changed.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.late_evaluation import assert_late_time, validate_lock
from src.next.guard import SealGuard
from src.next.bootstrap import paired_summary
from src.utils.artifacts import sha256, write_json, utc_now


def main():
    assert_late_time(ROOT)
    validate_lock(ROOT)
    SealGuard(ROOT).check_state()
    out = ROOT/'artifacts/next/late_contribution_contrasts'
    if out.exists():
        raise RuntimeError('Preserve the first late contribution comparison')
    references = ['V007_VARIANCE_FORECAST', 'Q005_VARIANCE_NO_MEMORY', 'Y001_LONG_VARIANCE_ARCH1']
    candidates = ['L101_VARIANCE_PAIR_50_50', 'L102_VARIANCE_PAIR_67_33', 'L103_VARIANCE_PAIR_33_67']
    sources, loaded = {}, {}
    lock = ROOT/'ROBUSTNESS_SPLIT_LOCK.json'
    for eid in [*references, *candidates]:
        folder = ROOT/'artifacts/next/experiments'/eid
        original = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
        policy = json.loads((folder/'full/POLICY.json').read_text(encoding='utf-8'))
        if original['status'] == 'BUG' or original['full']['status'] != 'COMPLETE' or policy['policy_sha256'] != sha256(lock):
            raise RuntimeError('Invalid or differently resampled original OOF')
        draws = np.load(folder/'full/bootstrap_scores.npy', allow_pickle=False)
        if draws.shape != (2001, 5) or not np.isfinite(draws).all():
            raise RuntimeError('Exactly the frozen final 2000 paired draws are required')
        np.testing.assert_allclose(draws[0], original['full']['fold_scores'], rtol=0, atol=1e-14)
        coordinates = []
        for fold in range(5):
            path = folder/f'full/fold_{fold}.json'
            report = json.loads(path.read_text(encoding='utf-8'))
            for name, digest in report['files_sha256'].items():
                if sha256(folder/'full'/name) != digest:
                    raise RuntimeError('Frozen original payload changed')
            coordinates.append(report['files_sha256'][f'fold_{fold}_global_rows.npy'])
        for path in (folder/'RESULTS.json', folder/'full/POLICY.json', folder/'full/bootstrap_scores.npy'):
            sources[path.relative_to(ROOT).as_posix()] = sha256(path)
        loaded[eid] = draws, coordinates
    rows = []
    for eid in candidates:
        left, left_coordinates = loaded[eid]
        for reference in references:
            right, right_coordinates = loaded[reference]
            if left_coordinates != right_coordinates:
                raise RuntimeError('Original validation coordinates differ')
            rows.append({'candidate': eid, 'reference': reference,
                'candidate_mean': float(left[0].mean()), 'reference_mean': float(right[0].mean()),
                'positive_fold_deltas': int(np.sum(left[0] > right[0])),
                'worst_fold_delta': float(left[0].min()-right[0].min()),
                'fold_deltas': json.dumps((left[0]-right[0]).tolist()), **paired_summary(right, left)})
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out/'PAIRED_CONTRIBUTIONS.csv', index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'contrasts': len(rows),
        'script_sha256': sha256(Path(__file__)), 'policy_sha256': sha256(lock), 'source_files_sha256': sources,
        'files_sha256': {'PAIRED_CONTRIBUTIONS.csv': sha256(out/'PAIRED_CONTRIBUTIONS.csv')},
        'new_fits': 0, 'new_predictions': 0, 'new_bootstrap_draws': 0, 'new_selection_rules': 0,
        'seal_rows': 0, 'reduced_new_uses': 0,
        'interpretation': 'Post-hoc conditional description of already frozen, time-gated three-weight results. Identical 2000 whole-series multiplicities within each original fold; no simultaneous or selection-adjusted interval claim.'})
    print('LATE CONTRIBUTION CONTRASTS PASS', len(rows), flush=True)


if __name__ == '__main__':
    main()

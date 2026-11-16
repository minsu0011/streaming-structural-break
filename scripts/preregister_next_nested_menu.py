"""Preregister a finite nested-menu diagnostic without another split search."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.nested_menu import unique_training_tasks
from src.utils.artifacts import write_json, sha256, utc_now

if __name__ == '__main__':
    out = ROOT/'configs/next_nested_menu_policy.json'
    if out.exists():
        raise RuntimeError('Nested-menu policy is immutable')
    guard = SealGuard(ROOT)
    candidates = ['PRIMARY', 'HIGHEST_MEAN', 'V007_VARIANCE_FORECAST',
        'Q005_VARIANCE_NO_MEMORY', 'Y001_LONG_VARIANCE_ARCH1', 'F101_VARIANCE_TREES50']
    reports = {}
    for eid in candidates[2:]:
        path = ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['status'] != 'ROBUSTNESS_PENDING':
            raise RuntimeError('A nominated menu member did not pass its original primary gates')
        reports[eid] = sha256(path)
    files = ['src/next/nested_menu.py', 'src/next/diagnostic_training.py',
        'scripts/nested_next_menu.py', 'src/next/fitting.py', 'src/next/candidate_io.py',
        'src/next/bootstrap.py', 'src/validation/group_bootstrap.py', 'src/scoring/ts_auc.py']
    write_json(out, {'status': 'LOCKED', 'created_utc': utc_now(), 'candidates': candidates,
        'candidate_reports_sha256': reports, 'split_sha256': guard.split.digest, 'seal_lock_sha256': guard.lock_sha,
        'files_sha256': {name: sha256(ROOT/name) for name in files}, 'tasks_per_candidate': unique_training_tasks(),
        'unique_fits': 60, 'held_out_inner_score_cells': 120,
        'partitioning': 'Use original fold k as outer holdout. Four original remaining folds are inner folds; train on the other three. No new fold assignment or split seed is introduced.',
        'reuse': 'The same three-fold training set occurs twice with outer/inner roles swapped; fit it once and predict both excluded folds. Original four-fold-trained outer OOF is reused only after the inner choice.',
        'selection': 'Highest unweighted mean of four inner TS-AUC scores; ties by higher worst inner fold, then lexical candidate ID.',
        'capacity_and_weighting': 'Every candidate retains its complete frozen feature/loss/tree recipe; all train-derived weights and constant-column removal use only its three training folds.',
        'nomination_limitation': 'The finite menu was nominated adaptively using full DEV results before this diagnostic. Therefore the nested outer score is not an unbiased or independent estimate of the whole historical research process.',
        'authority': 'DEV-only secondary menu-stability diagnostic; cannot replace original CV, change final-role gates or open a seal.'})
    print('FROZEN NESTED MENU', len(candidates), 60, sha256(out), flush=True)

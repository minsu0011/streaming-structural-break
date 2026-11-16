"""Bounded attribution and speed studies after the long-age ARCH1 result."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json, sha256, utc_now


if __name__ == '__main__':
    destination = ROOT/'configs/next_long_ablation_candidates.json'
    if destination.exists():
        raise RuntimeError('Long-age follow-up queue is already frozen')
    parent_path = ROOT/'artifacts/next/experiments/Y001_LONG_VARIANCE_ARCH1/RESULTS.json'
    report = json.loads(parent_path.read_text(encoding='utf-8'))
    if report['status'] != 'ROBUSTNESS_PENDING':
        raise RuntimeError('The proposed parent did not pass the frozen full primary gates')
    parent = report['candidate']
    keep = parent['keep_names']
    variants = [
        ('B101_LONG_VARIANCE_NO_MEMORY', 'Remove the two GLR memory features per channel.', [name for name in keep if '_memory_' not in name], None),
        ('B102_LONG_VARIANCE_NO_BAYES', 'Remove the two analytic Bayes tau-mixture features per channel.', [name for name in keep if '_bayes' not in name], None),
        ('B103_LONG_VARIANCE_FORECAST_ONLY', 'Retain only the variance-forecast evidence channel.', [name for name in keep if '__log_variance_forecast_' in name], None),
        ('B104_LONG_VARIANCE_NO_GLR_MEMORY', 'Remove explicit GLR and GLR memory features; retain local and Bayes evidence.', [name for name in keep if '_glr_' not in name and '_memory_' not in name], None),
        ('F201_LONG_VARIANCE_TREES50', 'Only reduce forest size from 100 to the already frozen 50-tree budget.', keep, 50),
        ('F202_LONG_VARIANCE_TREES25', 'Only reduce forest size from 100 to the already frozen 25-tree budget.', keep, 25)]
    candidates = []
    for eid, change, names, trees in variants:
        candidate = copy.deepcopy(parent)
        candidate.update(experiment_id=eid, parent_id=parent['experiment_id'], generation=17,
            phase='LONG_VARIANCE_MATCHED_ABLATION' if trees is None else 'LONG_VARIANCE_TREE_BUDGET',
            hypothesis='The +.00233 long-age gain over the 128-age parent needs matched attribution to GLR memory, Bayes tau integration and the forecast channel, plus a bounded speed tradeoff.',
            change=change, keep_names=names,
            expected_effect='Separate useful slow-change evidence from persistent maxima and evaluate fixed smaller forests without altering online statistics.',
            risks='Adaptively nominated from DEV; nested full evaluations are conditional. The same fixed screen can reject an ablation even if its unobserved full score would be useful. No failed-screen override is allowed.')
        if trees is not None:
            candidate['params']['n_estimators'] = trees
        candidates.append(candidate)
    write_json(destination, {'status': 'LOCKED', 'created_utc': utc_now(), 'generation': 17,
        'fitting_code_sha256': fitting_hash(), 'predictor_code_sha256': predictor_hash(),
        'robustness_policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'parent_result_sha256': sha256(parent_path), 'parent_full_mean': report['full']['mean'],
        'thresholds_splits_seeds_unchanged': True, 'maximum_new_candidates': 6,
        'scope': 'Four named feature removals and two preexisting tree budgets, all with identical long-age parent features or strict subsets. No new horizon, decay, depth, leaf, learning-rate, seed or weight grid.',
        'candidates': candidates})
    print('FROZEN LONG-AGE ATTRIBUTION', len(candidates), sha256(destination), flush=True)

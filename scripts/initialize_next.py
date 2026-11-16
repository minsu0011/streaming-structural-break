"""ID-only seal partition and preregistration for the independent NEXT session."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation.splits import FrozenSplit
from src.utils.artifacts import sha256, write_json, utc_now


def frozen_json(path, value):
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old != value:
            raise RuntimeError(f'Refusing to replace frozen file: {path.name}')
    else:
        write_json(path, value)


def main():
    split = FrozenSplit(ROOT / 'artifacts/validation')
    seal = sorted(i for i, (role, _) in split.mapping.items() if role == 'LOCAL_FINAL_SEAL')
    dev = sorted(set(split.mapping) - set(seal))
    assert len(seal) == 2000 and len(dev) == 8000
    salt = 'structural-break-next-seal-500x4-v1-20260909'
    ordered = sorted(seal, key=lambda i: (hashlib.sha256(f'{salt}:{i}'.encode('ascii')).digest(), i))
    partitions = {}
    # The first generated research artifacts use assignment IDs only.
    for j, name in enumerate('ABCD'):
        ids = ordered[500*j:500*(j+1)]
        path = ROOT / f'SEAL_{name}_IDS.txt'
        body = ''.join(f'{i}\n' for i in sorted(ids)).encode('ascii')
        if path.exists() and path.read_bytes() != body:
            raise RuntimeError('Existing seal partition differs')
        if not path.exists():
            path.write_bytes(body)
        partitions[f'SEAL_{name}'] = {'count': 500, 'ids_file': path.name,
                                      'ids_sha256': sha256(path), 'opened': False}
    frozen_json(ROOT / 'SEAL_PARTITION_500x4_LOCK.json', {
        'status': 'LOCKED', 'salt': salt,
        'algorithm': 'sort by SHA256(ASCII(salt + colon + integer_dataset_id)), ID tie-break; contiguous chunks of 500; files sorted numerically, LF ASCII',
        'original_assignments_sha256': split.digest,
        'original_seal_count': len(seal), 'dev_count': len(dev),
        'inputs': ['frozen assignment dataset_id and role only'],
        'labels_tau_features_predictions_used': False,
        'seal_union_preserved': True, 'partitions': partitions,
        'opening_permitted_this_session': False})
    out = ROOT / 'artifacts/next'
    out.mkdir(exist_ok=True)
    (ROOT / 'checkpoints').mkdir(exist_ok=True)
    spec = ROOT / 'docs/USER_SPECIFICATION_NEXT.md'
    source = Path(r'C:\Users\minsu\.codex\attachments\1bae348a-e00b-43b6-9b51-e8973622dcac\pasted-text.txt')
    if not spec.exists():
        shutil.copyfile(source, spec)
    session_path = out / 'SESSION_NEXT.json'
    if not session_path.exists():
        started = datetime(2026, 9, 8, 17, 10, 1, tzinfo=timezone.utc)
        protected = ['artifacts/validation/REAL_SESSION.json', 'artifacts/real/RESEARCH_STATE.json',
                     'DEV_CHAMPION_CANDIDATES.json', 'configs/deployment.json',
                     'REDUCED_TEST_USAGE_LEDGER.json', 'artifacts/validation/GROUP_ASSIGNMENTS.parquet']
        protected += [str(p.relative_to(ROOT)).replace('\\', '/') for directory in
                      ['src/features', 'src/models', 'src/streaming']
                      for p in (ROOT / directory).rglob('*.py')]
        frozen_json(session_path, {
            'session': 'NEXT_20260909', 'research_start_utc': started.isoformat(),
            'minimum_research_seconds': 36000,
            'research_earliest_close_utc': (started + timedelta(hours=10)).isoformat(),
            'additional_final_audit_min_seconds': 2700,
            'additional_final_audit_target_seconds': 3300,
            'created_utc': utc_now(), 'specification_sha256': sha256(spec),
            'old_reference_hashes': {p: sha256(ROOT/p) for p in protected if (ROOT/p).is_file()},
            'git_commit_at_start': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'reduced_previous_uses': 2, 'reduced_new_uses_permitted': 0,
            'public_submissions_permitted': 0, 'seal_openings_permitted': 0,
            'idle_sleep_counts_as_research': False})
    # Splits, screening, regime and mechanism rules are fixed before any NEXT score.
    alternates = []
    for seed in [202609091, 202609092, 202609093]:
        assignments = {str(i): int.from_bytes(hashlib.sha256(
            f'structural-break-next-robustness:{seed}:{split.groups[i]}'.encode('ascii')).digest(), 'big') % 5 for i in dev}
        alternates.append({'seed': seed, 'fold_by_dataset_id': assignments})
    screen_ids = sorted(dev, key=lambda i: hashlib.sha256(f'next-screen-v1:20260909:{i}'.encode('ascii')).digest())[:3000]
    frozen_json(ROOT / 'ROBUSTNESS_SPLIT_LOCK.json', {
        'status': 'LOCKED', 'authority': 'original primary 5-fold remains selection authority; alternate splits diagnostic only',
        'original_assignments_sha256': split.digest, 'algorithm': 'SHA256 of literal prefix, fixed seed, historical duplicate group representative modulo 5',
        'alternates': alternates, 'seed_shopping_permitted': False,
        'screen': {'ids': sorted(screen_ids), 'id_selection_salt': 'next-screen-v1:20260909',
                   'folds': [0, 4], 'primary_fold_assignments_preserved': True,
                   'minimum_mean_delta_for_full': -0.002,
                   'maximum_admitted_exploratory_failed_screen_candidates': 0},
        'bootstrap': {'unit': 'whole series within original fold', 'seed': 202609094,
                      'screen_replicates': 500, 'final_replicates': 2000,
                      'metric': 'exact pair-count weighted time-stratified AUC per fold, arithmetic mean of five folds',
                      'paired_common_multiplicities': True, 'conditional_fixed_oof': True},
        'historical_regime': {'k': 5, 'seed': 202609095, 'n_init': 10,
                              'standardize': 'DEV historical descriptor mean/std; clip standardized descriptors to [-8,8]',
                              'clustering': 'KMeans algorithm lloyd; deterministic single-thread fit',
                              'stress': 'leave one cluster out, train four; global H-only grouping is transductive diagnostic'},
        'mechanism': {'window': 128, 'minimum_each_side': 32,
                      'mean_effect_threshold': 0.5, 'absolute_log_sd_ratio_threshold': 0.4,
                      'max_acf_difference_threshold': 0.2, 'tail_exceedance_rate_difference_threshold': 0.08,
                      'acf_lags': [1, 2, 4, 8], 'tail_cutoff_pre_robust_scale': 2.5,
                      'mixed_rule': 'two or more threshold-normalized effects >=1 and largest <1.5 times second largest',
                      'weak_rule': 'insufficient windows or every normalized effect <1',
                      'dominant_rule': 'largest normalized effect otherwise; deterministic order mean, scale, dependence, tail',
                      'training_or_feature_use_forbidden': True},
        'promotion': {'mean_delta_min': 0.0015, 'median_delta_min': 0.0,
                      'worst_delta_min': -0.0015, 'bootstrap_probability_min': 0.85,
                      'alternate_nonnegative_count_min': 2, 'historical_worst_delta_min': -0.005,
                      'mechanism_delta_min_on_supported_category': -0.01,
                      'mechanism_min_break_series': 100,
                      'point_latency_ratio_max': 3.0, 'amortized_latency_ratio_max': 3.0},
        'new_generation': {'unconditional_mean_delta_min': 0.004,
                           'conditional_mean_delta_min': 0.003, 'bootstrap_probability_min': 0.95,
                           'clear_robustness_gain': 'all 3 alternate deltas nonnegative and alternate mean >=0.0015; regime worst nonworse',
                           'independent_mechanism_gain': 'matched ablation of a new statistic has primary mean delta >=0.001 and improves at least one supported mechanism by >=0.003',
                           'requires_all_promotion_gates': True, 'seal_opening_still_forbidden': True},
        'feature_grids': {'ar_order': [1, 2, 4, 8, 12], 'glr_ages': [1, 2, 4, 8, 16, 32, 64, 128],
                         'memory_half_lives': [8, 32, 128], 'cdf_bins': [8, 16],
                         'cross_order_pairs': [[4, 8], [8, 12]],
                         'late_ensemble_weights': [[0.5, 0.5], [0.67, 0.33], [0.33, 0.67]]}})
    print(json.dumps({'seal_partition': 'LOCKED_500x4', 'opened': [False]*4,
                      'session': str(session_path), 'robustness_policy': 'LOCKED'}, indent=2))


if __name__ == '__main__':
    main()

"""Post-release controlled-coordinate checks of unchanged discovery gates.

These synthetic sensors are not members of the competition feature bank. The
experiment checks the core gate's reachability, not the power of the complete
153-feature/H-context procedure or a new RP candidate.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
from src.next8.cache import FeatureCache
from src.next8.data import FitScope
from src.next8.common import BASE, read_json, digest, atomic, utc_now

WORLDS = ('NULL', 'MONOTONE', 'FIXED_U', 'OWN_STATE_REVERSAL')
RHOS = (0.0, 0.9)
SEEDS = (8101611, 8101629)
BOUND = 0.8416212335729143


def cases():
    return [(world, rho, seed) for world in WORLDS for rho in RHOS for seed in SEEDS]


def coordinate(noise, labels, signs, world):
    """A simulator may know its regime; the gate receives observations only."""
    if world not in WORLDS:
        raise ValueError('Unregistered controlled world')
    noise = np.asarray(noise, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    signs = np.asarray(signs)
    assert noise.ndim == 2 and noise.shape == labels.shape == signs.shape
    assert np.isfinite(noise).all() and np.isin(labels, (0, 1)).all()
    assert np.isin(signs, (-1, 1)).all()
    value = np.empty(noise.shape, dtype=np.float32)
    previous = np.zeros(len(noise), dtype=np.float32)
    for t in range(noise.shape[1]):
        z, y = noise[:, t], labels[:, t]
        if world == 'NULL':
            raw = z
        elif world == 'MONOTONE':
            raw = z + 2.0 * y
        elif world == 'FIXED_U':
            raw = signs[:, t] * (np.abs(z) + 2.0 * y)
        else:
            coefficient = np.where(previous < -BOUND, 1., np.where(previous > BOUND, -1., 0.))
            raw = z + 2.0 * y * coefficient
        value[:, t] = np.clip(raw, -12., 12.)
        previous = value[:, t]
    return value


def controlled_cache(world, rho, seed, *, series=5120, length=512):
    """No FeatureCache constructor, research loader, raw file or RP metadata."""
    assert world in WORLDS and rho in RHOS and series >= 6 and length >= 17
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    noise_rng, label_rng, group_rng, sign_rng = streams
    shocks = noise_rng.normal(size=(length + 16, series))
    last = noise_rng.normal(size=series)
    noise = np.empty((series, length + 16), dtype=np.float64)
    for t in range(length + 16):
        last = rho * last + np.sqrt(1. - rho * rho) * shocks[t]
        noise[:, t] = last
    positive = label_rng.permutation(series)[:series // 2]
    tau = np.full(series, max(length, 512) + 1, dtype=np.int32)
    early = len(positive) // 10
    tau[positive[:early]] = 0
    tau[positive[early:]] = label_rng.integers(1, 400, len(positive) - early)
    online_y = (np.arange(length)[None, :] >= tau[:, None]).astype(np.uint8)
    labels = np.column_stack((np.zeros((series, 16), dtype=np.uint8), online_y))
    signs = np.where(sign_rng.integers(0, 2, size=noise.shape), 1, -1)
    values = coordinate(noise, labels, signs, world)
    ids = np.arange(500000, 500000 + series, dtype=np.int64)
    permutation = group_rng.permutation(series)
    assignment = np.empty(series, dtype=np.int8)
    assignment[permutation] = np.arange(series) % 3
    h_lengths = np.asarray((1000, 1500, 2000, 3000, 4500))[group_rng.permutation(series) % 5]
    h_rho = 0.0 if world == 'FIXED_U' else rho
    cache = FeatureCache.__new__(FeatureCache)
    cache.values = values[:, 16:].reshape(-1, 1)
    cache.series_index = np.repeat(np.arange(series, dtype=np.int32), length)
    cache.series = {
        'dataset_id': ids, 'H_length': h_lengths,
        'seed_values': values[:, :16, None],
        'low': np.full((series, 1), -BOUND), 'high': np.full((series, 1), BOUND),
        'valid': np.ones((series, 1), dtype=bool),
        'context': np.column_stack((np.log1p(h_lengths), np.zeros(series), np.full(series, h_rho),
                                   np.full(series, h_rho ** 4), np.full(series, .002699796063260207), np.zeros(series))).astype(np.float32),
    }
    cache.meta = {'dataset_id': np.repeat(ids, length),
                  'time_online': np.tile(np.arange(length, dtype=np.int32), series),
                  'target': online_y.reshape(-1)}
    # Nonmaterialized exclusion sentinel satisfies the scope type's required
    # nonempty validation set. It has no observation, label, fit or score.
    sentinel = frozenset((999999999,))
    prefix = 'CONTROL_' + world + '_rho' + str(rho) + '_seed' + str(seed)
    parent = FitScope(frozenset(map(int, ids)), sentinel, sentinel, prefix)
    scopes = []
    for k in range(3):
        valid = frozenset(map(int, ids[assignment == k]))
        scopes.append(FitScope(parent.training_ids - valid, valid, sentinel, prefix + '_inner' + str(k)))
    return SimpleNamespace(cache=cache, parent=parent, scopes=scopes, assignment=assignment,
                           positive_series_count=len(positive), zero_time_changes=early)


@contextmanager
def single_control_coordinate(world):
    """Temporary globals in this isolated diagnostic process, never RP workers."""
    import src.next8.screening as screening
    before = screening.NAMES, screening.DETAILED
    screening.NAMES, screening.DETAILED = ('ARTIFICIAL_CONTROL_' + world,), (0,)
    try:
        yield screening
    finally:
        screening.NAMES, screening.DETAILED = before


def summarize_case(case_index, fixture, panels, corrections, directory):
    world, rho, seed = cases()[case_index]
    with single_control_coordinate(world) as screening:
        selection = screening.summarize_screen(panels, corrections, fixture.parent)
    detail = []
    for k, panel in enumerate(panels):
        r = panel[0]
        conditional = r['conditional']
        failed = [{key: item.get(key) for key in ('mode', 'domain_type', 'domain', 'TS_AUC', 'ci_low', 'ci_high',
                                                 'positive_series', 'negative_series', 'support_status')}
                  for item in r['atlas'] if item['support_status'] != 'SUPPORTED' or item['ci_low'] <= .47]
        detail.append({'inner': k, 'invariant_pass': r['invariant_direction_supported'],
                       'fixed_nonlinear_pass': r['fixed_nonlinear_supported'], 'scores': r['scores'],
                       'coverage': r['domain_coverage'], 'nonlinear_coverage': r['nonlinear_domain_coverage'],
                       'all_converged': r['all_converged'], 'failed_or_sparse_domains': failed,
                       'conditional': conditional, 'corrections': corrections[k],
                       'statistical_fixture_fits': 3 + (4 if conditional['status'] == 'COMPLETE' else 0)})
    result = {'status': 'COMPLETE', 'timestamp': utc_now(), 'case_index': case_index,
              'world': world, 'noise_rho': rho, 'seed': seed, 'series': len(fixture.parent.training_ids),
              'positive_series_count': fixture.positive_series_count, 'changes_at_zero': fixture.zero_time_changes,
              'selection': selection, 'inner_results': detail,
              'statistical_fixture_fits': sum(r['statistical_fixture_fits'] for r in detail),
              'competition_RP_fits': 0, 'competition_RP_predictions_or_scores': 0,
              'full_raw_feature_or_H_normalizer_replay': False,
              'single_coordinate_multiplicity_not_full153_feature_family': True,
              'H_context_second_stage_selection_not_replayed': True,
              'nonmaterialized_exclusion_sentinel': 999999999,
              'not_an_empirical_power_curve_or_false_positive_rate_estimate': True,
              'new_scientific_candidate_or_model_promotion': False,
              'policy_sha256': digest(BASE / 'GATE_REACHABILITY_CONTROL_POLICY_NEXT8.json'),
              'inner_DONE_sha256': {str(k): digest(directory / ('inner' + str(k)) / 'DONE.json') for k in range(3)}}
    atomic(directory / 'DONE.json', result)
    return result

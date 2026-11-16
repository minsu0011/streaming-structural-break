import copy
import numpy as np
import pytest
from src.next7.pairwise import make_pair_plan, plan_hash, objective, fit_pairwise, PairwiseCorrection

PROVENANCE = {'strict_nested_OOF': True, 'outer_validation_excluded': True}


def test_exhaustive_plan_and_time_pair_mass_weights():
    y = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    t = np.array([0, 0, 1, 1, 1, 1, 2, 3])
    plan = make_pair_plan(y, t, np.arange(len(y)))
    expected = {(0, 1), (2, 3), (2, 4), (5, 3), (5, 4)}
    assert set(zip(plan['positive_row'], plan['negative_row'])) == expected
    assert plan['total_pair_mass'] == 5
    np.testing.assert_array_equal(plan['weights'], np.full(5, .2))
    assert np.all(t[plan['positive_row']] == t[plan['negative_row']])


def test_sampled_plan_is_deterministic_and_each_time_has_correct_mass():
    y = np.tile([0, 0, 1, 1], 3)
    t = np.repeat([0, 1, 2], 4)
    ids = np.tile(np.arange(4), 3)
    a = make_pair_plan(y, t, ids, cap=2)
    b = make_pair_plan(y, t, ids, cap=2)
    assert plan_hash(a) == plan_hash(b)
    for at in range(3):
        assert a['weights'][a['time'] == at].sum() == pytest.approx(1 / 3)
    assert np.all(ids[a['positive_row']] != ids[a['negative_row']])


def test_analytic_gradient_matches_central_difference():
    differences = np.sin(np.arange(80).reshape(20, 4) * .13)
    offsets = np.linspace(-4, 4, 20)
    weights = np.arange(1., 21.)
    weights /= weights.sum()
    beta = np.array([.2, -.3, .04, .5])
    _, grad = objective(beta, differences, offsets, weights)
    actual = []
    for j in range(4):
        delta = np.zeros(4)
        delta[j] = 1e-6
        actual.append((objective(beta + delta, differences, offsets, weights)[0]
            - objective(beta - delta, differences, offsets, weights)[0]) / 2e-6)
    np.testing.assert_allclose(actual, grad, atol=1e-8, rtol=1e-7)


def test_zero_within_time_contrast_has_no_ranking_gradient():
    _, gradient = objective(np.zeros(10), np.zeros((20, 10)), np.linspace(-10, 10, 20), np.full(20, .05))
    np.testing.assert_array_equal(gradient, np.zeros(10))


def test_invalid_same_history_time_and_cross_time_only_labels_rejected():
    with pytest.raises(ValueError):
        make_pair_plan([0, 1], [0, 0], [1, 1])
    with pytest.raises(ValueError):
        make_pair_plan([0, 1], [0, 1], [1, 2])


def sample():
    i = np.arange(100)
    features = np.sin((i[:, None] + 1) * np.arange(1, 11)[None, :] * .033)
    times, series = i // 10, i % 10
    target = ((times + series) % 2).astype(int)
    return features, np.full(100, .5), target, times, series


def test_fit_is_reproducible_scalar_batch_equal_and_sources_validated(tmp_path):
    values = sample()
    a = fit_pairwise(*values, PROVENANCE)
    b = fit_pairwise(*values, PROVENANCE)
    np.testing.assert_array_equal(a.beta, b.beta)
    batch = a.predict(values[1], values[0])
    scalar = np.array([a.predict_one(p, x) for p, x in zip(values[1], values[0])])
    np.testing.assert_allclose(scalar, batch, atol=2e-15, rtol=0)
    assert a.training['objective'] <= a.training['initial_objective']
    path = tmp_path / 'pairwise.joblib'
    a.save(path)
    loaded = PairwiseCorrection.load(path)
    np.testing.assert_array_equal(loaded.predict(values[1], values[0]), batch)
    loaded.pairwise_source_sha256 = 'wrong'
    with pytest.raises(RuntimeError):
        loaded.validate()


def test_provenance_and_frozen_pair_plan_checked_before_fit():
    values = sample()
    with pytest.raises(RuntimeError):
        fit_pairwise(*values, {})
    plan = make_pair_plan(values[2], values[3], values[4])
    changed = copy.deepcopy(plan)
    changed['weights'][0] += .1
    with pytest.raises(RuntimeError):
        fit_pairwise(*values, PROVENANCE, frozen_plan=changed)


def test_standardization_exactly_matches_frozen_G2_parent():
    from src.next7.model import fit_correction
    values = sample()
    parent = fit_correction('M_FULL', values[0], values[1], values[2], np.ones(len(values[2])), PROVENANCE)
    child = fit_pairwise(*values, PROVENANCE)
    np.testing.assert_array_equal(child.center, parent.center)
    np.testing.assert_array_equal(child.spread, parent.spread)

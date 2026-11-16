import numpy as np
import pytest
from src.next5.fitted_fold_summary import fixed_artifact_mean, describe_fold_values


def test_common_seed_variation_is_not_divided_by_number_of_folds():
    variation = np.r_[0., np.linspace(-.04, .04, 2000)]
    old = {f: np.full(2001, .6) for f in range(5)}
    new = {f: .61 + variation for f in range(5)}
    report = fixed_artifact_mean(old, new, range(5))
    single = fixed_artifact_mean({0: old[0]}, {0: new[0]}, [0])
    assert report['estimate_mean_delta'] == pytest.approx(.01)
    for key in ('ci_2_5', 'ci_97_5'):
        assert report[key] == pytest.approx(single[key])
    assert report['positive_artifacts'] == 5


def test_fold_heterogeneity_is_retained_without_independent_fold_interval():
    rows = [dict(candidate='C', scope='screen', fold=f, metric='excess', delta=d)
            for f, d in [(0, -.2), (4, .1)]]
    report = describe_fold_values(rows, ['candidate', 'scope', 'metric'], 'delta', {('C', 'screen'): [0, 4]})[0]
    assert report['mean_across_fixed_artifacts'] == pytest.approx(-.05)
    assert report['minimum_across_fixed_artifacts'] == -.2
    assert report['maximum_across_fixed_artifacts'] == .1
    assert report['negative_artifacts'] == report['positive_artifacts'] == 1
    assert report['independent_fold_uncertainty_claimed'] is False


@pytest.mark.parametrize('folds', [[0], [0, 0], [0, 1]])
def test_missing_duplicate_or_wrong_screen_folds_rejected(folds):
    rows = [dict(candidate='C', scope='screen', fold=f, delta=.1) for f in folds]
    with pytest.raises(ValueError):
        describe_fold_values(rows, ['candidate', 'scope'], 'delta', {('C', 'screen'): [0, 4]})


def test_changed_draw_count_rejected():
    with pytest.raises(ValueError):
        fixed_artifact_mean({0: np.full(2002, .5)}, {0: np.full(2002, .6)}, [0])

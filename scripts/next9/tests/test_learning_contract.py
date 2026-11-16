import src.next9
import numpy as np
import pytest
from src.next9.learning import fit_classifier
from src.next9.data import FitScope


def test_classifier_no_outer_feedback():
    rng = np.random.default_rng(9101604)
    x = rng.normal(size=(640, 3))
    ids = np.repeat(np.arange(64), 10)
    y = (x[:, 0] > 0).astype(int)
    scope = FitScope(frozenset(range(64)), frozenset(range(64, 128)), frozenset(range(128, 192)), 'toy')
    model = fit_classifier(x, y, ids, ids, scope)
    before = model.predict_proba(x[:20])
    unrelated_validation_targets = 1-y
    assert len(unrelated_validation_targets) == len(y)
    np.testing.assert_array_equal(before, model.predict_proba(x[:20]))
    with pytest.raises(ValueError, match='outside'):
        fit_classifier(x, y, ids+64, ids+64, scope)
    coef, intercept = model.linear_parameters()
    from scipy.special import expit
    np.testing.assert_allclose(expit(x@coef+intercept), model.predict_proba(x)[:, 1], atol=1e-12)


def test_constant_and_duplicate_columns_cannot_create_fake_capacity():
    x = np.tile(np.arange(10), 64).astype(float)[:, None]
    ids = np.repeat(np.arange(64), 10)
    y = (x[:, 0] > 5).astype(int)
    scope = FitScope(frozenset(range(64)), frozenset(range(64, 128)), frozenset(), 'toy')
    base = fit_classifier(x, y, ids, ids, scope)
    padded = np.column_stack([x, x, np.ones(len(x))])
    expanded = fit_classifier(padded, y, ids, ids, scope)
    np.testing.assert_allclose(base.predict_proba(x), expanded.predict_proba(padded), atol=1e-12)
    constant = fit_classifier(np.ones((len(x), 3)), y, ids, ids, scope)
    assert constant.training['estimator_fit_calls'] == 0
    assert np.ptp(constant.predict_proba(np.ones((5, 3)))[:, 1]) == 0


def test_missing_transition_class_preserves_fixed_vocabulary():
    x = np.tile(np.arange(10), 64).astype(float)[:, None]
    ids = np.repeat(np.arange(64), 10)
    y = np.where(x[:, 0] > 5, 2, 0)
    scope = FitScope(frozenset(range(64)), frozenset(range(64, 128)), frozenset(), 'toy')
    model = fit_classifier(x, y, ids, ids, scope, classes=4)
    p = model.predict_proba(x)
    assert p.shape == (len(x), 4)
    np.testing.assert_allclose(p.sum(axis=1), 1)

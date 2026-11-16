import numpy as np
import pytest
from src.next9.data import FitScope
from src.next9.movement_screen import fit_ridge_bundle, paired_group_summary, group_weights


def test_auxiliary_fit_excludes_prediction_groups():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(120,9));y=rng.normal(size=(120,15))
    ids=np.repeat(np.arange(12),10)
    scope=FitScope(frozenset(range(8)),frozenset(range(8,12)),frozenset(range(8,12)),'toy')
    model=fit_ridge_bundle(x[:80],y[:80],ids[:80],ids[:80],scope)
    before=model.predict(x[80:])
    y[80:]=9999
    other=fit_ridge_bundle(x[:80],y[:80],ids[:80],ids[:80],scope)
    np.testing.assert_array_equal(before,other.predict(x[80:]))
    with pytest.raises(ValueError):
        fit_ridge_bundle(x,y,ids,ids,scope)


def test_future_F_only_aux_target():
    rng=np.random.default_rng(19)
    x=rng.normal(size=(200,9));teacher=rng.normal(size=(9,15))
    y=x@teacher
    ids=np.repeat(np.arange(20),10)
    scope=FitScope(frozenset(range(16)),frozenset(range(16,20)),frozenset(range(16,20)),'toy')
    model=fit_ridge_bundle(x[:160],y[:160],ids[:160],ids[:160],scope)
    assert model.predict(x[160:]).shape==(40,15)
    assert model.record()['auxiliary_targets_are_outputs_not_predictor_inputs']
    with pytest.raises(ValueError):
        model.predict(np.c_[x[160:],y[160:]])


def test_original_group_weighting_not_row_count_weighting():
    groups=np.array([1,1,1,1,2])
    weights=group_weights(groups)
    assert weights[:4].sum()==pytest.approx(weights[4])
    reference=np.array([1,1,1,1,0.])
    candidate=np.array([0,0,0,0,2.])
    d=paired_group_summary(reference,candidate,groups)
    assert d['groups']==2
    assert d['mean_gain']==-.5


def test_forecast_bundle_has_no_tau_or_break_age_predictor():
    from src.next9.movement_screen import DESIGN_NAMES
    assert len(DESIGN_NAMES)==9
    assert not any('tau' in n or 'break' in n or 'future' in n or 'target' in n for n in DESIGN_NAMES)

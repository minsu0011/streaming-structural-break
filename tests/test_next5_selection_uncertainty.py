import numpy as np
from src.next5.selection_uncertainty import summarize_catalog


def test_identical_models_receive_fractional_tie_credit_without_new_information():
    values=np.array([[.001,.001],[.002,.002],[0.,0.],[.0015,.0015]])
    result=summarize_catalog(values,['a','b'],['S','P'])
    assert [row['fixed_OOF_bootstrap_winner_fraction'] for row in result['candidates']]==[.5,.5]
    assert result['winner_gap']==0 and result['fixed_pair_gap_ci_lower']==0 and result['fixed_pair_gap_ci_upper']==0
    assert result['nominal_adaptive_familywise_coverage_claimed'] is False


def test_simultaneous_envelope_contains_each_centered_error_for_at_least_95_percent_of_draws():
    rng=np.random.default_rng(20260908);truth=np.array([.001,.0008,-.0002])
    values=np.vstack([truth,truth+rng.normal(0,.001,(2000,3))])
    result=summarize_catalog(values,['a','b','c'],['S','S','M']);radius=result['catalog_envelope_radius']
    assert np.mean(np.max(np.abs(values[1:]-truth),axis=1)<=radius)>=.95
    assert abs(sum(row['fixed_OOF_bootstrap_winner_fraction'] for row in result['candidates'])-1)<1e-12
    assert abs(sum(row['fraction_catalog_winner_from_family'] for row in result['families'])-1)<1e-12

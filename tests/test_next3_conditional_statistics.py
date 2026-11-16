import numpy as np
import pytest
from src.next3.conditional_statistics import load_saved_aggregate, summarize_saved_contrasts


def toy():
    return np.asarray([[.25,.25],[0.,0.],[0.,0.],[0.,0.],[1.,1.]],dtype=np.float64)


def test_draw_only_sample_sd_strict_probability_and_linear_quantiles_have_hand_values():
    result = summarize_saved_contrasts(toy(),['A','B']); row = result['rows'][0]
    assert result['replicates'] == 4 and result['observed_row_excluded_from_draw_summaries']
    assert row['estimate_delta'] == .25 and row['conditional_standard_error'] == .5
    assert row['probability_delta_positive'] == .25
    assert row['conditional_percentile_ci_2_5'] == 0.
    assert row['conditional_percentile_ci_97_5'] == pytest.approx(.925)
    assert result['critical_value'] == pytest.approx(1.35)
    assert row['joint_conditional_envelope_lower'] == pytest.approx(-.425)
    assert row['joint_conditional_envelope_upper'] == pytest.approx(.925)
    assert row['selection_adjusted'] is False and result['new_draws'] == 0


def test_cross_candidate_draw_pairing_changes_joint_envelope_without_changing_marginals():
    original = summarize_saved_contrasts(toy(),['A','B'])
    changed = toy(); changed[1:,1] = changed[[4,1,2,3],1]
    result = summarize_saved_contrasts(changed,['A','B'])
    assert result['critical_value'] == pytest.approx(1.5)
    for row,reference in zip(result['rows'],original['rows']):
        for key in ('estimate_delta','conditional_standard_error','conditional_percentile_ci_2_5',
                    'conditional_percentile_ci_97_5','probability_delta_positive'):
            assert row[key] == reference[key]


@pytest.mark.parametrize('damage',['float32','nan','infinite','out_of_range','too_few_rows','wrong_columns','zero_draw_variance'])
def test_invalid_or_degenerate_saved_contrasts_are_rejected(damage):
    values = toy()
    if damage == 'float32': values = values.astype(np.float32)
    elif damage == 'nan': values[1,0] = np.nan
    elif damage == 'infinite': values[1,0] = np.inf
    elif damage == 'out_of_range': values[1,0] = 1.01
    elif damage == 'too_few_rows': values = values[:2]
    elif damage == 'wrong_columns': values = values[:,:1]
    elif damage == 'zero_draw_variance': values[1:] = 0.
    with pytest.raises(ValueError): summarize_saved_contrasts(values,['A','B'])


@pytest.mark.parametrize('names',[['A','A'],['A',''],['A',None],'AB'])
def test_candidate_names_cannot_be_ambiguous(names):
    with pytest.raises(ValueError): summarize_saved_contrasts(toy(),names)


def test_saved_archive_loads_only_exact_registered_aggregate_shapes(tmp_path):
    path = tmp_path/'aggregate.npz'
    np.savez(path,contrasts=toy(),candidates=np.asarray(['A','B']),max_abs_standardized_error=np.asarray([.5,.5,.5,1.5]))
    result = load_saved_aggregate(path,['A','B'],replicates=4)
    assert result['candidate_count'] == 2 and result['replicates'] == 4


@pytest.mark.parametrize('damage',['extra','name_order','wrong_dimensions','object_names','wrong_max_length','changed_max'])
def test_saved_aggregate_schema_or_pairing_damage_is_rejected(tmp_path,damage):
    values = {'contrasts':toy(),'candidates':np.asarray(['A','B']),
        'max_abs_standardized_error':np.asarray([.5,.5,.5,1.5])}
    if damage == 'extra': values['unregistered'] = np.asarray([0.])
    elif damage == 'name_order': values['candidates'] = values['candidates'][::-1]
    elif damage == 'wrong_dimensions': values['contrasts'] = toy()[1:]
    elif damage == 'object_names': values['candidates'] = np.asarray(['A','B'],dtype=object)
    elif damage == 'wrong_max_length': values['max_abs_standardized_error'] = np.asarray([.5,.5,1.5])
    elif damage == 'changed_max': values['max_abs_standardized_error'][0] = .25
    path = tmp_path/'aggregate.npz'; np.savez(path,**values)
    with pytest.raises(ValueError): load_saved_aggregate(path,['A','B'],replicates=4)

import copy
import pytest
from src.next8.computation_audit import aggregate_series, finite_error, FLOAT32_EPS


def valid_case(sid=5, fold=2, points=410):
    modes = {}
    for mode, count in [('NORMAL', points), ('EARLY', max(0, points - 400))]:
        modes[mode] = {'baseline_points': count, 'baseline_float64_max_error': 0. if count else None,
            'M6_fixed_composition_source_verified': True, 'M6_replay_composition_max_abs': 0. if count else None,
            'M6_eligible': False, 'head_results': {m: {'saved_points': count,
                'maximum_float32_error': 0. if count else None, 'bitwise_equal': True if count else None}
                for m in ('M1', 'M2', 'M3', 'M4', 'M5')}}
    return {'status': 'PASS', 'series_id': sid, 'outer_fold': fold, 'online_points': points,
        'normalized_features': 153, 'fresh_scalar_kernel_feature_cache_bitwise_equal': True,
        'head_features_and_graph_states_built_from_fresh_H_and_scalar_online_kernel': True,
        'saved_feature_cache_used_only_as_comparison_target': True, 'break_labels_or_tau_as_inference_input': False,
        'new_model_fits': 0, 'all_H_metadata_equal': ['center', 'spread', 'fit_length', 'epsilon', 'seed_values', 'valid', 'reference_sha256'],
        'full_H_original_reference_feature_parity': {k: {'bitwise_equal': True} for k in ('J102', 'C205')}, 'predictions': modes}


def test_full_replay_rejects_duplicate_omitted_wrong_fold_and_wrong_early_coverage():
    a, b = valid_case(), valid_case(9, 3, 200)
    result = aggregate_series([a, b], {5: 2, 9: 3}, 610)
    assert result['unique_RP_series'] == 2
    assert result['normal_and_early_replay']['EARLY']['saved_points_per_head'] == 10
    with pytest.raises(AssertionError):
        aggregate_series([a, a], {5: 2, 9: 3}, 820)
    with pytest.raises(AssertionError):
        aggregate_series([a], {5: 2, 9: 3}, 410)
    with pytest.raises(AssertionError):
        aggregate_series([a, b], {5: 2, 9: 4}, 610)
    wrong = copy.deepcopy(a)
    wrong['predictions']['EARLY']['baseline_points'] = 11
    with pytest.raises(AssertionError):
        aggregate_series([wrong, b], {5: 2, 9: 3}, 610)


def test_prediction_tolerance_does_not_turn_nonfinite_or_missing_into_pass():
    assert finite_error(FLOAT32_EPS, 1, FLOAT32_EPS) == FLOAT32_EPS
    for value in (float('nan'), float('inf'), -1., None):
        with pytest.raises(AssertionError):
            finite_error(value, 1, FLOAT32_EPS)
    with pytest.raises(AssertionError):
        finite_error(1e-12, 1, 1e-12, strict=True)
    with pytest.raises(AssertionError):
        finite_error(0., 0, FLOAT32_EPS)
    assert finite_error(None, 0, FLOAT32_EPS) == 0.

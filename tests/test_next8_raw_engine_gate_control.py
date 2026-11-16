import numpy as np
from src.next8.raw_engine_gate_control import sample_plan, raw_case, extract_case, BANKS
from src.next8.features import FeatureState


def test_raw_control_labels_groups_and_persistent_interventions_are_distinct():
    plan = sample_plan(120)
    assert sum(plan['tau'] <= 512) == 60 and sum(plan['tau'] == 0) == 6
    assert min(plan['dataset_id']) == 900000
    assert len(set(plan['dataset_id'])) == 120
    assert np.bincount(plan['inner_assignment']).tolist() == [40, 40, 40]
    h, variants = raw_case(4, 1000, 33, length=100)
    hh, short = raw_case(4, 1000, 33, length=50)
    np.testing.assert_array_equal(h, hh)
    for bank in BANKS:
        np.testing.assert_array_equal(variants[bank][:33], variants['NULL'][:33])
        np.testing.assert_array_equal(variants[bank][:50], short[bank])
    np.testing.assert_array_equal(variants['MEAN_PLUS2'][33:], variants['NULL'][33:] + np.float32(2))
    np.testing.assert_array_equal(variants['SCALE_SD2'][33:], variants['NULL'][33:] * np.float32(2))


def test_actual_H_feature_engine_control_prefix_and_snapshot_match_scalar(monkeypatch):
    import src.next8.cache as cache_module
    def forbidden():
        raise AssertionError('Raw synthetic control accessed competition loader')
    monkeypatch.setattr(cache_module, 'source', forbidden)
    h, variants = raw_case(5, 1000, 30, length=100)
    initial, features = extract_case(h, variants)
    assert initial.counter == 0
    assert initial.normalizer_fit_sha256 == initial.calibration_reference_sha256 == initial.online_reference_sha256
    for bank in BANKS:
        scalar = FeatureState.restore(initial)
        actual = np.asarray([scalar.update(x)[0] for x in variants[bank][:50]])
        np.testing.assert_array_equal(actual, features[bank][:50])
        changed = variants[bank].copy()
        changed[50:] = -7 * changed[50:] + .375
        np.testing.assert_array_equal(FeatureState.restore(initial).replay(changed)[0][:50], features[bank][:50])
        np.testing.assert_array_equal(FeatureState.restore(initial).replay(variants[bank][:50])[0], features[bank][:50])

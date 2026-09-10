from copy import deepcopy
import pytest
from scripts.collect_next_raw_reproduction import compare_all_dev


def records():
    original = {'status': 'PASS', 'model_sha256': 'model', 'all_dev_prediction_sha256': 'predictions',
        'training': {'training_series': 8000, 'seal_rows_fitted': 0, 'feature_matrix_sha256': 'matrix',
                     'ordered_targets_sha256': 'target', 'ordered_ids_sha256': 'ids'},
        'fingerprint_points': 4014405, 'training_quality_score_computed': False}
    independent = deepcopy(original)
    independent['all_dev_fingerprint_points'] = independent.pop('fingerprint_points')
    return original, independent


def test_reproduction_requires_training_coordinates_and_feature_identity():
    original, independent = records()
    assert compare_all_dev(original, independent)['model_sha256'] == 'model'
    for field in ('feature_matrix_sha256', 'ordered_targets_sha256', 'ordered_ids_sha256'):
        altered = deepcopy(independent)
        altered['training'][field] = 'changed'
        with pytest.raises(RuntimeError, match='training'):
            compare_all_dev(original, altered)


def test_reproduction_refuses_partial_or_scored_training_fingerprint():
    original, independent = records()
    for field, value in [('all_dev_fingerprint_points', 4014404), ('training_quality_score_computed', True), ('model_sha256', 'changed')]:
        altered = deepcopy(independent)
        altered[field] = value
        with pytest.raises(RuntimeError):
            compare_all_dev(original, altered)

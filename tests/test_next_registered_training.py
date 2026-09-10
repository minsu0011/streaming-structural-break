from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from src.next.registered_final_fitting import fit_all_development, training_feature_names


def test_registered_fit_rejects_missing_chronological_rows():
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root/'configs/next_log_energy_candidates.json').read_text(encoding='utf-8'))['candidates'][0]
    names = training_feature_names(candidate)
    guard = SimpleNamespace(dev_ids={1, 2}, admit=lambda ids, **kwargs: ids,
        split=SimpleNamespace(assignment=lambda sid: ('DEVELOPMENT_POOL', sid % 5)))
    metadata = {'dataset_id': np.array([1, 1, 2, 2]), 'time_online': np.array([0, 2, 0, 1]),
                'target': np.array([0, 1, 0, 1])}
    with pytest.raises(ValueError, match='chronological'):
        fit_all_development(np.zeros((4, len(names)), dtype=np.float32), metadata, candidate, guard, names)


@pytest.mark.parametrize('family', ['registered', 'serial'])
def test_registered_official_training_skips_opaque_seal_before_value_access(monkeypatch, tmp_path, family):
    import importlib
    module = importlib.import_module('src.next.'+family+'_training')
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Seal array access')
        def __len__(self):
            raise AssertionError('Seal length access')
        def __int__(self):
            raise AssertionError('Seal target access')
        def __bool__(self):
            raise AssertionError('Seal truth-value access')
    split = SimpleNamespace(mapping={1: None, 2: None}, digest='split', assignment=lambda sid: ('DEVELOPMENT_POOL', 0))
    guard = SimpleNamespace(dev_ids={1}, seal_ids={2}, split=split)
    monkeypatch.setattr(module, 'SealGuard', lambda root: guard)
    monkeypatch.setattr(module, 'training_feature_names', lambda candidate: ('feature',))
    class State:
        def __init__(self, historical, *args):
            assert len(historical) == 32
        def update(self, point):
            return np.array([point], dtype=np.float32)
    monkeypatch.setattr(module, 'ExtendedFeatureState', State)
    class ReachedExactDevFit(Exception):
        pass
    def fit(design, metadata, candidate, actual_guard, names):
        np.testing.assert_array_equal(design[:, 0], [3., 4., 5.])
        np.testing.assert_array_equal(metadata['dataset_id'], [1, 1, 1])
        np.testing.assert_array_equal(metadata['time_online'], [0, 1, 2])
        assert actual_guard is guard
        raise ReachedExactDevFit
    monkeypatch.setattr(module, 'fit_all_development', fit)
    spec = {'development_ids': [1], 'known_training_ids': [1, 2], 'split_sha256': 'split',
            'status': 'FROZEN_NEXT_RESEARCH_REFIT', 'candidate': {'extension': {'kind': 'log_energy'}}}
    rows = [(2, Poison(), Poison(), Poison()), (1, np.arange(32), np.array([3., 4., 5.]), 1)]
    with pytest.raises(ReachedExactDevFit):
        module.train_frozen_candidate(iter(rows), tmp_path/'model', spec, root=tmp_path)


@pytest.mark.parametrize('sid', [True, 3, 1.0])
@pytest.mark.parametrize('family', ['registered', 'serial'])
def test_registered_training_rejects_unknown_or_nonintegral_ids_before_arrays(monkeypatch, tmp_path, sid, family):
    import importlib
    module = importlib.import_module('src.next.'+family+'_training')
    guard = SimpleNamespace(dev_ids={1}, seal_ids={2}, split=SimpleNamespace(mapping={1: None, 2: None}, digest='split'))
    monkeypatch.setattr(module, 'SealGuard', lambda root: guard)
    monkeypatch.setattr(module, 'training_feature_names', lambda candidate: ('feature',))
    spec = {'development_ids': [1], 'known_training_ids': [1, 2], 'split_sha256': 'split',
            'status': 'FROZEN_NEXT_RESEARCH_REFIT', 'candidate': {}}
    with pytest.raises(RuntimeError, match='ID'):
        module.train_frozen_candidate(iter([(sid, object(), object(), object())]), tmp_path/'model', spec, root=tmp_path)

"""The late gate must precede component access; lock tampering must fail closed."""
from types import SimpleNamespace
from pathlib import Path
import json
import pytest
from src.next import late_evaluation as late
from src.next.research_clock import ResearchTimeError
from src.utils.artifacts import sha256


def test_early_combination_cannot_open_component(monkeypatch, tmp_path):
    class Clock:
        def __init__(self, root):
            pass
        def snapshot(self):
            return {'research_elapsed_seconds': 28799.999}
    monkeypatch.setattr(late, 'ResearchClock', Clock)
    def forbidden(*args, **kwargs):
        raise AssertionError('An early job touched real component predictions')
    monkeypatch.setattr(late, 'component_fold', forbidden)
    with pytest.raises(ResearchTimeError, match='hour eight'):
        late.combine_fold(SimpleNamespace(guard=SimpleNamespace(root=tmp_path)), {}, 'full', 0)


def test_time_boundary_and_model_coordinates(monkeypatch, tmp_path):
    import numpy as np
    class Clock:
        def __init__(self, root):
            pass
        def snapshot(self):
            return {'research_elapsed_seconds': 28800.}
    monkeypatch.setattr(late, 'ResearchClock', Clock)
    calls = []
    def component(data, eid, *args):
        calls.append(eid)
        rows = np.array([0, 1]) if eid == 'left' else np.array([1, 0])
        return None, np.array([.2, .8], dtype=np.float32), rows, {}
    monkeypatch.setattr(late, 'component_fold', component)
    candidate = {'left': 'left', 'right': 'right', 'left_weight': .5}
    with pytest.raises(AssertionError):
        late.combine_fold(SimpleNamespace(guard=SimpleNamespace(root=tmp_path)), candidate, 'full', 0)
    assert calls == ['left', 'right']


def test_implementation_lock_detects_queue_change(tmp_path):
    (tmp_path/'configs').mkdir()
    source = tmp_path/'configs/queue.json'
    source.write_text('{"weight":0.5}', encoding='utf-8')
    lock = {'status': 'LOCKED', 'files_sha256': {'configs/queue.json': sha256(source)}}
    (tmp_path/'configs/next_late_implementation_lock.json').write_text(json.dumps(lock), encoding='utf-8')
    assert late.validate_lock(tmp_path)['status'] == 'LOCKED'
    source.write_text('{"weight":0.67}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='queue changed'):
        late.validate_lock(tmp_path)

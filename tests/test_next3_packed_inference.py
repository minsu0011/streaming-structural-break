import gc
import weakref
import numpy as np
import pytest
from src.next3.artifacts import ROOT, read_json, atomic_json, digest
from src.next3 import packed_inference as api
from src.next2.canonical_model import load
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor


@pytest.fixture(params=['PACKED_SHARED', 'PACKED_PRECOMPUTED'])
def artifact(tmp_path, request):
    source = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    model = load(source)
    api.write_artifact(model, tmp_path, read_json(ROOT/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json'),
        variant=request.param, origin='SYNTHETIC_UNIT_TEST')
    assert digest(tmp_path/'model.sbnext2') == digest(source)
    return tmp_path, model


def test_packed_public_lazy_stream_and_no_source_model_retention(artifact, monkeypatch):
    path, model = artifact; rng = np.random.default_rng(312)
    h = rng.normal(size=256).astype(np.float32); o = rng.normal(size=41).astype(np.float32)
    touched = []; refs = []; original_load = api.load_artifact
    def tracked_load(directory):
        result = original_load(directory)
        refs.extend(weakref.ref(a) for a in result[0].primary.compact.args()[:-1])
        return result
    monkeypatch.setattr(api, 'load_artifact', tracked_load)
    def datasets():
        for repeat in range(2):
            touched.append('H')
            def online():
                for j, value in enumerate(o):
                    touched.append(j); yield value
            yield h, online()
    stream = api.infer(datasets(), path)
    assert not touched and next(stream) is None and not touched
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    assert set(stream.gi_frame.f_locals) == {'datasets', 'model_directory_path', 'prepared'}
    reference = PreparedSharedBlendPredictor(model, resident=False)
    for repeat in range(2):
        state = reference.new_state(h)
        for j, value in enumerate(o):
            before = len(touched); prediction = next(stream)
            assert type(prediction) is float and prediction == state.predict_one(value)
            assert len(touched)-before == (2 if j == 0 else 1)
    with pytest.raises(StopIteration): next(stream)


def test_packed_public_contract_tamper_invalid_point_and_overwrite(artifact):
    path, model = artifact
    descriptor = read_json(path/'model.json'); prior = descriptor['variant']; descriptor['variant'] = 'UNKNOWN'
    atomic_json(path/'model.json', descriptor)
    with pytest.raises(RuntimeError, match='descriptor'): api.load_artifact(path)
    descriptor['variant'] = prior; atomic_json(path/'model.json', descriptor)
    stream = api.infer([(np.ones(256, dtype=np.float32), iter([float('nan')]))], path)
    assert next(stream) is None
    with pytest.raises(ValueError, match='Finite'): next(stream)
    def forbidden():
        raise AssertionError('Training was consumed before overwrite rejection')
        yield None
    with pytest.raises(RuntimeError, match='overwrite'): api.train(forbidden(), path)
    content = (path/'model.sbnext2').read_bytes(); (path/'model.sbnext2').write_bytes(content+b'corrupt')
    with pytest.raises(RuntimeError, match='bytes'): api.load_artifact(path)

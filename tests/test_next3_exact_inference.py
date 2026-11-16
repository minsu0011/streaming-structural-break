import numpy as np
import pytest
from src.next3.artifacts import ROOT,read_json,atomic_json,digest
from src.next3.exact_inference import write_artifact,load_artifact,infer,train
from src.next2.canonical_model import load
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor


@pytest.fixture
def artifact(tmp_path):
    source=ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    model=load(source)
    write_artifact(model,tmp_path,read_json(ROOT/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json'),origin='SYNTHETIC_UNIT_TEST')
    assert digest(tmp_path/'model.sbnext2')==digest(source)
    return tmp_path,model


def test_first_none_and_current_point_and_series_reset(artifact):
    path,model=artifact;rng=np.random.default_rng(29)
    h=rng.normal(size=256).astype(np.float32);o=rng.normal(size=37).astype(np.float32)
    touched=[]
    def datasets():
        for _ in range(2):
            touched.append('H')
            def online():
                for j,value in enumerate(o):
                    touched.append(j);yield value
            yield h,online()
    stream=infer(datasets(),path)
    assert not touched and next(stream) is None and not touched
    expected=PreparedSharedBlendPredictor(model,resident=False)
    for repeat in range(2):
        state=expected.new_state(h)
        for j,value in enumerate(o):
            before=len(touched);prediction=next(stream)
            assert type(prediction) is float and prediction==state.predict_one(value)
            assert len(touched)-before==(2 if j==0 else 1)
    with pytest.raises(StopIteration):next(stream)


def test_wrong_runtime_and_model_bytes_rejected(artifact):
    path,model=artifact
    descriptor=read_json(path/'model.json');descriptor['runtime']='not_registered';atomic_json(path/'model.json',descriptor)
    with pytest.raises(RuntimeError,match='descriptor'):load_artifact(path)
    descriptor['runtime']='precomputed_shared_v1';atomic_json(path/'model.json',descriptor)
    content=(path/'model.sbnext2').read_bytes();(path/'model.sbnext2').write_bytes(content+b'corrupt')
    with pytest.raises(RuntimeError,match='bytes'):load_artifact(path)


def test_nonfinite_point_and_overwrite_fail_before_consumption(artifact):
    path,model=artifact
    stream=infer([(np.ones(256,dtype=np.float32),iter([float('nan')]))],path);assert next(stream) is None
    with pytest.raises(ValueError,match='Finite'):next(stream)
    def forbidden():
        raise AssertionError('Training iterator was consumed before overwrite rejection')
        yield None
    with pytest.raises(RuntimeError,match='overwrite'):train(forbidden(),path)

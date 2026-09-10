import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from src.features.streaming import feature_version,replay
from src.models.learners import MODEL_NAMES,fit_candidate
from src.streaming.inference import infer
from src.validation.synthetic import fixture
from src.validation.protocol import protocol_predictions
from tests.test_models import engineering_data
from src.features.config import CONFIG

@pytest.mark.parametrize('name',MODEL_NAMES)
def test_learned_infer_prefix_and_offline_parity(name,engineering_data,tmp_path):
    x,y=engineering_data
    with threadpool_limits(limits=1):
        model=fit_candidate(name,x,y)
        model.save(tmp_path/'model.joblib')
        (tmp_path/'model.json').write_text(json.dumps({'kind':'supervised','name':name,'feature_hash':feature_version()}),encoding='utf-8')
        h,o,_=fixture('ar_reverse',online_length=160,break_at=80)
        changed=np.r_[o[:90],np.full(10,100,dtype=np.float32)]
        first=list(infer(iter([(h,iter(o))]),tmp_path))
        second=list(infer(iter([(h,iter(changed))]),tmp_path))
        assert first[0] is second[0] is None
        np.testing.assert_array_equal(first[1:91],second[1:91])
        expected=model.predict(np.array(list(replay(h,o))))
        np.testing.assert_allclose(first[1:],expected,rtol=0,atol=1e-12)
        wire=protocol_predictions(infer,[(h,o)],tmp_path)
        np.testing.assert_array_equal(wire,np.asarray(first[1:],dtype=np.float32))

def test_serialized_variant_uses_its_own_feature_configuration(tmp_path):
    config=CONFIG.variant(scales=(3,12,48),normalization='median_mad')
    h,o,tau=fixture('variance_up',online_length=240,break_at=120)
    features=np.array(list(replay(h,o,config=config)))
    target=(np.arange(len(o))>=tau).astype('uint8')
    model=fit_candidate('M1_logistic',features,target,config=config)
    model.save(tmp_path/'model.joblib')
    (tmp_path/'model.json').write_text(json.dumps({'kind':'supervised','name':model.name,'feature_hash':feature_version(config)}),encoding='utf-8')
    live=np.asarray(list(infer(iter([(h,iter(o))]),tmp_path))[1:])
    np.testing.assert_allclose(live,model.predict(features),atol=1e-12,rtol=0)
    wrong=np.array(list(replay(h,o)))
    assert np.max(abs(model.predict(wrong)-live))>1e-4

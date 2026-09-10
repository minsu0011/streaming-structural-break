import numpy as np
import pytest
from src.next.engine import EngineConfig
from src.next.fitting import fit_binary
from src.next.extensions import ExtendedBundle,ExtendedFeatureState,names_and_groups
from src.next.paired_blend import SameBankBlendBundle
from src.next.paired_variance_runtime import PreparedSameBankVarianceBlend
from src.features.config import CONFIG
from src.features.streaming import feature_names,FEATURE_FAMILIES


def components():
    rng = np.random.default_rng(818)
    extension = {'kind':'conditional_variance','settings':{'order':8,'normalization':'arch1','max_age':128}}
    config = CONFIG.variant(normalization='median_mad',scales=(5,20,160))
    old = ['arch8__'+name for name,family in zip(feature_names(config),FEATURE_FAMILIES) if family in 'ABCD']
    new,_ = names_and_groups(extension)
    names = tuple(old+list(new[:26]))
    h = rng.normal(size=1400).astype(np.float32)
    o = rng.normal(size=760).astype(np.float32)
    state = ExtendedFeatureState(h,names,extension)
    x = np.asarray([state.update(point).copy() for point in o],dtype=np.float32)
    ids = np.repeat(np.arange(38),20)
    target = (x[:,-8]+rng.normal(scale=.5,size=len(x))>np.median(x[:,-8])).astype(np.uint8)
    metadata = {'dataset_id':ids,'time_online':np.tile(np.arange(20),38),'target':target}
    train,valid = np.flatnonzero(ids<28),np.flatnonzero(ids>=28)
    class Guard:
        def partition(self,left,right):
            assert not set(left)&set(right)
    fitted = []
    for selected in (np.arange(len(names)),np.asarray([j for j,name in enumerate(names) if '_memory_' not in name])):
        model,p,detail = fit_binary(x[:,selected],metadata,train,valid,guard=Guard(),config=EngineConfig(),
            names=tuple(names[j] for j in selected),params={'n_estimators':7})
        fitted.append(ExtendedBundle.wrap(model,extension))
    return fitted,h,o,x


@pytest.mark.parametrize('weight',[.5,.67,.33])
def test_shared_bank_fixed_average_matches_two_independent_models(weight,tmp_path):
    fitted,h,o,x = components()
    model = SameBankBlendBundle.create(*fitted,weight)
    model.save(tmp_path/'blend.joblib')
    loaded = SameBankBlendBundle.load(tmp_path/'blend.joblib')
    expected = loaded.predict(x)
    assert np.unique(expected).size>1
    detector = PreparedSameBankVarianceBlend(loaded).new_state(h)
    actual = np.asarray([detector.predict_one(point) for point in o],dtype=np.float32)
    np.testing.assert_array_equal(actual,expected)
    prefix = PreparedSameBankVarianceBlend(loaded).new_state(h)
    np.testing.assert_array_equal(np.asarray([prefix.predict_one(point) for point in o[:129]],dtype=np.float32),actual[:129])
    assert detector.state_array_bytes==prefix.state_array_bytes

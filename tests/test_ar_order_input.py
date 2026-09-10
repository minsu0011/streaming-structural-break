import numpy as np
import pytest
from src.features.config import CONFIG
from src.features.ar_residual_input import historical_filter as legacy_filter, ARResidualCalibratedState
from src.features.ar_order_input import historical_filter, AROrderCalibratedState
from src.models.ar_order_calibrated import AROrderCalibratedBundle
from src.models.learners import fit_candidate


def test_order_four_keeps_the_original_ar_contract_bitwise():
    rng=np.random.default_rng(985);config=CONFIG.variant(normalization='median_mad',scales=(5,20,160))
    for h in (rng.standard_t(3,900).astype(np.float32),np.array([],dtype=np.float32),np.array([0.,np.nan,np.inf,1.,2.],dtype=np.float32)):
        for a,b in zip(historical_filter(h,config,4),legacy_filter(h,config)):np.testing.assert_array_equal(a,b)
        old=ARResidualCalibratedState(h,config=config);new=AROrderCalibratedState(h,config=config,input_order=4)
        for point in rng.normal(size=50):np.testing.assert_array_equal(old.update_and_get(point),new.update_and_get(point))


@pytest.mark.parametrize('order',[2,8])
def test_order_variants_prefix_memory_and_serialized_predictions(order,tmp_path):
    rng=np.random.default_rng(782);h=rng.normal(size=500).astype(np.float32);o=rng.normal(size=120).astype(np.float32);o[60:]*=2
    state=AROrderCalibratedState(h,input_order=order);before=state.state_array_bytes
    features=np.array([state.update_and_get(p).copy() for p in o]);assert before==state.state_array_bytes and len(state.ring)==order
    changed=np.r_[o[:40],np.full(15,500,dtype=np.float32)];second=AROrderCalibratedState(h,input_order=order)
    alternate=np.array([second.update_and_get(p).copy() for p in changed]);np.testing.assert_array_equal(features[:40],alternate[:40])
    bundle=AROrderCalibratedBundle(fit_candidate('M4_lightgbm',features,np.arange(len(o))>=60),input_order=order)
    bundle.save(tmp_path/'model.joblib');loaded=AROrderCalibratedBundle.load(tmp_path/'model.joblib');stream=loaded.make_state(h)
    actual=np.array([loaded.predict_one(stream.update_and_get(p)) for p in o],dtype=np.float32)
    np.testing.assert_array_equal(actual,loaded.predict(features).astype(np.float32))

"""Existing fitted fold artifacts provide independent batch-oracle contracts."""
import numpy as np
import pytest
from src.next4.artifacts import ROOT,BASE,read_json
from src.next4.model import FeatureModel
from src.next4.robust_finalist import PreparedRobustness,ALLOWED
from src.next4.synthetic import load_reference
from src.next2.counterexamples import predict
from src.next2.blend import combine


@pytest.mark.parametrize('candidate',ALLOWED)
def test_existing_fitted_recipe_scalar_adapter(candidate):
    model=FeatureModel.load(BASE/'experiments'/candidate/'full/fold_0_expert.joblib')
    frozen,_=load_reference('full_fold0');prepared=PreparedRobustness(model,frozen.model,candidate)
    t=np.arange(512,dtype=np.float64);h=(np.sin(.13*t)+.31*np.cos(.57*t)).astype(np.float32)
    o=(np.sin(.13*(t[:96]+512))+.31*np.cos(.57*(t[:96]+512))).astype(np.float32);o[32]+=8
    expert=np.asarray(model.predict(model.make_state(h).replay(o)),dtype=np.float32)
    wanted=combine(expert,predict(frozen.complement,h,o),.8) if candidate.startswith('B1_') else combine(predict(frozen,h,o),expert,.9)
    state=prepared.new_state(h);before=state.state_array_bytes
    actual=np.array([state.predict_one(v) for v in o],dtype=np.float32)
    np.testing.assert_array_equal(actual,wanted);assert before==state.state_array_bytes
    suffix=o.copy();suffix[48:]*=-20;other=prepared.new_state(h)
    changed=np.array([other.predict_one(v) for v in suffix],dtype=np.float32)
    np.testing.assert_array_equal(actual[:48],changed[:48])
    with pytest.raises(ValueError):state.predict_one(float('nan'))

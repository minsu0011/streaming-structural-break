import src.next9
import numpy as np
import pytest
from src.next9.streaming import PrefixBank
from src.next9.runtime_laws import RuntimeSelfLaw,RuntimeCrossLaw
from src.next9.transitions import fit_law
from src.next9.transition_inputs import path_features
from src.next9.cross import CrossLaw,path_features as cross_path
from src.next9.data import FitScope


def test_first_observed_T2_self_and_cross_paths_have_zero_surprise():
    ids=np.arange(128);state=ids%3;run=ids%15+1;slope=np.sin(ids);context=np.zeros((128,6))
    scope=FitScope(frozenset(ids),frozenset([128]),frozenset([128]),'toy')
    law=fit_law(state,(ids//3)%3,run,slope,context,ids,ids,scope,'T2_SELF_RUN_SLOPE_H')
    bank=PrefixBank(np.zeros((16,153)),np.full((3,153),.02),np.zeros(6),512)
    with pytest.raises(ValueError):path_features(bank,np.array([0]),0,law)
    adapted=RuntimeSelfLaw(law)
    output=path_features(bank,np.array([0]),0,adapted)
    assert output.shape==(1,8) and np.isfinite(output).all() and output[0,2]==0
    np.testing.assert_allclose(adapted.probabilities(state,run,slope,context),law.probabilities(state,run,slope,context),rtol=0,atol=1e-14)
    cross=CrossLaw(0,(),law.classifier,np.full(4,.25),{})
    with pytest.raises(ValueError):cross_path(bank,np.array([0]),cross)
    wrapper=RuntimeCrossLaw(cross)
    result=cross_path(bank,np.array([0]),wrapper)
    assert result.shape==(1,4) and np.isfinite(result).all() and result[0,0]==0
    np.testing.assert_allclose(wrapper.probabilities(bank,np.array([0])),cross.probabilities(bank,np.array([0])),rtol=0,atol=1e-14)

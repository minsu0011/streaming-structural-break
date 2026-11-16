"""Exercise recovery from a nonstationary checkpoint without changing the fit objective."""
import src.next9
import numpy as np
from src.next9.data import FitScope
from src.next9.learning import fit_classifier
from src.next9.learning_converged_v2 import continue_classifier,logistic_stationarity

def test_checkpoint_continuation_preserves_transform_and_reaches_same_optimum():
    rng=np.random.default_rng(18);x=rng.normal(size=(240,5));y=(x[:,0]+.3*x[:,1]+rng.normal(size=240)>0).astype(int)
    ids=np.arange(240);scope=FitScope(frozenset(ids),frozenset([240]),frozenset([240]),'checkpoint_test')
    model=fit_classifier(x,y,ids,ids,scope);expected=model.predict_proba(x)
    center=model.center.copy();scale=model.scale.copy();columns=model.columns.copy()
    # A deliberately nonstationary numerical checkpoint exercises the recovery
    # branch. Actual250-iteration RP warnings have a separate frozen diagnosis.
    model.estimator.coef_[0,0]+=.1;model.training.update(converged=False,iterations=[250])
    before=logistic_stationarity(model,x,y,ids)['objective']
    recovered=continue_classifier(model,x,y,ids,ids,scope)
    assert recovered is model and recovered.estimator.C==.1
    for a,b in [(center,model.center),(scale,model.scale),(columns,model.columns)]:np.testing.assert_array_equal(a,b)
    assert model.training['converged'] and model.training['estimator_fit_calls']==2
    assert model.training['optimizer_continuation']['after']['objective']<before
    np.testing.assert_allclose(model.predict_proba(x),expected,rtol=0,atol=2e-6)

import src.next9
import numpy as np
from src.next9.data import FitScope
from src.next9.learning import fit_classifier as original
from src.next9.learning_converged_v2 import fit_classifier,logistic_stationarity

def test_already_converged_registered_fit_is_unchanged():
    rng=np.random.default_rng(5);x=rng.normal(size=(160,4));y=(x[:,0]+rng.normal(size=160)>.2).astype(int);ids=np.arange(160)
    scope=FitScope(frozenset(ids),frozenset([160]),frozenset([160]),'solver_test')
    a=original(x,y,ids,ids,scope);b=fit_classifier(x,y,ids,ids,scope)
    assert a.training['converged'] and b.training==a.training
    np.testing.assert_array_equal(a.predict_proba(x),b.predict_proba(x))

def test_stationarity_gradient_matches_objective_finite_difference():
    rng=np.random.default_rng(6);x=rng.normal(size=(100,3));y=(x[:,0]>.3).astype(int);ids=np.arange(100)
    scope=FitScope(frozenset(ids),frozenset([100]),frozenset([100]),'gradient_test')
    m=original(x,y,ids,ids,scope);m.estimator.coef_[0,0]+=.1
    stats=logistic_stationarity(m,x,y,ids)
    epsilon=1e-5;approx=[]
    for k in range(4):
        target=m.estimator.coef_[0] if k<3 else m.estimator.intercept_;index=k if k<3 else 0
        initial=target[index];target[index]=initial+epsilon;plus=logistic_stationarity(m,x,y,ids)['objective']
        target[index]=initial-epsilon;minus=logistic_stationarity(m,x,y,ids)['objective'];target[index]=initial
        approx.append((plus-minus)/(2*epsilon*100))
    assert abs(max(abs(v) for v in approx)-stats['normalized_gradient_max'])<1e-9

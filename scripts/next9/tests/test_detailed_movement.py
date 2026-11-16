import src.next9
import numpy as np
from src.next9.data import FitScope
from src.next9.detailed_movement import fit


def test_mechanical_duplicate_cannot_manufacture_forecast_gain():
    rng=np.random.default_rng(9101604)
    x=rng.normal(size=(128,4));y=rng.normal(size=(128,15));ids=np.repeat(np.arange(64),2)
    scope=FitScope(frozenset(range(64)),frozenset(range(64,128)),frozenset(),'toy')
    a=fit(x,y,ids,ids,scope,['a','b','c','d'],'D3')
    padded=np.column_stack([x,x[:,0],np.ones(len(x))])
    b=fit(padded,y,ids,ids,scope,['a','b','c','d','duplicate','constant'],'D5')
    np.testing.assert_allclose(a.predict(x),b.predict(padded),atol=1e-12)


def test_constant_forecast_is_training_group_mean():
    ids=np.repeat(np.arange(64),2);x=np.ones((128,4));y=np.tile(np.arange(15),(128,1))
    scope=FitScope(frozenset(range(64)),frozenset(range(64,128)),frozenset(),'toy')
    model=fit(x,y,ids,ids,scope,['a','b','c','d'],'D2')
    np.testing.assert_allclose(model.predict(x),y)

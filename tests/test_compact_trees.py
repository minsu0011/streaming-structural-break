import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from src.models.learners import fit_candidate

@pytest.mark.parametrize('name',['M3_histgb','M4_lightgbm'])
def test_split_threshold_neighbors(name):
    rng=np.random.default_rng(333)
    x=rng.normal(size=(1000,57)).astype(np.float32);y=(x[:,0]+x[:,1]>.3).astype(np.uint8)
    with threadpool_limits(limits=1):
        model=fit_candidate(name,x,y)
        tree=model.compact
        points=[]
        for f,threshold in zip(tree.features,tree.thresholds):
            if f<0:continue
            point=np.zeros(len(model.columns),dtype=np.float32)
            for value in (np.nextafter(np.float32(threshold),np.float32(-np.inf)),np.float32(threshold),np.nextafter(np.float32(threshold),np.float32(np.inf))):
                point[f]=value;points.append(point.copy())
        points=np.asarray(points)
        native=model.estimator.booster_.predict(points,num_threads=1) if name=='M4_lightgbm' else model.estimator.predict_proba(points)[:,1]
        np.testing.assert_allclose(tree.predict(points),native,rtol=0,atol=1e-12)

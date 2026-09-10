import numpy as np
from xgboost import XGBClassifier,DMatrix
from src.models.compact_xgboost import export_xgboost,margin_one

def test_float32_tree_export_boundaries_and_missing():
    rng=np.random.default_rng(55);x=rng.normal(size=(1500,8)).astype(np.float32);y=(x[:,0]*x[:,1]+x[:,2]>.7).astype('uint8')
    x[::19,2]=np.nan
    estimator=XGBClassifier(n_estimators=100,max_depth=3,n_jobs=2,tree_method='hist',random_state=20260908).fit(x,y)
    tree=export_xgboost(estimator);probes=[row.copy() for row in x[:100]]
    for feature,threshold in zip(tree.features,tree.thresholds):
        if feature<0:continue
        for value in (np.nextafter(threshold,np.float32(-np.inf)),threshold,np.nextafter(threshold,np.float32(np.inf)),np.float32(np.nan)):
            row=np.zeros(8,dtype=np.float32);row[feature]=value;probes.append(row)
    probes=np.asarray(probes,dtype=np.float32)
    native=estimator.get_booster().inplace_predict(probes)
    np.testing.assert_allclose(tree.predict(probes),native,rtol=0,atol=2e-7)
    native_margin=estimator.get_booster().predict(DMatrix(probes),output_margin=True)
    compact_margin=np.array([margin_one(row,*tree.args()) for row in probes],dtype=np.float32)
    np.testing.assert_allclose(compact_margin,native_margin,rtol=0,atol=2e-6)

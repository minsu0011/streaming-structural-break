import numpy as np
from src.features.config import CONFIG
from src.features.ar_residual_input import historical_filter, innovation_one, ARResidualCalibratedState
from src.models.ar_calibrated import ARCalibratedBundle
from src.models.learners import fit_candidate


def test_ar_residual_matches_explicit_lag_dot_product():
    rng=np.random.default_rng(397); h=rng.normal(size=1000).astype(np.float32)
    for t in range(4,len(h)):h[t]+=.6*h[t-1]-.2*h[t-4]
    config=CONFIG.variant(normalization='median_mad');reference,coefficients,ring,residual=historical_filter(h,config)
    assert np.isfinite(residual).all() and len(residual)==len(h)-4
    counter=np.zeros(1,dtype=np.int64);past=list(np.clip((h.astype(np.float64)-reference[0])/reference[1],-reference[2],reference[2]))
    for point in rng.normal(size=40):
        z=np.clip((point-reference[0])/reference[1],-reference[2],reference[2])
        predicted=0.
        for j in range(4):predicted+=coefficients[j]*past[-j-1]
        expected=np.float32(z-predicted); actual=innovation_one(float(point),reference,coefficients,ring,counter)
        np.testing.assert_array_equal(actual,expected);past.append(z)


def test_ar_input_prefix_memory_and_saved_prediction(tmp_path):
    rng=np.random.default_rng(728);h=rng.normal(size=600).astype(np.float32);o=rng.normal(size=140).astype(np.float32);o[70:]*=2
    state=ARResidualCalibratedState(h); before=state.state_array_bytes
    features=np.array([state.update_and_get(p).copy() for p in o]);assert before==state.state_array_bytes
    changed=np.r_[o[:40],np.full(9,100,dtype=np.float32)];second=ARResidualCalibratedState(h)
    alternate=np.array([second.update_and_get(p).copy() for p in changed]);np.testing.assert_array_equal(features[:40],alternate[:40])
    model=ARCalibratedBundle(fit_candidate('M4_lightgbm',features,np.arange(len(o))>=70));model.save(tmp_path/'model.joblib')
    loaded=ARCalibratedBundle.load(tmp_path/'model.joblib');stream=loaded.make_state(h)
    actual=np.array([loaded.predict_one(stream.update_and_get(p)) for p in o],dtype=np.float32)
    np.testing.assert_array_equal(actual,loaded.predict(features).astype(np.float32))
    for historical in ([],[0.],[2.]*10,[np.nan,np.inf]):
        stream=ARResidualCalibratedState(historical)
        for point in (0.,1.,np.nan,np.inf,1e200):assert np.isfinite(stream.update_and_get(point)).all()

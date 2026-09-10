import numpy as np
import pytest
from src.next.engine import EngineConfig
from src.next.extensions import ExtendedBundle,ExtendedFeatureState,names_and_groups
from src.next.registered_extensions import RegisteredExtendedBundle,register_extension
from src.next.fitting import fit_binary
from src.next.fused_runtime import PreparedNextPredictor
from src.features.streaming import feature_names
from src.features.config import CONFIG


class Guard:
    def partition(self,left,right):
        assert set(left).isdisjoint(right)


@pytest.fixture(scope='module',params=['ar_score','score_coordinates','markov_transition','ar_mismatch','conditional_variance','legacy_bank'])
def fitted_case(request):
    kind = request.param
    extension = {'kind':kind,'settings':{}}
    registration = None
    if kind in ('score_coordinates','markov_transition'):
        registration = {'kind':kind,'module':'score_coordinates' if kind=='score_coordinates' else 'markov_evidence'}
        register_extension(extension,registration)
    new_names,_ = names_and_groups(extension)
    old_names = [] if kind=='legacy_bank' else ['arch8__'+name for name in feature_names(CONFIG.variant(normalization='median_mad',scales=(5,20,160)))[:51]]
    names = old_names+new_names
    rng = np.random.default_rng(1938)
    hs,points,xs,ys,ids,times = [],[],[],[],[],[]
    for sid in range(6):
        h = rng.standard_t(6,800).astype(np.float32)
        o = rng.normal(size=180).astype(np.float32)
        tau = 60+10*(sid%3) if sid%2 else len(o)
        o[tau:] *= 2.5
        state = ExtendedFeatureState(h,names,extension)
        xs.append(np.array([state.update(point).copy() for point in o]))
        ys.append(np.asarray(np.arange(len(o))>=tau,dtype=np.uint8))
        ids.extend([sid]*len(o));times.extend(range(len(o)))
        hs.append(h);points.append(o)
    x = np.concatenate(xs)
    metadata = {'target':np.concatenate(ys),'dataset_id':np.asarray(ids),'time_online':np.asarray(times)}
    train,valid = np.flatnonzero(metadata['dataset_id']<4),np.flatnonzero(metadata['dataset_id']>=4)
    model,_,_ = fit_binary(x,metadata,train,valid,guard=Guard(),config=EngineConfig(),names=names,params={'n_estimators':16})
    model = RegisteredExtendedBundle.wrap_registered(model,extension,registration) if registration else ExtendedBundle.wrap(model,extension)
    yield model,hs[4:],points[4:],xs[4:]


def test_compiled_and_resident_predictions_match_original_float32(fitted_case):
    model,hs,streams,matrices = fitted_case
    for resident in (False,True):
        prepared = PreparedNextPredictor(model,resident=resident)
        for h,points,matrix in zip(hs,streams,matrices):
            expected = model.predict(matrix).astype(np.float32)
            assert np.ptp(expected)>0.001
            detector = prepared.new_state(h)
            size = detector.state_array_bytes
            actual = np.array([detector.predict_one(point) for point in points],dtype=np.float32)
            np.testing.assert_array_equal(actual,expected)
            assert size==detector.state_array_bytes
            prefix = prepared.new_state(h)
            short = np.array([prefix.predict_one(point) for point in points[:79]],dtype=np.float32)
            np.testing.assert_array_equal(short,actual[:79])

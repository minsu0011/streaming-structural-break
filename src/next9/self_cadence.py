"""Auxiliary self-law deployment audit, separate from frozen M-model branches."""
import copy,pickle,hashlib
import numpy as np
from src.next9.runtime_laws import RuntimeSelfLaw
from src.next9.streaming import PrefixBank
from src.next9.replay_checks import ArrayBank
from src.next9.transition_inputs import path_features,probabilities

VARIANTS=('T0_PRIOR','T1_SELF_MARKOV','T2_SELF_RUN_SLOPE_H','T3_SAME_INPUT_DEPTH3')


class CapturedLaw(RuntimeSelfLaw):
    """Record the already computed current forecast; never add a model call."""
    captured=None
    def probabilities(self,state,run,slope,context):
        out=super().probabilities(state,run,slope,context)
        if self.captured is None and len(out):self.captured=out.copy()
        return out


def replay_series(seed,values,epsilon,context,h_length,laws,chosen,mode):
    features=sorted(laws);n=len(values);rows=np.arange(n,dtype=np.int64)
    reference=ArrayBank(seed,values,epsilon,context,h_length)
    expected=np.empty((n,len(features),8),dtype=np.float32)
    expected_p=np.empty((n,len(features),4));loss=np.empty((max(n-1,0),len(features),4))
    wrappers={j:CapturedLaw(laws[j][chosen[j]]) for j in features}
    for column,j in enumerate(features):
        states=reference.pre_state(rows,j);ctx=np.tile(context,(n,1))
        expected[:,column]=path_features(reference,rows,j,laws[j][chosen[j]])
        expected_p[:,column]=probabilities(laws[j][chosen[j]],states,ctx)
        target=reference.pre_state(rows[1:],j)['S1_DIRECTION']
        for variant_index,variant in enumerate(VARIANTS):
            p=probabilities(laws[j][variant],states,ctx)
            loss[:,column,variant_index]=-np.log(np.maximum(p[np.arange(n-1),target],1e-8))
    before=hashlib.sha256(pickle.dumps(laws,protocol=5)).hexdigest()
    prefix=PrefixBank(seed,epsilon,context,h_length)
    actual=np.empty_like(expected);actual_p=np.empty_like(expected_p);restored=False
    for t in range(n):
        # No current F[t] has been staged when forecasts and path states issue.
        assert prefix.value is None and prefix.time==t
        for column,j in enumerate(features):
            wrapper=wrappers[j];wrapper.captured=None
            actual[t,column]=path_features(prefix,np.asarray([t]),j,wrapper)[0]
            actual_p[t,column]=wrapper.captured[0]
        prefix.stage(values[t]);prefix.commit()
        if t==min(31,n-1):
            prefix=pickle.loads(pickle.dumps(prefix,protocol=5));restored=True
    assert hashlib.sha256(pickle.dumps(laws,protocol=5)).hexdigest()==before
    error_p=float(np.max(np.abs(actual_p-expected_p),initial=0))
    error_path=float(np.max(np.abs(actual-expected),initial=0))
    assert error_p<=1e-12,('self_probability',error_p)
    # Float32 path summaries can cross a final rounding boundary when BLAS
    # reduces a batch dot product in a different order. State and momentum
    # fields remain exact; probability-derived summaries use a fixed tolerance.
    np.testing.assert_array_equal(actual[:,:,[0,1,6,7]],expected[:,:,[0,1,6,7]])
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-6)
    assert np.all(actual[0,:,2]==0)
    keep=np.arange(max(n-1,0))>=400 if mode=='EARLY' else np.ones(max(n-1,0),dtype=bool)
    sampled=keep&(np.arange(max(n-1,0))%8==0)
    eligible=bool(keep.any() and sampled.any())
    means=np.stack([loss[sampled].mean(axis=0),loss[keep].mean(axis=0)]) if eligible else np.full((2,len(features),4),np.nan)
    return means,dict(points=n,evaluated_points=int(keep.sum()),phase0_points=int(sampled.sum()),
        same_group_eligible=eligible,maximum_probability_error=error_p,maximum_path_error=error_path,
        state_and_momentum_bitwise=True,first_surprise_zero=True,save_restore=restored,
        forecast_issued_before_current_F=True,learned_parameters_unchanged=True)

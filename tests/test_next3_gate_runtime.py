import numpy as np
from src.next3.gate_runtime import replay_gate_features
from src.next3.stage_trace import replay_stage_trace
from src.next2.shared_inference import load_artifact
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next3.artifacts import ROOT
from src.next3.router import gate_design


def test_gate_runtime_all_six_inputs_match_scientific_pipeline():
    model,_=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    prepared=PreparedSharedBlendPredictor(model,resident=False)
    rng=np.random.default_rng(771)
    h=rng.normal(size=2048).astype(np.float32);o=rng.normal(size=93).astype(np.float32);o[13]=30.
    actual=replay_gate_features(o,prepared.new_state(h).call_args)
    state=prepared.new_state(h);stages=replay_stage_trace(o,state.call_args)
    np.testing.assert_array_equal(actual[:,:2],stages[:,38:40].astype(np.float32))
    np.testing.assert_array_equal(actual[:,2],np.max(np.abs(stages[:,[1,2,14,15]]),axis=1))
    np.testing.assert_array_equal(actual[:,4:],stages[:,[0,13]].astype(np.float32))
    from src.next2.shared_blend_runtime import shared_feature_step
    state=prepared.new_state(h);dep=[]
    for value in o:
        shared_feature_step(float(value),state.call_args)
        x=state.call_args[1][-1][-1]
        dep.append(max(abs(x[6*lag+j]) for lag in range(4) for j in (0,1)))
    np.testing.assert_array_equal(actual[:,3],np.asarray(dep,dtype=np.float32))
    for count in (1,2,17,92):
        np.testing.assert_array_equal(actual[:count],replay_gate_features(o[:count],prepared.new_state(h).call_args))

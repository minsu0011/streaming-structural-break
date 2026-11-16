import numpy as np
from pathlib import Path
from src.next3.stage_trace import replay_stage_trace,directional_bayes
from src.next2.shared_inference import load_artifact
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next2.counterexamples import predict

ROOT=Path(__file__).resolve().parents[1]


def test_stage_trace_preserves_frozen_scores_and_directional_bayes_sum():
    rng=np.random.default_rng(194)
    h=rng.normal(size=512).astype(np.float32);o=rng.normal(size=300).astype(np.float32)
    model,_=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    prepared=PreparedSharedBlendPredictor(model,resident=False)
    state=prepared.new_state(h)
    trace=replay_stage_trace(o,state.call_args)
    np.testing.assert_array_equal(trace[:,40].astype(np.float32),predict(prepared,h,o))
    pieces=directional_bayes(state.shared.call_args[5],state.shared.call_args[6])
    current=state.shared.pair_output
    for channel in range(2):
        for prior in range(2):
            total=np.logaddexp(pieces[channel*4+prior],pieces[channel*4+2+prior])
            np.testing.assert_allclose(total,current[channel*13+9+prior],rtol=0,atol=2e-6)

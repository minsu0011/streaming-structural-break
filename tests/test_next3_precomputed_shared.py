import numpy as np
from src.next3.precomputed_shared_runtime import PreparedPrecomputedSharedPredictor
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next2.shared_inference import load_artifact
from src.next3.artifacts import ROOT


def test_shared_precompute_exact_scores_features_and_constant_memory():
    model,_=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    expected=PreparedSharedBlendPredictor(model,resident=False)
    actual=PreparedPrecomputedSharedPredictor(model,resident=False)
    rng=np.random.default_rng(609)
    for count in (64,512,2048):
        h=rng.standard_t(5,count).astype(np.float32)
        o=rng.normal(size=1040).astype(np.float32);o[17]=-50.;o[701:]*=.5
        a=expected.new_state(h);b=actual.new_state(h);before=b.state_array_bytes
        for value in o:
            assert a.predict_one(value)==b.predict_one(value)
            np.testing.assert_array_equal(a.call_args[5],b.call_args[0][5])
            np.testing.assert_array_equal(a.call_args[1][-1][-1],b.call_args[0][1][-1][-1])
        assert b.state_array_bytes==before
        assert before<=2*a.state_array_bytes
        from src.next3.runtime_memory import accounting
        measured,allocations,private=accounting(b)
        baseline,_,_=accounting(a)
        assert measured['private_per_series_numeric_bytes']==baseline['private_per_series_numeric_bytes']+1200
        assert measured['retained_state_numeric_bytes_including_shared_selectors']==before


def test_interleaved_streams_do_not_mutate_shared_tables():
    model,_=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    prepared=PreparedPrecomputedSharedPredictor(model,resident=False)
    rng=np.random.default_rng(849);h=rng.normal(size=256).astype(np.float32);o=rng.normal(size=64).astype(np.float32)
    tables=(prepared.dependence_table.copy(),prepared.dependence_logs.copy())
    assert not prepared.dependence_table.flags.writeable and not prepared.dependence_logs.flags.writeable
    a,b=prepared.new_state(h),prepared.new_state(h)
    for value in o:assert a.predict_one(value)==b.predict_one(value)
    np.testing.assert_array_equal(prepared.dependence_table,tables[0]);np.testing.assert_array_equal(prepared.dependence_logs,tables[1])

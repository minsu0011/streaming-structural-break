import gc
import weakref
import numpy as np
import pytest
from src.next3.artifacts import ROOT
from src.next2.shared_inference import load_artifact
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next3.packed_shared_runtime import PreparedPackedSharedPredictor, accounting


@pytest.mark.parametrize('variant', ['PACKED_SHARED', 'PACKED_PRECOMPUTED'])
def test_packed_shared_exact_features_scores_interleaving_and_memory(variant):
    model, _ = load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    reference = PreparedSharedBlendPredictor(model, resident=False)
    packed = PreparedPackedSharedPredictor(model, variant=variant, resident=False)
    rng = np.random.default_rng(30092)
    tree_snapshot = [packed.primary.tree_args[1].tobytes(), packed.complement.tree_args[1].tobytes()]
    for length in (64, 512, 2048):
        h = rng.standard_t(5, length).astype(np.float32)
        o = rng.normal(size=1040).astype(np.float32); o[13] = -50.; o[700:] *= .5
        left = reference.new_state(h); right = packed.new_state(h); interleaved = packed.new_state(h)
        before = right.state_array_bytes
        for v in o:
            expected = left.predict_one(v)
            assert right.predict_one(v) == expected == interleaved.predict_one(v)
            np.testing.assert_array_equal(left.output.view(np.uint32), right.base.output.view(np.uint32))
            np.testing.assert_array_equal(left.shared.dependence_output.view(np.uint32),
                                          right.base.shared.dependence_output.view(np.uint32))
        assert before == right.state_array_bytes
        measured, _, private = accounting(right); baseline, _, _ = accounting(left)
        expected_extra = 1200 if variant == 'PACKED_PRECOMPUTED' else 0
        assert measured['private_per_series_numeric_bytes'] == baseline['private_per_series_numeric_bytes'] + expected_extra
        assert measured['retained_state_numeric_bytes_including_shared_selectors'] == before
        assert measured['model_and_prepared_shared_numeric_bytes'] < baseline['model_and_prepared_shared_numeric_bytes']
    assert packed.primary.tree_args[1].tobytes() == tree_snapshot[0]
    assert packed.complement.tree_args[1].tobytes() == tree_snapshot[1]
    assert not packed.primary.tree_args[1].flags.writeable


def test_packed_prepared_does_not_retain_original_model_arrays():
    from src.next2.canonical_model import load
    model = load(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2')
    references = [weakref.ref(a) for a in model.primary.compact.args()[:-1]]
    packed = PreparedPackedSharedPredictor(model, resident=False)
    del model
    gc.collect()
    assert all(r() is None for r in references)
    assert packed.primary.tree_args[1].dtype.itemsize == 16

from pathlib import Path
import numpy as np
import pytest
from src.next2.canonical_model import load
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next3.state_transfer import (BLOCKS, arrays, immutable_ids, clone_arguments,
    transferred_arguments, trace_scores, trace_components)
from src.next.fused_runtime import unique_array_bytes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared():
    model = load(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2')
    return PreparedSharedBlendPredictor(model, resident=False)


def data():
    t = np.arange(1024, dtype=np.float64)
    h = (np.sin(t*.37)+.2*np.cos(t*.11)).astype(np.float32)
    online = (np.cos(np.arange(256)*.29)+.1*np.sin(np.arange(256)*.05)).astype(np.float32)
    outlier = online.copy(); outlier[32] += 20
    return h, online, outlier


def test_deep_clone_preserves_aliases_without_recurrent_donor_aliases(prepared):
    h, online, _ = data(); state = prepared.new_state(h); clone = clone_arguments(state.call_args)
    assert not ((set(arrays(state.call_args))-immutable_ids(state.call_args))&set(arrays(clone)))
    before = unique_array_bytes(clone[:-1])
    np.testing.assert_array_equal(trace_scores(state.call_args, online), trace_scores(clone, online))
    assert unique_array_bytes(clone[:-1]) == before


@pytest.mark.parametrize('anchor', [32, 63, 159])
def test_complete_state_transfers_reproduce_both_original_futures(prepared, anchor):
    h, control, outlier = data()
    left = prepared.new_state(h).call_args; right = prepared.new_state(h).call_args
    trace_scores(left, control[:anchor+1]); trace_scores(right, outlier[:anchor+1])
    restored = transferred_arguments(right, left, BLOCKS)
    induced = transferred_arguments(left, right, BLOCKS)
    future = control[anchor+1:]; np.testing.assert_array_equal(future, outlier[anchor+1:])
    np.testing.assert_array_equal(trace_scores(restored, future), trace_scores(left, future))
    np.testing.assert_array_equal(trace_scores(induced, future), trace_scores(right, future))


def test_single_block_transfer_does_not_mutate_original_states(prepared):
    h, control, outlier = data(); left = prepared.new_state(h).call_args; right = prepared.new_state(h).call_args
    trace_scores(left, control[:64]); trace_scores(right, outlier[:64])
    saved = {key: value.copy() for key, value in arrays(right).items()}
    for block in BLOCKS:
        hybrid = transferred_arguments(right, left, [block])
        assert unique_array_bytes(hybrid[:-1]) == unique_array_bytes(right[:-1])
        trace_scores(hybrid, control[64:])
        for key, value in arrays(right).items(): np.testing.assert_array_equal(value, saved[key])


def test_unregistered_or_duplicate_state_blocks_are_rejected(prepared):
    h, _, _ = data(); args = prepared.new_state(h).call_args
    for blocks in (['UNKNOWN'], ['SHARED_AR_FILTER', 'SHARED_AR_FILTER']):
        with pytest.raises(ValueError): transferred_arguments(args, args, blocks)


def test_component_trace_reconstructs_float32_frozen_blend(prepared):
    h, online, _ = data(); p = trace_components(prepared.new_state(h).call_args, online)
    expected = (.8*p[:, 0].astype(np.float64)+(1.-.8)*p[:, 1].astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(p[:, 2], expected)

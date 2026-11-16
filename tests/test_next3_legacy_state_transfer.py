from pathlib import Path
import numpy as np
import pytest
from src.next2.canonical_model import load
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next.fused_runtime import unique_array_bytes
from src.next3.state_transfer import arrays, at, trace_scores, transferred_arguments
from src.next3.legacy_state_transfer import LEGACY_BLOCKS, transfer_legacy_arguments

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared():
    return PreparedSharedBlendPredictor(load(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'), resident=False)


def paired(prepared, anchor):
    h = (np.sin(np.arange(1024)*.31)+.3*np.cos(np.arange(1024)*.07)).astype(np.float32)
    control = (np.cos(np.arange(800)*.27)+.2*np.sin(np.arange(800)*.019)).astype(np.float32)
    outlier = control.copy(); outlier[64] += 20
    left = prepared.new_state(h).call_args; right = prepared.new_state(h).call_args
    trace_scores(left, control[:anchor+1]); trace_scores(right, outlier[:anchor+1])
    return left, right, control[anchor+1:]


@pytest.mark.parametrize('anchor', [64, 191, 319, 575])
@pytest.mark.parametrize('reverse', [False, True])
def test_all_fine_blocks_equal_complete_legacy_transfer(prepared, anchor, reverse):
    left, right, future = paired(prepared, anchor)
    destination, donor = (left, right) if reverse else (right, left)
    fine = transfer_legacy_arguments(destination, donor, LEGACY_BLOCKS)
    coarse = transferred_arguments(destination, donor, ['LEGACY_ARCH_ABCD'])
    np.testing.assert_array_equal(trace_scores(fine, future), trace_scores(coarse, future))


@pytest.mark.parametrize('block', LEGACY_BLOCKS)
def test_single_block_preserves_donors_state_size_and_deterministic_age(prepared, block):
    left, right, future = paired(prepared, 191)
    saved = {key: value.copy() for key, value in arrays((left, right)).items()}
    hybrid = transfer_legacy_arguments(right, left, [block])
    assert unique_array_bytes(hybrid[:-1]) == unique_array_bytes(right[:-1])
    np.testing.assert_array_equal(at(hybrid, (0, 2, 5))[[0, 4, 5, 6, 7]], at(right, (0, 2, 5))[[0, 4, 5, 6, 7]])
    trace_scores(hybrid, future)
    assert unique_array_bytes(hybrid[:-1]) == unique_array_bytes(right[:-1])
    for key, value in arrays((left, right)).items(): np.testing.assert_array_equal(value, saved[key])


@pytest.mark.parametrize('blocks', [[], ['UNKNOWN'], ['LEGACY_EWMA', 'LEGACY_EWMA']])
def test_invalid_blocks_rejected(prepared, blocks):
    left, right, _ = paired(prepared, 64)
    with pytest.raises(ValueError): transfer_legacy_arguments(right, left, blocks)


def test_different_anchor_age_cannot_be_used_as_a_donor(prepared):
    left, _, _ = paired(prepared, 64); _, right, _ = paired(prepared, 65)
    with pytest.raises(AssertionError): transfer_legacy_arguments(right, left, ['LEGACY_EWMA'])

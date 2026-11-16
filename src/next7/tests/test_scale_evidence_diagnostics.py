import numpy as np
import pytest
from src.next7.scale_evidence_diagnostics import first_detection_partition, scale_window_partition


@pytest.mark.parametrize('qualified,raw,expected', [
    ([63, 143], [63, 143], 'prebreak_qualified_memory'),
    ([143], [63, 143], 'correct_first_qualification'),
    ([], [143], 'postbreak_raw_confirmation_rejected'),
    ([], [63], 'no_postbreak_raw_confirmation')])
def test_first_event_outcomes_do_not_erase_prebreak_memory(qualified, raw, expected):
    values = np.zeros((256, 45))
    values[qualified, 40] = 1
    values[raw, 18] = 1
    assert first_detection_partition(values, 64, 256) == expected


@pytest.mark.parametrize('tau,first,eligible', [(63, 127, 9), (64, 127, 9), (65, 143, 8)])
def test_four_raw_blocks_must_be_fully_postbreak(tau, first, eligible):
    values = np.zeros((256, 45))
    points = np.arange(15, 256, 16)
    values[points, 15] = 1
    values[points, 9] = -4
    values[points, 11] = 2
    values[points, 42] = 2
    result = scale_window_partition(values, tau, 256, 2.)
    assert result['first_possible_fully_postbreak_window_point'] == first
    assert result['eligible_windows'] == eligible
    assert result['negative_raw_hard_ceiling_rejection'] == eligible
    assert result['negative_raw_exceeds_admission_cut'] == 0


@pytest.mark.parametrize('block_values,coherent_windows', [
    ([-4., -4., -4., 4., -4., -4., -4., 4.], 5),
    ([-.1, -.2, -.3, .05, -.4, -.8, .07, -.1], 4)])
def test_opposing_signed_evidence_and_physical_ring_order_match_real_kernel(block_values, coherent_windows):
    from src.next7.scale import confirmation
    values = np.zeros((128, 45))
    physical = np.zeros((4, 2))
    result = np.zeros(18)
    for i, value in enumerate(block_values):
        point = 16 * (i + 1) - 1
        physical[i % 4, 0] = value
        confirmation(physical, i + 1, result)
        values[point, 15] = 1
        values[point, 9] = value
        values[point, 11] = result[3] / 2.
        values[point, 42] = 1.
    actual = scale_window_partition(values, 0, 128, 2.)
    assert actual['negative_raw_confirmation'] == 0
    assert actual['negative_coherent_but_below_raw_cut'] == coherent_windows
    assert actual['neither_direction_coherent'] == 5 - coherent_windows

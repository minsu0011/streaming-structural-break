"""Registered evaluation-only refinement of the frozen legacy recurrent state."""
import numpy as np
from src.next3.state_transfer import at, replaced, clone_arguments, assert_historical_constants_equal

LEGACY_BLOCKS = ('LEGACY_AR_FILTER', 'LEGACY_ARCH_PREVIOUS', 'LEGACY_LAG_RING',
                 'LEGACY_EWMA', 'LEGACY_CUSUM_POSITIVE', 'LEGACY_CUSUM_NEGATIVE', 'LEGACY_RUNNING_MEAN')


def transfer_legacy_arguments(destination, donor, blocks):
    blocks = tuple(blocks)
    if not blocks or len(blocks) != len(set(blocks)) or any(block not in LEGACY_BLOCKS for block in blocks):
        raise ValueError('Distinct registered legacy state blocks required')
    assert_historical_constants_equal(destination, donor)
    left_count, right_count = at(destination, (0, 2, 5)), at(donor, (0, 2, 5))
    if left_count.shape != right_count.shape or left_count.shape != (8,):
        raise ValueError('Frozen eight-counter legacy schema required')
    np.testing.assert_array_equal(left_count[[0, 4, 5, 6, 7]], right_count[[0, 4, 5, 6, 7]])
    result = clone_arguments(destination); source = clone_arguments(donor)
    for block in blocks:
        if block == 'LEGACY_AR_FILTER':
            result = replaced(result, (0, 0), at(source, (0, 0)))
        elif block == 'LEGACY_ARCH_PREVIOUS':
            result = replaced(result, (0, 1, 1), at(source, (0, 1, 1)))
        elif block == 'LEGACY_LAG_RING':
            result = replaced(result, (0, 2, 3), at(source, (0, 2, 3)))
            at(result, (0, 2, 5))[4] = at(source, (0, 2, 5))[4]
        elif block == 'LEGACY_EWMA':
            result = replaced(result, (0, 2, 4), at(source, (0, 2, 4)))
        else:
            index = {'LEGACY_CUSUM_POSITIVE': 1, 'LEGACY_CUSUM_NEGATIVE': 2, 'LEGACY_RUNNING_MEAN': 3}[block]
            at(result, (0, 2, 5))[index] = at(source, (0, 2, 5))[index]
    return result

"""Evaluation-only state interventions in the frozen shared BL80 recurrence."""
import numpy as np
from numba import njit
from src.next2.shared_blend_runtime import shared_kernel
from src.models.compact_trees import tree_margin

BLOCKS = {
    'LEGACY_ARCH_ABCD': ((0,),),
    'SHARED_AR_FILTER': ((1, 0),),
    'CONDITIONAL_VARIANCE': ((1, 2),),
    'J_SERIAL_EVIDENCE': ((1, 5),),
    'C_DEPENDENCE_LAGS': ((1, 7), (1, 8)),
    'C_DEPENDENCE_EVIDENCE': ((1, 13),),
}


def arrays(value):
    result = {}
    def visit(item):
        if isinstance(item, np.ndarray): result[id(item)] = item
        elif isinstance(item, (tuple, list)):
            for child in item: visit(child)
    visit(value)
    return result


def immutable_ids(args):
    # Trained forests and integer selectors are shared; all recurrent/H arrays
    # are copied even when they happen to be read-only in a particular caller.
    return set(arrays((args[-1], args[2:5], args[0][2][11])))


def clone_arguments(args):
    immutable = immutable_ids(args); memo = {}
    def visit(item):
        if isinstance(item, np.ndarray):
            if id(item) in immutable: return item
            if id(item) not in memo:
                value = np.array(item, copy=True, order='K'); value.setflags(write=item.flags.writeable)
                memo[id(item)] = value
                if np.shares_memory(value, item): raise AssertionError('Mutable state clone aliases its donor')
            return memo[id(item)]
        if isinstance(item, tuple): return tuple(visit(child) for child in item)
        if isinstance(item, list): return [visit(child) for child in item]
        return item
    result = visit(args)
    if set(arrays(result)) & (set(arrays(args))-immutable): raise AssertionError('Recurrent state identity retained')
    return result


def at(args, path):
    for index in path: args = args[index]
    return args


def replaced(args, path, value):
    result = list(args); index = path[0]
    result[index] = value if len(path) == 1 else replaced(result[index], path[1:], value)
    return tuple(result)


def assert_historical_constants_equal(left, right):
    paths = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 0, 0), (1, 0, 1), (1, 1), (1, 3), (1, 5, 0), (1, 5, 1),
        (1, 6), (1, 9), (1, 10), (1, 12), (1, 13, 0), (1, 13, 1)]
    paths += [(0, 2, k) for k in (0, 1, 2, 6, 7, 8, 9, 10, 11)]
    for path in paths: np.testing.assert_array_equal(at(left, path), at(right, path))
    for a, b in zip(arrays(left[-1]).values(), arrays(right[-1]).values()): np.testing.assert_array_equal(a, b)
    assert left[-1][-1] == right[-1][-1] == .8


def transferred_arguments(destination, donor, blocks):
    if len(destination) != 7 or len(destination[1]) != 14: raise ValueError('Frozen shared argument schema required')
    blocks = tuple(blocks)
    if len(blocks) != len(set(blocks)) or any(block not in BLOCKS for block in blocks):
        raise ValueError('Distinct registered recurrent blocks required')
    assert_historical_constants_equal(destination, donor)
    result = clone_arguments(destination); source = clone_arguments(donor)
    for block in blocks:
        for path in BLOCKS[block]: result = replaced(result, path, at(source, path))
    return result


@njit(cache=False)
def trace_components(args, points):
    result = np.empty((len(points), 3), dtype=np.float32)
    for k, point in enumerate(points):
        result[k, 2] = shared_kernel(float(point), args)
        margin_j = tree_margin(args[5], *args[-1][0])
        margin_c = tree_margin(args[1][-1][-1], *args[-1][1])
        result[k, 0] = np.float32(1./(1.+np.exp(-margin_j)))
        result[k, 1] = np.float32(1./(1.+np.exp(-margin_c)))
    return result


@njit(cache=False)
def trace_scores(args, points):
    result = np.empty(len(points), dtype=np.float32)
    for k, point in enumerate(points): result[k] = shared_kernel(float(point), args)
    return result

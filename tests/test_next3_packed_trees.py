import numpy as np
import pytest
from src.models.compact_trees import _arrays, tree_margins
from src.next3.packed_trees import NODE_DTYPE, pack_tree_args, packed_tree_margins


def example():
    return _arrays([0, 3], [0, -1, -1, 1, -1, -1],
        [.25, 0., 0., -0., 0., 0.], [1, -1, -1, 4, -1, -1],
        [2, -1, -1, 5, -1, -1], [0., .123, -.333, 0., 1.23, -3.21],
        [True, False, False, False, False, False], .031)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_packing_keeps_exact_margin_boundaries_missing_and_addition(dtype):
    tree = example()
    near = np.asarray([-.0, .0, .25, np.nextafter(dtype(.25), dtype(-np.inf)),
                       np.nextafter(dtype(.25), dtype(np.inf)), np.nan, np.inf, -np.inf], dtype=dtype)
    x = np.asarray([(a, b) for a in near for b in near], dtype=dtype)
    packed = pack_tree_args(tree.args())
    np.testing.assert_array_equal(tree_margins(x, *tree.args()), packed_tree_margins(x, *packed))
    assert NODE_DTYPE.itemsize == 16 and packed[1].flags.aligned
    assert not packed[0].flags.writeable and not packed[1].flags.writeable
    assert sum(a.nbytes for a in packed[:2]) < sum(a.nbytes for a in tree.args()[:-1])


@pytest.mark.parametrize('damage', ['cycle', 'child', 'feature', 'root', 'missing', 'float_index', 'nan_threshold', 'nan_leaf'])
def test_packing_rejects_malformed_or_narrowing_overflow(damage):
    tree = example()
    if damage == 'cycle': tree.left[0] = 0
    elif damage == 'child': tree.right[0] = 6
    elif damage == 'feature': tree.features[0] = 32768
    elif damage == 'root': tree.roots[0] = -1
    elif damage == 'missing': tree.missing_left = np.full(6, 2, dtype=np.uint8)
    elif damage == 'float_index': tree.left = tree.left.astype(float)
    elif damage == 'nan_threshold': tree.thresholds[0] = np.nan
    elif damage == 'nan_leaf': tree.values[1] = np.nan
    with pytest.raises(ValueError): pack_tree_args(tree.args())


def test_packing_rejects_unreachable_and_oversized_forests():
    tree = example(); tree.roots = tree.roots[:1]
    with pytest.raises(ValueError, match='Unreachable'): pack_tree_args(tree.args())
    n = 32768
    tree = _arrays([0], [-1]*n, [0.]*n, [-1]*n, [-1]*n, [0.]*n, [False]*n, 0.)
    with pytest.raises(ValueError, match='int16'): pack_tree_args(tree.args())

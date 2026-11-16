"""Bounded, validated, prediction-exact 16-byte binary tree nodes."""
import numpy as np
from numba import njit


NODE_DTYPE = np.dtype([
    ('feature', '<i2'), ('left', '<i2'), ('right', '<i2'),
    ('missing_left', 'u1'), ('reserved', 'u1'), ('payload', '<f8'),
], align=True)
assert NODE_DTYPE.itemsize == 16 and NODE_DTYPE.fields['payload'][1] == 8


def pack_tree_args(args):
    """Reject unsupported input before narrowing; preserve only live node data."""
    roots, features, thresholds, left, right, values, missing, bias = args
    arrays = (roots, features, thresholds, left, right, values, missing)
    if any(not isinstance(a, np.ndarray) or a.ndim != 1 for a in arrays):
        raise ValueError('One-dimensional numeric tree arrays required')
    n = len(features)
    if not 0 < n <= np.iinfo(np.int16).max or not len(roots):
        raise ValueError('Packed tree size exceeds fixed int16 contract')
    if any(len(a) != n for a in arrays[2:]):
        raise ValueError('Tree node array lengths differ')
    if any(a.dtype.kind not in 'iu' for a in (roots, features, left, right)):
        raise ValueError('Integer tree indices required')
    if np.any(features < -1) or np.any(features > np.iinfo(np.int16).max):
        raise ValueError('Unsupported feature index')
    if np.any(roots < 0) or np.any(roots >= n):
        raise ValueError('Root outside tree node array')
    split = features >= 0
    if (np.any(left[split] < 0) or np.any(left[split] >= n) or
            np.any(right[split] < 0) or np.any(right[split] >= n)):
        raise ValueError('Child outside tree node array')
    if missing.dtype.kind not in 'biu' or np.any((missing != 0) & (missing != 1)):
        raise ValueError('Boolean missing direction required')
    if (thresholds.dtype != np.dtype('float64') or values.dtype != np.dtype('float64') or
            not np.isfinite(thresholds[split]).all() or
            not np.isfinite(values[~split]).all() or not np.isfinite(bias)):
        raise ValueError('Finite float64 live payloads and bias required')
    # Every node must belong to exactly one acyclic tree. Do not depend on a
    # prediction loop to detect corruption, and never silently narrow indices.
    seen = np.zeros(n, dtype=bool)
    for root in roots:
        stack = [int(root)]
        while stack:
            node = stack.pop()
            if seen[node]:
                raise ValueError('Repeated node or cyclic tree')
            seen[node] = True
            if split[node]:
                stack.extend((int(right[node]), int(left[node])))
    if not seen.all():
        raise ValueError('Unreachable node in packed forest')
    nodes = np.zeros(n, dtype=NODE_DTYPE)
    nodes['feature'] = features
    nodes['left'] = -1
    nodes['right'] = -1
    nodes['left'][split] = left[split]
    nodes['right'][split] = right[split]
    nodes['missing_left'] = missing
    nodes['payload'] = np.where(split, thresholds, values)
    packed_roots = roots.astype(np.int16, copy=True)
    packed_roots.setflags(write=False)
    nodes.setflags(write=False)
    return packed_roots, nodes, float(bias)


@njit(cache=False)
def packed_tree_margin(x, roots, nodes, bias):
    result = bias
    for root in roots:
        node = root
        while nodes[node]['feature'] >= 0:
            record = nodes[node]
            value = x[record['feature']]
            go_left = bool(record['missing_left']) if np.isnan(value) else value <= record['payload']
            node = record['left'] if go_left else record['right']
        result += nodes[node]['payload']
    return result


@njit(cache=False)
def packed_tree_margins(x, roots, nodes, bias):
    out = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        out[i] = packed_tree_margin(x[i], roots, nodes, bias)
    return out

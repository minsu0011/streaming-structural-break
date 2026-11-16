"""Independent live-node comparison and deterministic artificial tree witnesses.

These feature vectors test a numeric tree engine. They are not claims about
reachable recurrent H/O states, model quality, or a compiler's formal semantics.
"""
import hashlib
import numpy as np


def branch(value, threshold, missing_left):
    # A NumPy float32 scalar and a Python float otherwise use weak scalar
    # promotion, which may narrow a nonrepresentable original float64 threshold.
    return bool(missing_left) if np.isnan(value) else float(value) <= float(threshold)


def equal_float_bits(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def certify_nodes(original, original_columns, packed, packed_columns):
    roots, features, thresholds, left, right, values, missing, bias = original
    packed_roots, nodes, packed_bias = packed
    old_columns = np.asarray(original_columns)
    new_columns = np.asarray(packed_columns)
    n = len(features)
    if n == 0 or any(np.asarray(a).shape != (n,) for a in (features, thresholds, left, right, values, missing)):
        raise ValueError('Original node arrays must have one matching nonempty dimension')
    for columns in (old_columns, new_columns):
        if (columns.ndim != 1 or not len(columns) or columns.dtype.kind not in 'iu'
                or np.any(columns < 0) or len(np.unique(columns)) != len(columns)):
            raise ValueError('Unique nonnegative integer scientific columns required')
    if (nodes.shape != (n,) or nodes.dtype.itemsize != 16
            or nodes.dtype.names != ('feature', 'left', 'right', 'missing_left', 'reserved', 'payload')
            or nodes.dtype.fields['payload'][1] != 8):
        raise ValueError('Expected actual 16-byte packed nodes')
    if not np.array_equal(roots, packed_roots) or not equal_float_bits(bias, packed_bias):
        raise ValueError('Root order or bias bits differ')
    if np.any(features < -1) or np.any(nodes['feature'] < -1):
        raise ValueError('Only -1 marks a leaf')
    split = features >= 0
    if not np.array_equal(split, nodes['feature'] >= 0):
        raise ValueError('Split/leaf kinds differ')
    if np.any(features[split] >= len(old_columns)) or np.any(nodes['feature'][split] >= len(new_columns)):
        raise ValueError('Live feature index outside its column map')
    if not np.array_equal(old_columns[features[split]], new_columns[nodes['feature'][split]]):
        raise ValueError('Scientific feature identity differs after column remapping')
    if not np.array_equal(left[split], nodes['left'][split]) or not np.array_equal(right[split], nodes['right'][split]):
        raise ValueError('Live child topology differs')
    if not np.array_equal(missing[split], nodes['missing_left'][split]):
        raise ValueError('Live missing-value direction differs')
    if not equal_float_bits(thresholds[split], nodes['payload'][split]):
        raise ValueError('Live threshold payload bits differ')
    if not equal_float_bits(values[~split], nodes['payload'][~split]):
        raise ValueError('Live leaf payload bits differ')
    if np.any(nodes['reserved'] != 0): raise ValueError('Packed reserved bytes are not zero')
    if not np.isfinite(thresholds[split]).all() or not np.isfinite(values[~split]).all() or not np.isfinite(bias):
        raise ValueError('Live tree values must be finite')
    visited = set()
    for root in roots:
        pending = [int(root)]
        while pending:
            node = pending.pop()
            if node < 0 or node >= n or node in visited:
                raise ValueError('Duplicated, cyclic or out-of-bounds original node')
            visited.add(node)
            if split[node]: pending.extend((int(left[node]), int(right[node])))
    if len(visited) != n: raise ValueError('Node absent from the original forest')
    return {'nodes':n, 'trees':len(roots), 'split_nodes':int(split.sum()), 'leaf_nodes':int((~split).sum()),
        'used_scientific_columns':len(np.unique(old_columns[features[split]])),
        'all_live_node_payload_bits_and_column_identities_exact':True,
        'root_order_and_bias_bits_exact':True, 'topology_reachable_nodes':len(visited)}


def path_satisfied(vector, path):
    return all(branch(vector[feature], threshold, missing_left) == go_left
        for feature, threshold, go_left, missing_left in path)


def path_witness(path, feature_count, dtype):
    """Find a representable finite/infinite or NaN witness for exact ancestors."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Only the two explicit IEEE floating dtypes are supported')
    result = np.zeros(feature_count, dtype=dtype)
    constraints = {}
    for feature, threshold, go_left, missing_left in path:
        if not 0 <= feature < feature_count or not np.isfinite(threshold):
            raise ValueError('Finite threshold and in-range scientific feature required')
        item = constraints.setdefault(feature, {'lower':None, 'upper':None, 'nan_allowed':True})
        item['nan_allowed'] &= go_left == missing_left
        key = 'upper' if go_left else 'lower'
        old = item[key]
        item[key] = threshold if old is None else (min(old, threshold) if go_left else max(old, threshold))
    for feature, item in constraints.items():
        lower, upper = item['lower'], item['upper']
        def allowed(value):
            return (lower is None or float(value) > lower) and (upper is None or float(value) <= upper)
        candidate = dtype.type(0.)
        if not allowed(candidate):
            with np.errstate(over='ignore', invalid='ignore'):
                if upper is not None:
                    candidate = dtype.type(upper)
                    if float(candidate) > upper:
                        candidate = np.nextafter(candidate, dtype.type(-np.inf), dtype=dtype)
                else:
                    candidate = dtype.type(lower)
                    if float(candidate) <= lower:
                        candidate = np.nextafter(candidate, dtype.type(np.inf), dtype=dtype)
        if allowed(candidate): result[feature] = candidate
        elif item['nan_allowed']: result[feature] = dtype.type(np.nan)
        else: return None
    if not path_satisfied(result, path): raise RuntimeError('Witness construction contradicted its exact path')
    return result


def forest_paths(original, columns):
    roots, features, thresholds, left, right, _, missing, _ = original
    for root in roots:
        pending = [(int(root), ())]
        while pending:
            node, path = pending.pop()
            yield int(root), node, path
            if features[node] >= 0:
                feature = int(columns[features[node]])
                threshold = float(thresholds[node]); direction = bool(missing[node])
                pending.append((int(right[node]), path+((feature, threshold, False, direction),)))
                pending.append((int(left[node]), path+((feature, threshold, True, direction),)))


def walk_tree(vector, root, original, columns):
    _, features, thresholds, left, right, _, missing, _ = original
    node = int(root); visited = []
    while features[node] >= 0:
        if node in visited: raise ValueError('Cyclic tree during witness check')
        visited.append(node)
        value = vector[int(columns[features[node]])]
        node = int(left[node] if branch(value, thresholds[node], missing[node]) else right[node])
    visited.append(node)
    return node, visited


def artificial_probes(original, columns, feature_count, dtype):
    """Attempt every leaf and six local values at every split; retain omissions."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')): raise ValueError('Explicit float dtype required')
    _, features, thresholds, _, _, _, _, _ = original
    vectors = []; records = []
    for root, node, path in forest_paths(original, columns):
        witness = path_witness(path, feature_count, dtype)
        if features[node] < 0:
            attempts = [('LEAF_PATH', None)]
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                threshold = dtype.type(thresholds[node])
                before = np.nextafter(threshold, dtype.type(-np.inf), dtype=dtype)
                after = np.nextafter(threshold, dtype.type(np.inf), dtype=dtype)
            attempts = [('NEGATIVE_INFINITY', dtype.type(-np.inf)), ('POSITIVE_INFINITY', dtype.type(np.inf)),
                ('NAN', dtype.type(np.nan)), ('CAST_THRESHOLD', threshold),
                ('NEIGHBOR_BELOW_CAST', before), ('NEIGHBOR_ABOVE_CAST', after)]
        for kind, value in attempts:
            vector = None if witness is None else witness.copy()
            if vector is not None and value is not None:
                vector[int(columns[features[node]])] = value
            reachable = vector is not None and path_satisfied(vector, path)
            record = {'root':root, 'target_node':node, 'case_kind':kind, 'ancestor_depth':len(path),
                'reachable_for_this_dtype':bool(reachable), 'matrix_row':None, 'input_sha256':None,
                'omission_reason':None if reachable else ('NO_ANCESTOR_WITNESS' if witness is None else 'LOCAL_VALUE_VIOLATES_ANCESTOR')}
            if reachable:
                leaf, visited = walk_tree(vector, root, original, columns)
                if node not in visited or (kind == 'LEAF_PATH' and leaf != node):
                    raise RuntimeError('Artificial probe did not reach its certified node')
                record.update(matrix_row=len(vectors), input_sha256=hashlib.sha256(vector.tobytes()).hexdigest())
                vectors.append(vector)
            records.append(record)
    matrix = np.stack(vectors) if vectors else np.empty((0, feature_count), dtype=dtype)
    return matrix, records

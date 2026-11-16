import numpy as np
import pytest
from src.models.compact_trees import _arrays, tree_margins
from src.next3.packed_trees import pack_tree_args, packed_tree_margins
from src.next3.tree_certificate import artificial_probes, branch, certify_nodes, path_satisfied, path_witness


def forest():
    return _arrays([0, 3], [0,-1,-1,1,-1,-1], [.25,0,0,-0.,0,0], [1,-1,-1,4,-1,-1],
        [2,-1,-1,5,-1,-1], [0,.125,-.25,0,.0625,-.03125], [True,False,False,False,False,False], .03125)


def permuted(tree):
    args = tree.args(); features = args[1].copy(); live = features >= 0; features[live] = 1-features[live]
    return pack_tree_args((args[0],features,*args[2:]))


def test_certificate_accepts_different_selected_column_order_with_same_scientific_identity():
    tree = forest(); report = certify_nodes(tree.args(), np.array([2,0]), permuted(tree), np.array([0,2]))
    assert report['nodes'] == 6 and report['leaf_nodes'] == 4 and report['split_nodes'] == 2
    assert report['used_scientific_columns'] == 2 and report['root_order_and_bias_bits_exact']


@pytest.mark.parametrize('damage', ['column','child','missing','threshold','leaf','signed_zero','root_order','reserved'])
def test_certificate_rejects_live_representation_damage(damage):
    tree = forest(); roots, nodes, bias = permuted(tree); roots = roots.copy(); nodes = nodes.copy()
    columns = np.array([0,2])
    if damage == 'column': columns = np.array([2,0])
    elif damage == 'child': nodes['left'][0] = 2
    elif damage == 'missing': nodes['missing_left'][0] = 0
    elif damage == 'threshold': nodes['payload'][0] = np.nextafter(.25,np.inf)
    elif damage == 'leaf': nodes['payload'][1] = np.nextafter(.125,np.inf)
    elif damage == 'signed_zero': nodes['payload'][3] = .0
    elif damage == 'root_order': roots = roots[::-1].copy()
    elif damage == 'reserved': nodes['reserved'][1] = 1
    with pytest.raises(ValueError): certify_nodes(tree.args(), np.array([2,0]), (roots,nodes,bias), columns)


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_all_leaf_witnesses_and_local_probes_match_both_actual_engines(dtype):
    tree = forest(); old = np.array([2,0]); new = np.array([0,2]); packed = permuted(tree)
    certify_nodes(tree.args(), old, packed, new)
    matrix, records = artificial_probes(tree.args(), old, 3, dtype)
    assert matrix.dtype == dtype and len(records) == len(matrix) == 16
    assert sum(row['case_kind'] == 'LEAF_PATH' for row in records) == 4
    expected = tree_margins(matrix[:,old].copy(), *tree.args())
    actual = packed_tree_margins(matrix[:,new].copy(), *packed)
    assert expected.tobytes() == actual.tobytes()


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_witness_respects_strict_right_and_inclusive_left(dtype):
    value = dtype(.25); upper = float(np.nextafter(value,dtype(np.inf)))
    path = [(0,float(value),False,True),(0,upper,True,False)]
    witness = path_witness(path,1,dtype)
    assert witness is not None and float(witness[0]) == upper and path_satisfied(witness,path)


def test_narrow_interval_has_float64_witness_but_no_float32_witness():
    lower = .25; upper = np.nextafter(lower,np.inf)
    path = [(0,lower,False,True),(0,upper,True,False)]
    assert path_witness(path,1,np.float64) is not None
    assert path_witness(path,1,np.float32) is None


def test_nan_only_path_is_distinct_from_impossible_path():
    nan_path = [(0,0.,True,True),(0,1.,False,False)]
    witness = path_witness(nan_path,1,np.float32)
    assert witness is not None and np.isnan(witness[0]) and path_satisfied(witness,nan_path)
    assert path_witness([(0,0.,True,False),(0,1.,False,False)],1,np.float32) is None


def test_unreachable_local_values_are_retained_and_not_counted_as_executed():
    tree = _arrays([0],[0,0,-1,-1,-1],[0.,1.,0.,0.,0.],[1,2,-1,-1,-1],
        [4,3,-1,-1,-1],[0,0,.1,.2,.3],[False,False,False,False,False],0.)
    matrix, records = artificial_probes(tree.args(),np.array([0]),1,np.float32)
    impossible_leaf = [row for row in records if row['target_node'] == 3 and row['case_kind'] == 'LEAF_PATH']
    assert len(impossible_leaf) == 1 and not impossible_leaf[0]['reachable_for_this_dtype']
    assert impossible_leaf[0]['matrix_row'] is None and impossible_leaf[0]['input_sha256'] is None
    assert len(matrix) == sum(row['reachable_for_this_dtype'] for row in records) < len(records)


def test_invalid_feature_and_integer_dtype_are_rejected():
    with pytest.raises(ValueError): path_witness([(2,0.,True,True)],1,np.float32)
    with pytest.raises(ValueError): path_witness([],1,np.int32)


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_scalar_oracle_keeps_original_double_threshold_below_representable_value(dtype):
    value = dtype(.25)
    below = float(np.nextafter(np.float64(.25),np.float64(-np.inf)))
    above = float(np.nextafter(np.float64(.25),np.float64(np.inf)))
    assert branch(value,below,False) is False
    assert branch(value,above,False) is True
    witness = path_witness([(0,below,False,True)],1,dtype)
    assert witness is not None and float(witness[0]) > below and path_satisfied(witness,[(0,below,False,True)])

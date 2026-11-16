from pathlib import Path
import ast
import copy
import hashlib
import json
import zlib
import numpy as np
import pytest
from src.next3.lean_model import loads, encode_record, feature_contract, tree_args, MAGIC
from src.next3.lean_shared_runtime import PreparedLeanSharedPredictor, accounting as lean_accounting
from src.next3.packed_shared_runtime import PreparedPackedSharedPredictor, accounting as packed_accounting
from src.next2.canonical_model import loads as old_loads

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def content():
    return (ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2').read_bytes()


@pytest.fixture(scope='module')
def contract():
    return feature_contract()


def test_extracted_definitions_preserve_frozen_function_bodies():
    build = json.loads((ROOT/'configs/next3/LEAN_NUMERIC_BUILD.json').read_text(encoding='utf-8'))
    generated_source = (ROOT/'src/next3/lean_numeric.py').read_text(encoding='utf-8')
    generated = {n.name: n for n in ast.parse(generated_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for receipt in build['definitions']:
        original_source = (ROOT/receipt['source']).read_text(encoding='utf-8')
        original = next(n for n in ast.parse(original_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == receipt['name'])
        actual = generated[receipt['name']]
        assert ast.dump(ast.Module(body=original.body, type_ignores=[])) == ast.dump(ast.Module(body=actual.body, type_ignores=[]))
        assert hashlib.sha256((ROOT/receipt['source']).read_bytes()).hexdigest() == receipt['original_source_sha256']


def test_classless_roundtrip_and_prepared_arrays_exact(content, contract):
    record = loads(content, contract=contract); assert encode_record(record) == content
    reference = PreparedPackedSharedPredictor(old_loads(content), variant='PACKED_SHARED', resident=False)
    lean = PreparedLeanSharedPredictor(record, resident=False, contract=contract)
    for name in ('old_columns', 'old_positions', 'new_positions', 'new_columns'):
        np.testing.assert_array_equal(getattr(reference.primary, name), getattr(lean.primary, name))
        assert not getattr(lean.primary, name).flags.writeable
    assert reference.primary.names == lean.primary.names
    for role in ('primary', 'complement'):
        for a, b in zip(getattr(reference, role).tree_args, getattr(lean, role).tree_args):
            np.testing.assert_array_equal(a, b)
    assert set(vars(lean.model)) == {'weight'}


@pytest.mark.parametrize('corruption', ['empty', 'truncated', 'trailing', 'checksum', 'length', 'noncanonical'])
def test_classless_envelope_rejects_corruption(content, contract, corruption):
    offset = len(MAGIC)
    altered = {'empty': b'', 'truncated': content[:-1], 'trailing': content+b'extra',
        'checksum': content[:offset+8]+b'0'*32+content[offset+40:],
        'length': MAGIC+(8_000_001).to_bytes(8, 'little')+content[offset+8:]}.get(corruption)
    if corruption == 'noncanonical':
        record = loads(content, contract=contract)
        payload = json.dumps({'format': 'NEXT2_CANONICAL_V1', 'model': record}, indent=1).encode()
        altered = MAGIC+len(payload).to_bytes(8, 'little')+hashlib.sha256(payload).digest()+zlib.compress(payload, 9)
    with pytest.raises((ValueError, UnicodeError)): loads(altered, contract=contract)


@pytest.mark.parametrize('corruption', ['feature_name', 'weight', 'source', 'float_index', 'overflow', 'bad_child', 'repeat_root', 'nan', 'extra_key', 'missing_boolean'])
def test_classless_contract_and_graph_fail_closed(content, contract, corruption):
    record = loads(content, contract=contract); tree = record['primary']['tree']
    if corruption == 'feature_name': record['primary']['names'][0] = 'unregistered'
    elif corruption == 'weight': record['weight'] = .9
    elif corruption == 'source': record['primary']['engine_hash'] = '0'*64
    elif corruption == 'float_index': tree['features'][0] = .0
    elif corruption == 'overflow': tree['features'][0] = 65536
    elif corruption == 'bad_child': tree['left'][0] = 0
    elif corruption == 'repeat_root': tree['roots'].append(tree['roots'][0])
    elif corruption == 'nan': tree['values'][0] = float('nan')
    elif corruption == 'extra_key': record['primary']['unregistered'] = 1
    elif corruption == 'missing_boolean': tree['missing_left'][0] = 1
    with pytest.raises(ValueError): loads(encode_record(record), contract=contract)


@pytest.mark.parametrize('length', [128, 512, 1024])
def test_lean_streaming_features_predictions_and_allocations_exact(content, contract, length):
    index = np.arange(length+300, dtype=np.float64)
    values = (np.sin(index*.37)+.2*np.cos(index*.11)+.1*np.sin(index*.017)).astype(np.float32)
    packed = PreparedPackedSharedPredictor(old_loads(content), variant='PACKED_SHARED').new_state(values[:length])
    lean = PreparedLeanSharedPredictor(loads(content, contract=contract), contract=contract).new_state(values[:length])
    before = lean.state_array_bytes
    for point in values[length:]:
        assert packed.predict_one(point) == lean.predict_one(point)
        np.testing.assert_array_equal(packed.base.output.view(np.uint32), lean.output.view(np.uint32))
        np.testing.assert_array_equal(packed.base.shared.dependence_output.view(np.uint32), lean.shared.dependence_output.view(np.uint32))
    assert lean.state_array_bytes == before
    a, _, _ = packed_accounting(packed); b, _, _ = lean_accounting(lean)
    for key in ('private_per_series_numeric_bytes', 'model_and_prepared_shared_numeric_bytes', 'total_one_stream_plus_one_model_numeric_bytes'):
        assert a[key] == b[key]

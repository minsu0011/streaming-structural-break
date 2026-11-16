"""Bounded classless decoding of the unchanged frozen 80/20 canonical format."""
from pathlib import Path
import hashlib
import json
import zlib
import numpy as np
from src.next3.packed_trees import pack_tree_args

ROOT = Path(__file__).resolve().parents[2]
MAGIC = b'SB_NEXT2_MODEL_V1\n'
MAX_PAYLOAD = 8_000_000
TREE_FIELDS = ('roots', 'features', 'thresholds', 'left', 'right', 'values', 'missing_left')


def feature_contract(*, verify_sources=True):
    contract = json.loads((ROOT/'configs/next3/LEAN_FEATURE_CONTRACT.json').read_text(encoding='utf-8'))
    if contract['format'] != 'NEXT3_LEAN_FROZEN_80_20_V1':
        raise ValueError('Unknown lean scientific feature contract')
    if verify_sources:
        for relative, expected in contract['scientific_sources_sha256'].items():
            path = (ROOT/relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen scientific source contract changed: '+relative)
        build = json.loads((ROOT/'configs/next3/LEAN_NUMERIC_BUILD.json').read_text(encoding='utf-8'))
        if hashlib.sha256((ROOT/'src/next3/lean_numeric.py').read_bytes()).hexdigest() != build['generated_source_sha256']:
            raise RuntimeError('Extracted numerical source changed')
        inputs = json.loads((ROOT/'configs/next3/LEAN_RUNTIME_INPUTS.json').read_text(encoding='utf-8'))
        for relative, expected in inputs['files_sha256'].items():
            path = (ROOT/relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen runtime configuration changed: '+relative)
    return contract


def encode_record(record):
    payload = json.dumps({'format': 'NEXT2_CANONICAL_V1', 'model': record}, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    if not 0 < len(payload) <= MAX_PAYLOAD: raise ValueError('Canonical payload size invalid')
    return MAGIC+len(payload).to_bytes(8, 'little')+hashlib.sha256(payload).digest()+zlib.compress(payload, 9)


def integer_array(values, *, lower, upper, maximum, name):
    if (not isinstance(values, list) or not 0 < len(values) <= maximum or
            any(type(value) is not int or not lower <= value <= upper for value in values)):
        raise ValueError('Bounded integer array required: '+name)
    return np.asarray(values, dtype=np.int32)


def tree_args(component):
    names = component['names']
    columns = integer_array(component['columns'], lower=0, upper=len(names)-1, maximum=len(names), name='columns')
    if len(np.unique(columns)) != len(columns): raise ValueError('Duplicate selected columns')
    record = component['tree']
    if not isinstance(record, dict) or set(record) != set(TREE_FIELDS)|{'bias'}:
        raise ValueError('Exact finite tree schema required')
    features = integer_array(record['features'], lower=-1, upper=len(columns)-1, maximum=32767, name='features')
    size = len(features)
    roots = integer_array(record['roots'], lower=0, upper=size-1, maximum=1000, name='roots')
    left = integer_array(record['left'], lower=-1, upper=size-1, maximum=size, name='left')
    right = integer_array(record['right'], lower=-1, upper=size-1, maximum=size, name='right')
    numeric = []
    for name in ('thresholds', 'values'):
        values = record[name]
        if (not isinstance(values, list) or len(values) != size or
                any(type(value) not in (int, float) for value in values)):
            raise ValueError('Finite numeric tree array required: '+name)
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all(): raise ValueError('Nonfinite tree parameter')
        numeric.append(array)
    missing = record['missing_left']
    if not isinstance(missing, list) or len(missing) != size or any(type(value) is not bool for value in missing):
        raise ValueError('Boolean missing directions required')
    bias = record['bias']
    if type(bias) not in (int, float) or not np.isfinite(bias): raise ValueError('Finite numeric bias required')
    if len(left) != size or len(right) != size: raise ValueError('Node array dimensions differ')
    interior = features >= 0; indices = np.arange(size)
    if any(np.any(children[interior] <= indices[interior]) for children in (left, right)):
        raise ValueError('Canonical forward child order required')
    mapped = features.copy(); mapped[interior] = columns[features[interior]]
    result = (roots, mapped, numeric[0], left, right, numeric[1], np.asarray(missing, dtype=np.bool_), float(bias))
    # The packer also rejects duplicate, cyclic and unreachable nodes before JIT.
    pack_tree_args(result)
    return result


def validate_record(record, contract):
    metadata = contract['model_metadata']
    if not isinstance(record, dict) or set(record) != set(metadata):
        raise ValueError('Exact frozen blend schema required')
    for key in ('class', 'implementation_sha256', 'weight'):
        if record[key] != metadata[key]: raise ValueError('Frozen blend contract differs: '+key)
    if type(record['weight']) is not float: raise ValueError('Explicit frozen float weight required')
    for role in ('primary', 'complement'):
        component = record[role]; expected = metadata[role]
        if not isinstance(component, dict) or set(component) != set(expected)|{'tree', 'columns'}:
            raise ValueError('Exact component schema required')
        if {k: v for k, v in component.items() if k not in ('tree', 'columns')} != expected:
            raise ValueError('Frozen named feature or scientific source contract differs: '+role)
        tree_args(component)
    return record


def loads(content, *, contract=None):
    if not isinstance(content, bytes) or len(content) > MAX_PAYLOAD or not content.startswith(MAGIC):
        raise ValueError('Canonical model envelope invalid')
    offset = len(MAGIC); length = int.from_bytes(content[offset:offset+8], 'little')
    checksum = content[offset+8:offset+40]
    if not 0 < length <= MAX_PAYLOAD or len(checksum) != 32: raise ValueError('Canonical payload length invalid')
    decoder = zlib.decompressobj()
    try: payload = decoder.decompress(content[offset+40:], length+1)
    except zlib.error as error: raise ValueError('Invalid canonical compression') from error
    if (len(payload) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or
            hashlib.sha256(payload).digest() != checksum):
        raise ValueError('Canonical checksum or decompression boundary invalid')
    item = json.loads(payload)
    if not isinstance(item, dict) or set(item) != {'format', 'model'} or item['format'] != 'NEXT2_CANONICAL_V1':
        raise ValueError('Exact canonical envelope schema required')
    record = item['model']
    if encode_record(record) != content: raise ValueError('Noncanonical encoding')
    return validate_record(record, feature_contract() if contract is None else contract)


def load(path, *, contract=None):
    return loads(Path(path).read_bytes(), contract=contract)

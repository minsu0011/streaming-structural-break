"""Validation-only compatibility with the frozen C205 CV metadata omission.

The original CV C205 bundles contain no old ARCH features and leave their unused
old_arch_hash field empty. The final all-DEV bundle carries the frozen hash.
Both retain identical complete dependence feature/source contracts. This helper
does not alter any serialized record and is never imported by public inference.
"""
import copy
from src.next3.lean_model import feature_contract, loads
from src.next3.lean_shared_runtime import PreparedLeanSharedPredictor


def prepare_heldout_record(content, *, variant='LEAN_PACKED_SHARED'):
    if variant != 'LEAN_PACKED_SHARED': raise ValueError('One fixed E003 variant')
    # The canonical class loader is used only by validation, never cold inference.
    from src.next2.canonical_model import loads as canonical_loads, dumps
    frozen = canonical_loads(content); assert dumps(frozen) == content
    contract = feature_contract()
    unused_hash = frozen.complement.old_arch_hash
    expected = contract['model_metadata']['complement']
    if unused_hash == '':
        if any(name.startswith('arch8__') for name in frozen.complement.names):
            raise ValueError('Absent old ARCH hash cannot validate an old-feature component')
        if tuple(frozen.complement.names) != tuple(expected['names']):
            raise ValueError('Only the exact complete C205 dependence names permit the unused metadata omission')
        contract = copy.deepcopy(contract)
        contract['model_metadata']['complement']['old_arch_hash'] = ''
    elif unused_hash != expected['old_arch_hash']:
        raise ValueError('Unknown old ARCH hash remains forbidden')
    record = loads(content, contract=contract)
    return PreparedLeanSharedPredictor(record, contract=contract)

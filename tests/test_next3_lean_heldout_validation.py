from pathlib import Path
import json
import numpy as np
import pytest
from src.next3.lean_heldout_validation import prepare_heldout_record
from src.next3.lean_model import encode_record, loads, feature_contract
from src.next2.canonical_model import dumps, record
from src.next2.blend import FixedBlendBundle, source_hash
from src.next2.io import model_class
from src.next3.packed_shared_runtime import PreparedPackedSharedPredictor

ROOT = Path(__file__).resolve().parents[1]


def test_actual_original_fold_legacy_metadata_and_live_arrays_match():
    components = []
    for prefix, name in (('next', 'J102_SERIAL_VARIANCE_512'), ('next2', 'C205_DEPENDENCE')):
        folder = ROOT/'artifacts'/prefix/'experiments'/name
        candidate = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))['candidate']
        components.append(model_class(candidate).load(folder/'full/fold_0.joblib'))
    model = FixedBlendBundle(*components, .8, source_hash()); assert model.complement.old_arch_hash == ''
    content = dumps(model)
    with pytest.raises(ValueError): loads(content)
    lean = prepare_heldout_record(content); packed = PreparedPackedSharedPredictor(model, variant='PACKED_SHARED')
    for role in ('primary', 'complement'):
        for a, b in zip(getattr(lean, role).tree_args, getattr(packed, role).tree_args): np.testing.assert_array_equal(a, b)
    assert dumps(model) == content


@pytest.mark.parametrize('field', ['old_arch_hash', 'engine_hash', 'names'])
def test_unknown_live_or_legacy_contract_is_still_rejected(field):
    content = (ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2').read_bytes()
    model = loads(content)
    if field == 'names': model['complement']['names'][0] = 'arch8__unregistered'
    else: model['complement'][field] = '0'*64
    with pytest.raises((ValueError, RuntimeError)): prepare_heldout_record(encode_record(model))

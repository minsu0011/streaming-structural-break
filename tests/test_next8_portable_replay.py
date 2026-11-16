"""Portable data exclusion and true saved-state replay, including nonempty heads."""
import src.next8
from src.next8.rawfree_boundary import denied_path
from src.next8.portable_replay import run
from src.next8.streaming_runtime import PreparedJoint
from tests.test_next8_streaming_runtime import actual_cache, fitted_fixture


def test_portable_boundary_blocks_data_and_missing_original_assets(tmp_path):
    original = tmp_path / 'original'
    payload = original / 'delivery' / 'extract'
    output = payload / 'replay'
    assert denied_path(payload / 'data' / 'any.npy', payload, original, output)
    assert denied_path(original / 'data' / 'cache.npy', payload, original, output)
    assert denied_path(original / 'artifacts' / 'missing.joblib', payload, original, output)
    assert denied_path(payload / 'artifacts' / 'model.joblib', payload, original, output) is None
    assert denied_path(original / '.venv' / 'Lib' / 'module.py', payload, original, output) is None
    assert denied_path(output / 'PREDICTIONS.npz', payload, original, output) is None


def test_portable_replay_exercises_nonempty_learned_graph_head(actual_cache):
    cache, h, online = actual_cache
    model, selection, graph, _ = fitted_fixture(cache, 'M5')
    prepared = PreparedJoint(model, selection, 'M5', {0: graph}, resident=False)
    values, receipt = run(prepared.new_state, h, online[:289])
    assert len(values) == 289
    assert receipt['pickle_resume_at73_bitwise'] and receipt['future_suffix_prefix73_bitwise']
    assert prepared.direct_features


def test_portable_replay_preserves_resident_BL80_snapshot():
    from src.next8.common import ROOT, BASE, read_json, digest
    from src.next5.model import FeatureModel
    from src.next5.runtime import SharedResearchBaseline
    from src.next8.synthetic_controls import matched_path
    provenance = read_json(BASE / 'BASELINE_PROVENANCE_NEXT8.json')
    models = []
    for kind in ('J102', 'C205'):
        name = f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_0/model.joblib'
        assert digest(ROOT / name) == provenance['files_sha256'][name]
        models.append(FeatureModel.load(ROOT / name))
    prepared = SharedResearchBaseline(*models)
    h, online = matched_path('Gaussian', 0, 512)
    values, receipt = run(prepared.new_state, h, online[:289])
    assert len(values) == 289 and receipt['fresh_scalar_chunk_bitwise']

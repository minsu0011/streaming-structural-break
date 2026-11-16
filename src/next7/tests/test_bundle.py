import builtins
import io
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from src.next7.common import ROOT, BASE, digest
from src.next7.bundle import load


@pytest.fixture
def bundle_copy(tmp_path):
    target = tmp_path / 'bundle'
    shutil.copytree(BASE / 'research_bundles/fold_0', target)
    return target


def rewrite_manifest(path, change):
    file = path / 'MANIFEST.json'
    value = json.loads(file.read_text(encoding='utf-8'))
    change(value)
    file.write_text(json.dumps(value), encoding='utf-8')


def test_role_and_raw_qualified_mappings_cannot_be_reassigned(bundle_copy):
    rewrite_manifest(bundle_copy, lambda m: m.update(role='PRIMARY'))
    with pytest.raises(ValueError, match='primary role'):
        load(bundle_copy)
    rewrite_manifest(bundle_copy, lambda m: m.update(role='RESEARCH_FOLD_ONLY_NOT_PROMOTED'))
    rewrite_manifest(bundle_copy, lambda m: m['qualified_state_by_recipe'].update(G2_FULL=False))
    with pytest.raises(ValueError, match='mapping changed'):
        load(bundle_copy)


def test_head_from_different_fold_rejected_even_if_file_hash_updated(bundle_copy):
    shutil.copyfile(BASE / 'research_bundles/fold_4/G2_FULL.joblib', bundle_copy / 'G2_FULL.joblib')
    rewrite_manifest(bundle_copy, lambda m: m['files_sha256'].update({'G2_FULL.joblib': digest(bundle_copy / 'G2_FULL.joblib')}))
    with pytest.raises(ValueError, match='training membership mismatch'):
        load(bundle_copy)


def test_portable_load_and_stream_do_not_read_original_data_or_artifacts(bundle_copy, monkeypatch):
    original_open, original_io_open = builtins.open, io.open
    data_root, artifact_root = (ROOT / 'data').resolve(), (ROOT / 'artifacts').resolve()
    allowed_cache = (BASE / 'numba_cache').resolve()
    def checked_opener(original):
        def call(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                path = Path(file).resolve()
                allowed = path.is_relative_to(bundle_copy.resolve()) or path.is_relative_to(allowed_cache)
                if not allowed and (path.is_relative_to(data_root) or path.is_relative_to(artifact_root)):
                    pytest.fail('Portable inference tried to access original data/artifact: ' + str(path))
            return original(file, *args, **kwargs)
        return call
    monkeypatch.setattr(builtins, 'open', checked_opener(original_open))
    monkeypatch.setattr(io, 'open', checked_opener(original_io_open))
    bundle = load(bundle_copy)
    assert bundle.prepare('G2_FULL', resident=False).qualified
    assert not bundle.prepare('I4_EVENT_SCALE_TREE', resident=False).qualified
    grid = np.arange(2048, dtype=np.float64)
    h = (np.sin(grid * .37) + .2 * np.cos(grid * .11)).astype(np.float32)
    online = h[:128].copy()
    state = bundle.new_state(h, 'G2_FULL', resident=False)
    predictions = state.replay(online)
    assert np.isfinite(predictions).all() and np.all((predictions >= 0) & (predictions <= 1))
    assert bundle.manifest['default_method'] == 'M0'

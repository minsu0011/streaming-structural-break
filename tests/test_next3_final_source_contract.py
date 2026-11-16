"""Final evidence must not survive source additions, removals or hidden config edits."""
from pathlib import Path
import pytest
from src.next3.final_source_contract import FIXED_ROOT_FILES, assert_source_snapshot, snapshot_sources


def project(tmp_path):
    for folder in ('src', 'scripts', 'tests', 'configs'):
        (tmp_path/folder).mkdir()
    for name in FIXED_ROOT_FILES:
        (tmp_path/name).write_bytes(b'fixed\n')
    (tmp_path/'src/model.py').write_bytes(b'VALUE = 1\n')
    (tmp_path/'scripts/train.py').write_bytes(b'pass\n')
    (tmp_path/'tests/test_model.py').write_bytes(b'pass\n')
    (tmp_path/'configs/model.json').write_bytes(b'{"seed":0}\n')
    (tmp_path/'main.py').write_bytes(b'pass\n')
    (tmp_path/'SEAL_PARTITION_500x4_LOCK.json').write_bytes(b'{}\n')
    return tmp_path


def test_complete_source_and_contract_snapshot(tmp_path):
    root = project(tmp_path); snapshot = snapshot_sources(root)
    assert snapshot['file_count'] == len(FIXED_ROOT_FILES)+6
    assert 'configs/model.json' in snapshot['files']
    assert 'SEAL_PARTITION_500x4_LOCK.json' in snapshot['files']
    assert assert_source_snapshot(root, snapshot) == snapshot


@pytest.mark.parametrize('name', ['src/model.py', 'configs/model.json', 'requirements-lock.txt',
    'main.py', 'SEAL_PARTITION_500x4_LOCK.json', 'REDUCED_TEST_USAGE_LEDGER.json'])
def test_existing_source_or_guard_change_is_rejected(tmp_path, name):
    root = project(tmp_path); snapshot = snapshot_sources(root)
    (root/name).write_bytes(b'changed\n')
    with pytest.raises(RuntimeError, match='changed'): assert_source_snapshot(root, snapshot)


def test_new_unlisted_source_is_rejected(tmp_path):
    root = project(tmp_path); snapshot = snapshot_sources(root)
    (root/'src/new_model.py').write_bytes(b'pass\n')
    with pytest.raises(RuntimeError, match='added='): assert_source_snapshot(root, snapshot)


def test_removed_source_is_rejected(tmp_path):
    root = project(tmp_path); snapshot = snapshot_sources(root)
    (root/'src/model.py').unlink()
    with pytest.raises(RuntimeError, match='removed='): assert_source_snapshot(root, snapshot)


def test_runtime_reports_and_bytecode_do_not_change_source_set(tmp_path):
    root = project(tmp_path); snapshot = snapshot_sources(root)
    (root/'MAINTENANCE_LOG.json').write_text('{}')
    (root/'src/__pycache__').mkdir(); (root/'src/__pycache__/temporary.py').write_text('pass')
    assert assert_source_snapshot(root, snapshot) == snapshot


def test_missing_required_dependency_file_is_rejected(tmp_path):
    root = project(tmp_path); (root/'requirements-lock.txt').unlink()
    with pytest.raises(ValueError, match='dependency'): snapshot_sources(root)


def test_new_local_instructions_are_part_of_source_set(tmp_path):
    root = project(tmp_path); snapshot = snapshot_sources(root)
    (root/'src/AGENTS.md').write_text('Additional instructions')
    with pytest.raises(RuntimeError, match='added='): assert_source_snapshot(root, snapshot)


def test_snapshot_digest_cannot_disagree_with_same_files(tmp_path):
    root = project(tmp_path); snapshot = snapshot_sources(root)
    snapshot['content_set_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='digest'): assert_source_snapshot(root, snapshot)


def test_top_level_junction_is_rejected(tmp_path, monkeypatch):
    root = project(tmp_path); original = Path.is_junction
    monkeypatch.setattr(Path, 'is_junction', lambda path: path == root/'configs' or original(path))
    with pytest.raises(ValueError, match='top-level'): snapshot_sources(root)


def test_nested_junction_is_rejected_even_when_not_walked(tmp_path, monkeypatch):
    root = project(tmp_path); nested = root/'src/nested'; nested.mkdir(); original = Path.is_junction
    monkeypatch.setattr(Path, 'is_junction', lambda path: path == nested or original(path))
    with pytest.raises(ValueError, match='Linked directory'): snapshot_sources(root)


def test_linked_file_is_rejected(tmp_path, monkeypatch):
    root = project(tmp_path); target = root/'src/model.py'; original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda path: path == target or original(path))
    with pytest.raises(ValueError, match='ordinary'): snapshot_sources(root)

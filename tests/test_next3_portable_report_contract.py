"""Tampered extracted inputs and output collisions must fail before inference."""
import hashlib
import json
from pathlib import Path
import pytest
from src.next3.portable_report_contract import new_output_path, ordinary_payload_file, verify_extracted_manifest


def payload(tmp_path, *, record_override=None, name='src/model.py'):
    content = b'print(1)\n'; target = tmp_path/name
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
    record = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    if record_override: record.update(record_override)
    manifest = {'files': {name: record}}
    (tmp_path/'SOURCE_MANIFEST_NEXT3.json').write_text(json.dumps(manifest), encoding='utf-8')
    return target, manifest


def rewrite(tmp_path, manifest):
    (tmp_path/'SOURCE_MANIFEST_NEXT3.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_ordinary_verified_copy_allows_reviewer_environment_extra(tmp_path):
    payload(tmp_path); (tmp_path/'extra.txt').write_text('reviewer notes')
    result = verify_extracted_manifest(tmp_path)
    assert result['file_count'] == 2 and 'extra.txt' not in result['files']
    assert result['uncompressed_bytes'] == sum(row['bytes'] for row in result['files'].values())


def test_same_length_modified_input_is_rejected(tmp_path):
    target, _ = payload(tmp_path); target.write_bytes(b'print(2)\n')
    with pytest.raises(RuntimeError, match='differs'): verify_extracted_manifest(tmp_path)


def test_missing_member_is_rejected(tmp_path):
    target, _ = payload(tmp_path); target.unlink()
    with pytest.raises(ValueError, match='missing'): verify_extracted_manifest(tmp_path)


@pytest.mark.parametrize('record', [{'bytes': True}, {'bytes': -1}, {'bytes': '9'},
    {'sha256': 'A'*64}, {'sha256': '0'*63}, {'sha256': '../outside'}])
def test_invalid_manifest_record_is_rejected(tmp_path, record):
    payload(tmp_path, record_override=record)
    with pytest.raises(ValueError, match='checksum or size'): verify_extracted_manifest(tmp_path)


def test_declared_length_mismatch_is_rejected(tmp_path):
    payload(tmp_path, record_override={'bytes': 8})
    with pytest.raises(RuntimeError, match='differs'): verify_extracted_manifest(tmp_path)


@pytest.mark.parametrize('name', ['../outside.txt', '/absolute.txt', 'C:/outside.txt', 'src/../model.py',
    'src/model.py:stream', 'src/CON.txt', 'src/extra.'])
def test_unsafe_member_name_is_rejected(tmp_path, name):
    with pytest.raises(ValueError): ordinary_payload_file(tmp_path, name)


def test_case_equivalent_alias_is_rejected_on_windows(tmp_path):
    target, manifest = payload(tmp_path)
    manifest['files']['SRC/MODEL.PY'] = manifest['files']['src/model.py']
    # If the host is case-sensitive, create the equivalent spelling as well.
    other = tmp_path/'SRC/MODEL.PY'
    if not other.exists(): other.parent.mkdir(); other.write_bytes(target.read_bytes())
    rewrite(tmp_path, manifest)
    with pytest.raises(ValueError, match='equivalent'): verify_extracted_manifest(tmp_path)


def test_recursive_manifest_entry_is_rejected(tmp_path):
    _, manifest = payload(tmp_path)
    manifest['files']['SOURCE_MANIFEST_NEXT3.json'] = {'sha256': '0'*64, 'bytes': 0}
    rewrite(tmp_path, manifest)
    with pytest.raises(ValueError, match='nonrecursive'): verify_extracted_manifest(tmp_path)


def test_empty_manifest_is_rejected(tmp_path):
    rewrite(tmp_path, {'files': {}})
    with pytest.raises(ValueError, match='Nonempty'): verify_extracted_manifest(tmp_path)


def test_payload_bound_cannot_be_bypassed_with_declared_size(tmp_path):
    payload(tmp_path, record_override={'bytes': 256*1024*1024})
    with pytest.raises(ValueError, match='bounded'): verify_extracted_manifest(tmp_path)


def test_linked_manifest_member_is_rejected_without_link_privilege(tmp_path, monkeypatch):
    target, _ = payload(tmp_path); original = Path.is_junction
    monkeypatch.setattr(Path, 'is_junction', lambda path: path == target or original(path))
    with pytest.raises(ValueError, match='ordinary'): verify_extracted_manifest(tmp_path)


def test_new_json_output_is_resolved_without_writing(tmp_path):
    path = new_output_path(tmp_path, 'review.json', {})
    assert path == tmp_path/'review.json' and not path.exists()


def test_existing_output_is_preserved(tmp_path):
    path = tmp_path/'review.json'; path.write_bytes(b'prior result')
    with pytest.raises(FileExistsError): new_output_path(tmp_path, path, {})
    assert path.read_bytes() == b'prior result'


def test_missing_manifest_input_cannot_be_replaced_by_output(tmp_path):
    with pytest.raises(ValueError, match='verified input'):
        new_output_path(tmp_path, 'model.json', {'model.json': {'sha256': '0'*64, 'bytes': 0}})


def test_output_requires_existing_parent(tmp_path):
    with pytest.raises(ValueError, match='parent'): new_output_path(tmp_path, 'missing/review.json', {})


def test_output_rejects_non_json(tmp_path):
    with pytest.raises(ValueError, match='JSON'): new_output_path(tmp_path, 'script.py', {})


def test_output_rejects_junction_ancestor(tmp_path, monkeypatch):
    parent = tmp_path/'linked'; parent.mkdir(); original = Path.is_junction
    monkeypatch.setattr(Path, 'is_junction', lambda path: path == parent or original(path))
    with pytest.raises(ValueError, match='junction'): new_output_path(tmp_path, parent/'review.json', {})

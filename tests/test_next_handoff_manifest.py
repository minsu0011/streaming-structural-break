import hashlib
import json
import zipfile
import pytest
from scripts.verify_next_handoff_manifest import verify_archive, MANIFEST


def make_archive(path, *, corrupted=False, extra=None):
    payload = {'artifacts/next/deployment/final/PRIMARY/model.joblib': b'opaque test model',
               'report.txt': b'fixed report'}
    files = {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
             for name, value in payload.items()}
    manifest = {'status': 'FROZEN_PACKAGE_SOURCE_MANIFEST', 'files': files,
        'files_excluding_this_manifest': len(files),
        'uncompressed_bytes_excluding_this_manifest': sum(r['bytes'] for r in files.values()),
        'selected_model_files_sha256': {name: files[name]['sha256'] for name in files if name.endswith('.joblib')}}
    if corrupted:
        payload['report.txt'] = b'other report'  # Same size, different content.
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payload.items():
            archive.writestr(name, value)
        archive.writestr(MANIFEST, json.dumps(manifest))
        if extra is not None:
            archive.writestr(extra, b'unexpected')


def test_manifest_checks_archive_without_extracting(tmp_path):
    path = tmp_path/'handoff.zip'
    make_archive(path)
    result = verify_archive(path)
    assert result['status'] == 'PASS' and result['entries'] == 3
    assert result['archive_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert list(tmp_path.iterdir()) == [path]


def test_manifest_rejects_valid_crc_with_wrong_content(tmp_path):
    path = tmp_path/'changed.zip'
    make_archive(path, corrupted=True)
    with pytest.raises(ValueError, match='SHA256'):
        verify_archive(path)


@pytest.mark.parametrize('extra', ['unlisted.txt', '../outside.txt', 'C:/outside.txt', 'REPORT.TXT'])
def test_manifest_rejects_unlisted_unsafe_or_case_colliding_members(tmp_path, extra):
    path = tmp_path/'unsafe.zip'
    make_archive(path, extra=extra)
    with pytest.raises(ValueError):
        verify_archive(path)

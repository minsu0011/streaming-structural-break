import hashlib
import json
import stat
import zipfile
import pytest
from src.next3.archive_contract import (MANIFEST_NAME, member_name, validate_members,
    verify_archive, extract_verified_archive)


@pytest.mark.parametrize('name', ['../escape.txt', '/absolute.txt', 'C:/drive.txt', 'a\\b.txt',
    'a//b.txt', './relative.txt', '.', 'tail.', 'tail ', 'NUL.txt', 'sub/COM1.log', 'data:stream', 'bad\tname'])
def test_invalid_portable_member_names_are_rejected(name):
    with pytest.raises(ValueError): member_name(name)


def test_case_duplicate_and_symlink_entries_are_rejected():
    for infos in ([zipfile.ZipInfo('File.py'), zipfile.ZipInfo('file.py')],
                  [zipfile.ZipInfo('same'), zipfile.ZipInfo('same')]):
        with pytest.raises(ValueError): validate_members(infos)
    link = zipfile.ZipInfo('link'); link.create_system = 3; link.external_attr = (stat.S_IFLNK|0o777) << 16
    with pytest.raises(ValueError): validate_members([link])


def test_size_and_encryption_bounds_are_checked_before_reading():
    info = zipfile.ZipInfo('file'); info.file_size = 101
    with pytest.raises(ValueError): validate_members([info], maximum_bytes=100)
    info.file_size = 0; info.flag_bits = 1
    with pytest.raises(ValueError): validate_members([info])


def make_archive(path, *, corrupt_hash=False, extra=False):
    content = b'ordinary source bytes\n'
    expected = hashlib.sha256(content).hexdigest()
    manifest = {'files': {'src/example.py': {'sha256': '0'*64 if corrupt_hash else expected, 'bytes': len(content)}}}
    encoded = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('src/example.py', content); archive.writestr(MANIFEST_NAME, encoded)
        if extra: archive.writestr('unexpected.txt', b'extra')
    return content, hashlib.sha256(encoded).hexdigest()


def test_actual_archive_all_bytes_are_verified_and_extracted(tmp_path):
    path = tmp_path/'source.zip'; content, manifest_sha = make_archive(path)
    report = verify_archive(path, expected_manifest_sha256=manifest_sha)
    assert report['entries'] == 2 and report['all_member_sha256_verified'] and report['crc_verified']
    destination = tmp_path/'extracted'; extracted = extract_verified_archive(path, destination, expected_manifest_sha256=manifest_sha)
    assert extracted == report and (destination/'src/example.py').read_bytes() == content
    with pytest.raises(RuntimeError): extract_verified_archive(path, destination)


@pytest.mark.parametrize('corrupt_hash,extra', [(True, False), (False, True)])
def test_tampered_hash_or_unlisted_file_fails_before_extraction(tmp_path, corrupt_hash, extra):
    path = tmp_path/'source.zip'; make_archive(path, corrupt_hash=corrupt_hash, extra=extra)
    destination = tmp_path/'must_not_exist'
    with pytest.raises(ValueError): extract_verified_archive(path, destination)
    assert not destination.exists()


def test_unexpected_manifest_hash_is_rejected(tmp_path):
    path = tmp_path/'source.zip'; make_archive(path)
    with pytest.raises(ValueError): verify_archive(path, expected_manifest_sha256='0'*64)

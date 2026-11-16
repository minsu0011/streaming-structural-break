"""Archive membership and checksum checks must reject unsafe delivery paths."""
import src.next8
import pytest
from src.next8 import delivery_payload as payload


@pytest.mark.parametrize('name', ['../outside.py', '/absolute.py', 'C:/outside.py', 'folder\\outside.py',
                                 'folder/../outside.py', 'folder//file.py', './file.py', 'folder/./file.py'])
def test_archive_paths_cannot_escape_fresh_payload(name):
    with pytest.raises(ValueError):
        payload.relative_member(name)


def test_fresh_zip_verifies_exact_files_and_content(tmp_path, monkeypatch):
    monkeypatch.setattr(payload, 'BASE', tmp_path)
    source = tmp_path / 'source.py'
    source.write_text('value = 17\n', encoding='utf-8')
    staged = tmp_path / 'delivery' / 'payload'
    hashes = payload.materialize({'src/example.py': source}, staged)
    archive = tmp_path / 'test.zip'
    payload.zip_payload(staged, archive)
    extracted = tmp_path / 'delivery' / 'fresh_extract'
    result = payload.verify_extract(archive, extracted)
    assert result['manifest_files_verified'] == 1 and result['CRC_verified']
    assert payload.digest(extracted / 'src/example.py') == hashes['src/example.py']
    with pytest.raises(AssertionError):
        payload.verify_extract(archive, extracted)

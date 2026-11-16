import src.next9
import pytest
from src.next9 import delivery_payload as payload

@pytest.mark.parametrize('name',['../escape','/absolute','C:/drive','a\\b','a//b','a/./b'])
def test_archive_paths_reject_traversal(name):
    with pytest.raises(ValueError):payload.relative_member(name)

def test_exact_zip_hash_roundtrip_and_corruption_rejection(tmp_path,monkeypatch):
    import zipfile
    monkeypatch.setattr(payload,'BASE',tmp_path)
    source=tmp_path/'source.txt';source.write_text('synthetic fixture metadata',encoding='utf-8')
    root=tmp_path/'delivery';root.mkdir()
    staged=root/'p';payload.materialize({'folder/source.txt':source},staged)
    archive=root/'fixture.zip';payload.zip_payload(staged,archive)
    result=payload.verify_extract(archive,root/'x')
    assert result['CRC_and_manifest_exact']
    assert (root/'x/folder/source.txt').read_bytes()==source.read_bytes()
    bad=root/'bad.zip'
    with zipfile.ZipFile(archive) as original,zipfile.ZipFile(bad,'x') as edited:
        for name in original.namelist():edited.writestr(name,b'changed' if name=='folder/source.txt' else original.read(name))
    with pytest.raises(AssertionError):payload.verify_extract(bad,root/'bad_x')
    assert not (root/'bad_x').exists()

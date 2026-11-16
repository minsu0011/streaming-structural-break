from pathlib import Path
import json
import stat
import zipfile
import pytest
from src.next7.handoff_archive import MANIFEST, checked_name, create_archive, extract_verified


@pytest.mark.parametrize('name', ['../escape.py','/absolute.py','C:/absolute.py','a\\escape.py',
    'a/../escape.py','data/raw.csv','a/.env','a/secret.key','old.zip',
    'artifacts/next7/EVENT_TIMING_FORENSIC_G1.csv','a/CON.txt','a/trailing.'])
def test_private_or_unsafe_entries_are_rejected(name):
    with pytest.raises(ValueError):
        checked_name(name)


def test_explicit_roundtrip_and_fresh_destination(tmp_path):
    source = tmp_path/'source'
    (source/'src').mkdir(parents=True)
    (source/'src/demo.py').write_text('print(1)\n')
    (source/'unselected.csv').write_text('not exported')
    archive = tmp_path/'handoff.zip'
    manifest = create_archive(source,['src/demo.py'],archive,{'new_role':False})
    target = tmp_path/'extracted'
    assert extract_verified(archive,target) == manifest
    assert (target/'src/demo.py').read_bytes() == (source/'src/demo.py').read_bytes()
    assert not (target/'unselected.csv').exists()
    with pytest.raises(FileExistsError):
        extract_verified(archive,target)


def test_windows_case_collision_rejected_before_archive_creation(tmp_path):
    with pytest.raises(ValueError):
        create_archive(tmp_path,['src/a.py','src/A.py'],tmp_path/'bad.zip',{})
    assert not (tmp_path/'bad.zip').exists()


def test_only_named_small_bootstrap_numpy_files_are_allowed():
    assert checked_name('artifacts/next7/synthetic_stats/COUNTS_2000x512.npy')
    assert checked_name('artifacts/next7/bootstrap/g1/M0/fold_0.npy')
    for name in ('artifacts/next7/bootstrap/g1/M0/point_predictions.npy',
                 'artifacts/next7/rp_features.npy','artifacts/next7/bootstrap/g1/M0/fold_5.npy'):
        with pytest.raises(ValueError):
            checked_name(name)


def test_reviewed_data_loader_source_is_distinct_from_raw_data():
    assert checked_name('src/data/loader.py')
    assert checked_name('src/data/__init__.py')
    for name in ('data/loader.py','src/data/raw.csv','src/data/private.npy','src/data/unknown.py'):
        with pytest.raises(ValueError):checked_name(name)


def malicious(path, names, *, bad_hash=False, symlink=False):
    entries = {name:{'sha256':'0'*64,'bytes':1} for name in names}
    manifest = {'format':'NEXT7_EXPLICIT_BYTE_MANIFEST_V1','files':entries}
    with zipfile.ZipFile(path,'x') as archive:
        for name in names:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info,b'x')
        archive.writestr(MANIFEST,json.dumps(manifest))


@pytest.mark.parametrize('kind',['traversal','hash','case','symlink','size'])
def test_malicious_zip_rejected_before_any_extraction(tmp_path,kind):
    archive = tmp_path/'bad.zip'
    names = ['../outside.py'] if kind=='traversal' else ['a.py','A.py'] if kind=='case' else ['a.py']
    malicious(archive,names,symlink=kind=='symlink')
    destination = tmp_path/'extracted'
    with pytest.raises(ValueError):
        extract_verified(archive,destination,max_entry_bytes=0 if kind=='size' else 200_000_000)
    assert not destination.exists()
    assert not (tmp_path/'outside.py').exists()

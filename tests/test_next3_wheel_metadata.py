import zipfile
import pytest
from src.next3.wheel_metadata import distribution_metadata


def write_wheel(tmp_path, entries):
    path = tmp_path/'package.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in entries.items(): archive.writestr(name, content)
    return path


def test_vendored_distribution_metadata_is_not_the_wheel_identity(tmp_path):
    path = write_wheel(tmp_path, {'py-1.11.0.dist-info/METADATA': 'Name: py\nVersion: 1.11.0\n',
        'py/_vendored_packages/apipkg-2.0.0.dist-info/METADATA': 'Name: apipkg\nVersion: 2.0.0\n',
        'py/_vendored_packages/iniconfig-1.1.1.dist-info/METADATA': 'Name: iniconfig\nVersion: 1.1.1\n'})
    record = distribution_metadata(path, 'py', '1.11.0')
    assert record['distribution'] == 'py' and record['version'] == '1.11.0'
    assert record['nested_vendored_metadata_files'] == 2


@pytest.mark.parametrize('mode', ['only_nested', 'two_top_level', 'wrong_folder', 'wrong_name', 'wrong_version', 'duplicate_name'])
def test_inconsistent_distribution_identity_is_rejected(tmp_path, mode):
    entries = {'py-1.11.0.dist-info/METADATA': 'Name: py\nVersion: 1.11.0\n'}
    if mode == 'only_nested': entries = {'nested/py-1.11.0.dist-info/METADATA': next(iter(entries.values()))}
    elif mode == 'two_top_level': entries['other-1.0.dist-info/METADATA'] = 'Name: other\nVersion: 1.0\n'
    elif mode == 'wrong_folder': entries = {'other-1.11.0.dist-info/METADATA': next(iter(entries.values()))}
    elif mode == 'wrong_name': entries['py-1.11.0.dist-info/METADATA'] = 'Name: other\nVersion: 1.11.0\n'
    elif mode == 'wrong_version': entries['py-1.11.0.dist-info/METADATA'] = 'Name: py\nVersion: 1.12.0\n'
    else: entries['py-1.11.0.dist-info/METADATA'] = 'Name: py\nName: other\nVersion: 1.11.0\n'
    with pytest.raises(ValueError): distribution_metadata(write_wheel(tmp_path, entries), 'py', '1.11.0')

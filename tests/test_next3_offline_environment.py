from pathlib import Path
import pytest
from src.next3.artifacts import atomic_json, digest
from src.next3.offline_environment import copy_verified_wheels


def cache(tmp_path):
    root = tmp_path/'origin'; base = root/'artifacts/next3'; folder = base/'maintenance/cache/wheels'
    folder.mkdir(parents=True); (root/'requirements-lock.txt').write_text('fixed test pins\n', encoding='utf-8')
    files = {}
    for index in range(63):
        path = folder/f'package{index}-1-py3-none-any.whl'; path.write_bytes(('verified fixture '+str(index)).encode())
        files[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    receipt = folder.parent/'RESULTS.json'
    atomic_json(receipt, {'status': 'PASS', 'pin_count': 63, 'pins_sha256': digest(root/'requirements-lock.txt'),
        'files': files, 'wheel_directory': str(folder)})
    pointer = base/'maintenance/PINNED_WHEEL_CACHE_SELECTION.json'
    atomic_json(pointer, {'status': 'PASS', 'pin_count': 63, 'pins_sha256': digest(root/'requirements-lock.txt'),
        'receipt': receipt.relative_to(root).as_posix(), 'receipt_sha256': digest(receipt), 'wheel_directory': str(folder)})
    return root, base, folder


def test_offline_wheels_are_ordinary_new_copies_not_reused_links(tmp_path):
    root, base, folder = cache(tmp_path); target = tmp_path/'fresh/packages'
    result = copy_verified_wheels(root, base, target)
    assert result['pin_count'] == 63 and result['environment_installed'] is False
    name = next(iter(result['files'])); (target/name).write_bytes(b'change copied bytes')
    assert (folder/name).read_bytes() != (target/name).read_bytes()


@pytest.mark.parametrize('mode', ['changed_wheel', 'extra_file', 'changed_pins', 'changed_receipt', 'existing_destination', 'inside_origin'])
def test_offline_install_inputs_reject_changed_or_nonfresh_sources(tmp_path, mode):
    root, base, folder = cache(tmp_path); target = tmp_path/'fresh/packages'
    if mode == 'changed_wheel': next(folder.glob('*.whl')).write_bytes(b'altered')
    elif mode == 'extra_file': (folder/'unexpected.whl').write_bytes(b'extra')
    elif mode == 'changed_pins': (root/'requirements-lock.txt').write_text('different pins', encoding='utf-8')
    elif mode == 'changed_receipt': (folder.parent/'RESULTS.json').write_text('{}', encoding='utf-8')
    elif mode == 'existing_destination': target.mkdir(parents=True)
    else: target = root/'copied_packages'
    with pytest.raises(RuntimeError): copy_verified_wheels(root, base, target)

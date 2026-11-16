"""Copy verified exact wheels into a fresh reproduction workspace without installing."""
from pathlib import Path
import shutil
from src.next3.artifacts import digest, read_json


def copy_verified_wheels(root, base, destination):
    root, base, destination = Path(root), Path(base), Path(destination)
    pointer_path = base/'maintenance/PINNED_WHEEL_CACHE_SELECTION.json'
    if not pointer_path.exists(): return None
    pointer = read_json(pointer_path)
    if pointer.get('status') != 'PASS' or pointer.get('pin_count') != 63:
        raise RuntimeError('A complete verified 63-pin wheel cache is required')
    receipt_path = (root/pointer['receipt']).resolve()
    if not receipt_path.is_relative_to((base/'maintenance').resolve()):
        raise RuntimeError('Wheel receipt escapes the maintenance evidence')
    if digest(receipt_path) != pointer['receipt_sha256']:
        raise RuntimeError('Wheel cache receipt changed')
    record = read_json(receipt_path)
    pins_sha = digest(root/'requirements-lock.txt')
    if (record.get('status') != 'PASS' or record.get('pin_count') != 63 or len(record.get('files', {})) != 63
            or record.get('pins_sha256') != pins_sha or pointer.get('pins_sha256') != pins_sha):
        raise RuntimeError('Wheel cache does not match the exact locked environment')
    source = Path(pointer['wheel_directory']).resolve()
    if source != Path(record['wheel_directory']).resolve() or not source.is_relative_to((base/'maintenance').resolve()):
        raise RuntimeError('Wheel directory escapes the declared maintenance cache')
    if destination.exists() or destination.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Wheel copies require a new location outside the original project')
    expected_files = record['files']
    if {p.name for p in source.iterdir()} != set(expected_files):
        raise RuntimeError('Wheel cache contains missing or extra files')
    if len({name.casefold() for name in expected_files}) != 63:
        raise RuntimeError('Wheel names collide on Windows')
    for name, item in expected_files.items():
        if Path(name).name != name or not name.endswith('.whl'):
            raise RuntimeError('Flat binary wheel filenames required')
        path = source/name
        if not path.is_file() or path.is_symlink() or path.is_junction() or path.resolve().parent != source:
            raise RuntimeError('An ordinary verified wheel file is required')
        if digest(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise RuntimeError('A cached wheel changed after official-release verification')
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name, item in expected_files.items():
        target = destination/name; shutil.copyfile(source/name, target)
        if digest(target) != item['sha256']: raise RuntimeError('Copied wheel bytes differ')
        copied[name] = {'sha256': item['sha256'], 'bytes': item['bytes']}
    return {'status': 'COPIED_VERIFIED_OFFLINE_WHEELS', 'directory': str(destination.resolve()),
        'files': copied, 'pin_count': 63, 'pins_sha256': pins_sha,
        'cache_selection_sha256': digest(pointer_path), 'cache_receipt_sha256': digest(receipt_path),
        'all_ordinary_copies': True, 'environment_installed': False}

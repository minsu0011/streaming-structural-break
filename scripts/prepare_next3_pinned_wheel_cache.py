"""Cache the existing 63 exact pins for offline final-environment installation."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def main():
    out = BASE/'maintenance/pinned_wheel_cache_v1'; out.mkdir(parents=True, exist_ok=False)
    expected = {}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8-sig').splitlines():
        name, version = line.strip().split('==')
        key = canonicalize_name(name)
        assert key not in expected; expected[key] = version
    assert len(expected) == 63 and shutil.disk_usage(ROOT).free > 3_000_000_000
    wheels = out/'wheels'; wheels.mkdir()
    command = [sys.executable, '-m', 'pip', '--isolated', '--disable-pip-version-check',
        'download', '--index-url', 'https://pypi.org/simple', '--only-binary=:all:', '--no-deps', '--no-cache-dir',
        '--requirement', str(ROOT/'requirements-lock.txt'), '--dest', str(wheels)]
    atomic_json(out/'POLICY.json', {'status': 'REGISTERED_MAINTENANCE', 'created_utc': utc_now(),
        'purpose': 'Keep final fresh-environment installation independent of a later network interruption.',
        'required_pins_sha256': digest(ROOT/'requirements-lock.txt'), 'expected_versions': expected,
        'command': command, 'source_sha256': digest(Path(__file__)),
        'environment_modified': False, 'raw_data_copied': False, 'models_fitted': False,
        'policy': 'Official public PyPI binary wheels only; verify every file against release JSON SHA256. No version substitution, source build, install or SDK execution.'}, immutable=True)
    progress('Pinned wheel cache preparation', 'Downloading the existing exact 63 pins only; no installation or raw-data reproduction')
    with (out/'ENVIRONMENT_SETUP.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        while child.poll() is None:
            time.sleep(3)
            progress('Pinned wheel cache preparation', 'Owned pip download child active; exact versions and official index fixed')
    if child.returncode:
        atomic_json(out/'FAILURE.json', {'status': 'FAIL', 'created_utc': utc_now(), 'returncode': child.returncode,
            'log_sha256': digest(out/'ENVIRONMENT_SETUP.log'), 'partial_downloads_preserved': True}, immutable=True)
        raise RuntimeError('Pinned wheel download failed; preserve partial cache and use an explicit later review')
    supported = set(sys_tags()); files = {}; observed = {}
    for path in sorted(wheels.iterdir()):
        assert path.is_file() and path.suffix == '.whl'
        name, version, build, tags = parse_wheel_filename(path.name); key = canonicalize_name(name)
        assert key in expected and str(version) == expected[key] and key not in observed
        assert tags.intersection(supported), 'Wheel incompatible with the current pinned interpreter'
        url = 'https://pypi.org/pypi/'+key+'/'+expected[key]+'/json'
        request = urllib.request.Request(url, headers={'User-Agent': 'StructuralBreak-NEXT3-pinned-wheel-maintenance'})
        with urllib.request.urlopen(request, timeout=25) as response:
            release = json.loads(response.read(4_000_001))
        matches = [item for item in release['urls'] if item['filename'] == path.name]
        assert len(matches) == 1 and not matches[0]['yanked']
        item = matches[0]; sha = digest(path)
        assert sha == item['digests']['sha256'] and path.stat().st_size == item['size']
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            metadata = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            assert len(metadata) == 1
            # Merely read distribution metadata; no wheel contents are imported.
            content = archive.read(metadata[0]).decode('utf-8')
            lines = content.splitlines()
            declared_name = next(line[6:] for line in lines if line.startswith('Name: '))
            declared_version = next(line[9:] for line in lines if line.startswith('Version: '))
            assert canonicalize_name(declared_name) == key and declared_version == expected[key]
        observed[key] = str(version)
        files[path.name] = {'sha256': sha, 'bytes': path.stat().st_size, 'distribution': key,
            'version': str(version), 'release_metadata_url': url, 'official_file_url': item['url'],
            'compatible_interpreter_tag': True, 'zip_CRC_pass': True,
            'metadata_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest()}
        progress('Pinned wheel integrity verification', f'{len(files)}/63 exact distribution/version, official SHA256, CRC and interpreter tags verified')
    assert observed == expected
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'wheel_directory': str(wheels.resolve()),
        'python': sys.version, 'pin_count': len(expected), 'pins_sha256': digest(ROOT/'requirements-lock.txt'),
        'versions': observed, 'files': files, 'total_wheel_bytes': sum(value['bytes'] for value in files.values()),
        'environment_modified': False, 'SDK_executed': False, 'raw_data_read': False, 'new_fits': 0,
        'seal_values_accessed': 0, 'source_sha256': digest(Path(__file__)),
        'package_scope': 'Wheel binaries stay in the local maintenance cache and are excluded from GPT handoff. The hash manifest is shareable metadata.',
        'scope': 'Official release integrity and current Windows/Python wheel compatibility. Installing and pip-checking a fresh environment remain final-audit requirements.'}, immutable=True)
    print('NEXT3 PINNED WHEEL CACHE', len(files), 'OFFICIAL SHA256 AND CRC PASS', flush=True)


if __name__ == '__main__': main()

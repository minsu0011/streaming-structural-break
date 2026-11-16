"""Re-verify preserved wheels after the explicit vendored-metadata correction."""
from pathlib import Path
import json
import sys
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.wheel_metadata import distribution_metadata


def main():
    source = BASE/'maintenance/pinned_wheel_cache_v1'; policy = read_json(source/'POLICY.json')
    expected = policy['expected_versions']; assert len(expected) == 63
    assert digest(ROOT/'requirements-lock.txt') == policy['required_pins_sha256']
    wheels = source/'wheels'; out = source/'verification_v2'; out.mkdir(parents=True, exist_ok=False)
    failed = BASE/'jobs/M624_PINNED_WHEEL_CACHE/attempt_1/RESULT.json'
    assert read_json(failed)['status'] == 'FAILED'
    atomic_json(out/'REPAIR.json', {'status': 'REGISTERED_VERIFICATION_CORRECTION', 'created_utc': utc_now(),
        'original_policy_sha256': digest(source/'POLICY.json'), 'original_failed_attempt_sha256': digest(failed),
        'original_executed_script_sha256': digest(ROOT/'scripts/prepare_next3_pinned_wheel_cache.py'),
        'actual_failure': 'py-1.11.0 wheel legitimately contains two nested vendored METADATA files. V1 incorrectly required only one METADATA anywhere in the archive.',
        'correction': 'Require exactly one top-level dist-info/METADATA; verify folder, distribution Name and exact Version, while counting vendored metadata separately.',
        'corrected_metadata_source_sha256': digest(ROOT/'src/next3/wheel_metadata.py'),
        'corrected_driver_sha256': digest(Path(__file__)), 'original_wheels_and_failure_preserved': True,
        'new_download_or_install': False}, immutable=True)
    paths = sorted(wheels.iterdir()); assert len(paths) == 63
    supported = set(sys_tags()); files = {}; observed = {}
    for path in paths:
        assert path.is_file() and path.suffix == '.whl'
        name, version, build, tags = parse_wheel_filename(path.name); key = canonicalize_name(name)
        assert key in expected and str(version) == expected[key] and key not in observed and tags.intersection(supported)
        url = 'https://pypi.org/pypi/'+key+'/'+expected[key]+'/json'
        request = urllib.request.Request(url, headers={'User-Agent': 'StructuralBreak-NEXT3-pinned-wheel-maintenance'})
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = response.read(4_000_001)
        assert len(payload) <= 4_000_000
        release = json.loads(payload); matches = [item for item in release['urls'] if item['filename'] == path.name]
        assert len(matches) == 1 and not matches[0]['yanked']
        item = matches[0]; sha = digest(path)
        assert sha == item['digests']['sha256'] and path.stat().st_size == item['size']
        metadata = distribution_metadata(path, key, expected[key]); observed[key] = str(version)
        files[path.name] = {'sha256': sha, 'bytes': path.stat().st_size, 'distribution': key,
            'version': str(version), 'release_metadata_url': url, 'official_file_url': item['url'],
            'compatible_interpreter_tag': True, 'zip_CRC_pass': True, 'distribution_metadata': metadata}
        progress('Corrected pinned wheel integrity verification', f'{len(files)}/63 exact pins, official SHA256, CRC and top-level distribution identities verified')
    assert observed == expected
    result = {'status': 'PASS', 'created_utc': utc_now(), 'wheel_directory': str(wheels.resolve()),
        'python': sys.version, 'pin_count': len(expected), 'pins_sha256': digest(ROOT/'requirements-lock.txt'),
        'versions': observed, 'files': files, 'total_wheel_bytes': sum(value['bytes'] for value in files.values()),
        'environment_modified': False, 'SDK_executed': False, 'raw_data_read': False, 'new_fits': 0,
        'seal_values_accessed': 0, 'new_package_downloads': 0, 'repair_sha256': digest(out/'REPAIR.json'),
        'source_sha256': digest(Path(__file__)), 'metadata_helper_sha256': digest(ROOT/'src/next3/wheel_metadata.py'),
        'package_scope': 'Wheel binaries remain in the excluded local maintenance cache; only this hash manifest belongs in research report.',
        'scope': 'Official release integrity and Windows/Python compatibility only. Fresh environment installation and pip check remain final-audit requirements.'}
    atomic_json(out/'RESULTS.json', result, immutable=True)
    atomic_json(BASE/'maintenance/PINNED_WHEEL_CACHE_SELECTION.json', {'status': 'PASS', 'created_utc': utc_now(),
        'receipt': (out/'RESULTS.json').relative_to(ROOT).as_posix(), 'receipt_sha256': digest(out/'RESULTS.json'),
        'wheel_directory': str(wheels.resolve()), 'pin_count': 63, 'pins_sha256': result['pins_sha256'],
        'original_failed_attempt_preserved': True}, immutable=True)
    print('NEXT3 CORRECTED PINNED WHEEL CACHE 63 OFFICIAL SHA256/CRC/METADATA PASS', flush=True)


if __name__ == '__main__': main()

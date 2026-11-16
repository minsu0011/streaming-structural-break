"""Verify a NEXT handoff ZIP without raw data, model loading or dependencies.

Usage: python scripts/verify_next_handoff_manifest.py path/to/handoff.zip
Nothing is extracted or written. The manifest proves internal integrity, not
authenticity; compare the full archive SHA256 with the separately supplied receipt.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import stat
import zipfile


MANIFEST = 'SOURCE_MANIFEST_NEXT.json'
MAX_BYTES = 512*1024*1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: '+key)
        result[key] = value
    return result


def digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        digest.update(block)
    return digest.hexdigest()


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError('Duplicate or case-colliding archive members')
        for item in entries:
            name = item.filename
            parts = name.split('/')
            if (not name or '\\' in name or ':' in name or
                    PurePosixPath(name).is_absolute() or any(part in ('', '.', '..') for part in parts)):
                raise ValueError('Unsafe archive member: '+name)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Symlink archive member: '+name)
        if MANIFEST not in names or archive.getinfo(MANIFEST).file_size > 16*1024*1024:
            raise ValueError('Missing or unexpectedly large NEXT manifest')
        manifest = json.loads(archive.read(MANIFEST), object_pairs_hook=unique_object)
        if manifest.get('status') != 'FROZEN_PACKAGE_SOURCE_MANIFEST':
            raise ValueError('Unexpected NEXT manifest status')
        files = manifest['files']
        if set(names) != set(files) | {MANIFEST} or MANIFEST in files:
            raise ValueError('Archive member set differs from manifest')
        total = 0
        for name, record in files.items():
            size, expected = record['bytes'], record['sha256']
            if type(size) is not int or size < 0 or not re.fullmatch('[0-9a-f]{64}', expected):
                raise ValueError('Malformed member identity: '+name)
            total += size
            if total > MAX_BYTES or archive.getinfo(name).file_size != size:
                raise ValueError('Archive member size differs: '+name)
            with archive.open(name) as stream:
                if digest_stream(stream) != expected:
                    raise ValueError('Archive member SHA256 differs: '+name)
        if len(files) != manifest['files_excluding_this_manifest'] or total != manifest['uncompressed_bytes_excluding_this_manifest']:
            raise ValueError('Manifest counts differ')
        models = manifest['selected_model_files_sha256']
        binaries = {name for name in files if name.endswith('.joblib')}
        if not 1 <= len(binaries) <= 3 or binaries != {name for name in models if name.endswith('.joblib')}:
            raise ValueError('Final model count or membership differs')
        for name, expected in models.items():
            if name not in files or files[name]['sha256'] != expected:
                raise ValueError('Selected model identity differs: '+name)
    with Path(path).open('rb') as stream:
        archive_digest = digest_stream(stream)
    return {'status': 'PASS', 'archive_sha256': archive_digest, 'entries': len(names),
        'payload_bytes': total, 'selected_models': len(binaries),
        'all_member_sizes_sha256_and_crc_verified': True,
        'scope': 'Internal integrity only; compare archive SHA256 with the external receipt. No extraction, model loading or raw data access.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive), indent=2))


if __name__ == '__main__':
    main()

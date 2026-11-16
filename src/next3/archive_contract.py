"""Bounded byte verification and ordinary extraction for the actual NEXT3 ZIP."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import stat
import unicodedata
import zipfile

MAX_UNCOMPRESSED_BYTES = 256*1024*1024
MANIFEST_NAME = 'SOURCE_MANIFEST_NEXT3.json'
WINDOWS_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))}


def member_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '//' in name:
        raise ValueError('Unsafe archive member spelling')
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or name != path.as_posix() or any(part in ('.', '..') for part in path.parts):
        raise ValueError('Relative canonical archive member required')
    for part in path.parts:
        if (part[-1:] in ('.', ' ') or any(ord(c) < 32 for c in part) or
                any(c in '<>"|?*' for c in part) or part.split('.')[0].upper() in WINDOWS_RESERVED):
            raise ValueError('Archive name is not an ordinary Windows filename')
    return path


def validate_members(infos, *, maximum_bytes=MAX_UNCOMPRESSED_BYTES):
    seen = set(); total = 0
    for info in infos:
        member_name(info.filename)
        normalized = unicodedata.normalize('NFC', info.filename).casefold()
        if normalized in seen: raise ValueError('Duplicate or case-equivalent archive members')
        seen.add(normalized)
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
            raise ValueError('Archive contains a non-ordinary file entry')
        if info.flag_bits & 1: raise ValueError('Encrypted archive members are unsupported')
        if info.file_size < 0: raise ValueError('Invalid uncompressed member size')
        total += info.file_size
        if total > maximum_bytes: raise ValueError('Archive exceeds bounded uncompressed payload')
    return total


def stream_sha256(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def verify_archive(path, *, expected_manifest_sha256=None):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist(); total = validate_members(infos)
        names = [info.filename for info in infos]
        if MANIFEST_NAME not in names: raise ValueError('NEXT3 source manifest missing')
        content = archive.read(MANIFEST_NAME); manifest_sha = hashlib.sha256(content).hexdigest()
        if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
            raise ValueError('Archive source manifest checksum differs')
        manifest = json.loads(content); files = manifest['files']
        if not isinstance(files, dict) or MANIFEST_NAME in files:
            raise ValueError('Manifest must list all other files without a recursive self hash')
        if set(names) != set(files)|{MANIFEST_NAME}: raise ValueError('Archive contents differ from exact source manifest')
        expected = {MANIFEST_NAME: {'sha256': manifest_sha, 'bytes': len(content)}}
        for name, record in files.items():
            member_name(name)
            if (not isinstance(record, dict) or not re.fullmatch('[0-9a-f]{64}', record.get('sha256', '')) or
                    type(record.get('bytes')) is not int or record['bytes'] < 0):
                raise ValueError('Invalid manifest file record')
            expected[name] = record
        for info in infos:
            record = expected[info.filename]
            if info.file_size != record['bytes']: raise ValueError('Manifest member size differs: '+info.filename)
            # A complete ZipExtFile read verifies its CRC as well as our SHA256.
            with archive.open(info) as stream: actual = stream_sha256(stream)
            if actual != record['sha256']: raise ValueError('Manifest member checksum differs: '+info.filename)
    with path.open('rb') as stream: archive_sha = stream_sha256(stream)
    return {'archive_sha256': archive_sha, 'manifest_sha256': manifest_sha, 'files': expected,
        'entries': len(expected), 'uncompressed_bytes': total, 'all_member_sha256_verified': True, 'crc_verified': True}


def extract_verified_archive(path, destination, *, expected_manifest_sha256=None):
    proof = verify_archive(path, expected_manifest_sha256=expected_manifest_sha256)
    destination = Path(destination).resolve()
    if destination.exists(): raise RuntimeError('Preserve any existing archive replay directory')
    destination.mkdir()
    with zipfile.ZipFile(path) as archive:
        for name, record in proof['files'].items():
            target = destination.joinpath(*member_name(name).parts).resolve()
            if not target.is_relative_to(destination): raise ValueError('Extraction escaped the new destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open('xb') as output:
                result = hashlib.sha256(); count = 0
                for block in iter(lambda: source.read(1024*1024), b''):
                    output.write(block); result.update(block); count += len(block)
            if result.hexdigest() != record['sha256'] or count != record['bytes']:
                raise ValueError('Extracted archive member differs: '+name)
            if target.is_symlink() or (hasattr(target, 'is_junction') and target.is_junction()):
                raise ValueError('Extracted member is not an ordinary file')
    return proof

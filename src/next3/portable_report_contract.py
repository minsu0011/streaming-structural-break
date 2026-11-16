"""Read-only manifest and output contracts for a reviewer-owned extracted copy."""
from pathlib import Path
import json
import re
import unicodedata
from src.next3.archive_contract import MAX_UNCOMPRESSED_BYTES, MANIFEST_NAME, member_name
from src.next3.artifacts import digest


def ordinary_payload_file(root, name):
    root = Path(root).resolve(); relative = member_name(name)
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Manifest member is missing or escapes the extracted copy: '+name)
    current = path
    while current != root:
        if current.is_symlink() or current.is_junction():
            raise ValueError('Manifest member must use ordinary paths: '+name)
        current = current.parent
    return path


def verify_extracted_manifest(root):
    root = Path(root).resolve(); path = ordinary_payload_file(root, MANIFEST_NAME)
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or MANIFEST_NAME in files:
        raise ValueError('Nonempty nonrecursive source manifest required')
    seen = set(); total = path.stat().st_size; verified = {}
    for name, record in files.items():
        source = ordinary_payload_file(root, name)
        normalized = unicodedata.normalize('NFC', name).casefold()
        if normalized in seen: raise ValueError('Case or Unicode equivalent manifest member')
        seen.add(normalized)
        if (not isinstance(record, dict) or not re.fullmatch('[0-9a-f]{64}', record.get('sha256', '')) or
                type(record.get('bytes')) is not int or record['bytes'] < 0):
            raise ValueError('Invalid extracted manifest checksum or size')
        total += record['bytes']
        if total > MAX_UNCOMPRESSED_BYTES: raise ValueError('Extracted manifest exceeds the bounded payload')
        if source.stat().st_size != record['bytes'] or digest(source) != record['sha256']:
            raise RuntimeError('Extracted member differs from manifest: '+name)
        verified[name] = {'sha256': record['sha256'], 'bytes': record['bytes']}
    verified[MANIFEST_NAME] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return {'files': verified, 'manifest_sha256': verified[MANIFEST_NAME]['sha256'],
        'file_count': len(verified), 'uncompressed_bytes': total,
        'scope': 'All listed extracted members verified; reviewer environment/output extras are permitted. This does not independently verify original ZIP bytes.'}


def new_output_path(root, value, verified_members):
    root = Path(root).resolve(); path = Path(value)
    if not path.is_absolute(): path = root/path
    # resolve() would hide an existing symlink; inspect the lexical path first.
    for current in (path, *path.parents):
        if current.is_symlink() or current.is_junction():
            raise ValueError('Replay output cannot traverse a symlink or junction')
    path = path.resolve()
    if path.exists(): raise FileExistsError('Preserve an earlier review output: '+str(path))
    if not path.parent.is_dir(): raise ValueError('Review output parent must already exist')
    if path.is_relative_to(root) and path.relative_to(root).as_posix() in verified_members:
        raise ValueError('Replay output would overwrite a verified input')
    if path.suffix.lower() != '.json': raise ValueError('A new JSON review result path is required')
    return path

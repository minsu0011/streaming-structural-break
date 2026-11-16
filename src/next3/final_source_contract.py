"""Bind final execution to the complete current code, config and dependency set."""
from pathlib import Path
import hashlib
import json
import os
from src.next3.artifacts import digest

FIXED_ROOT_FILES = ('requirements-lock.txt', 'requirements-next2-figures-lock.txt',
    'pyproject.toml', '.gitattributes', 'REDUCED_TEST_USAGE_LEDGER.json')


def source_files(root):
    root = Path(root).resolve(); selected = set()
    for folder in ('src', 'scripts', 'tests', 'configs'):
        directory = root/folder
        if not directory.is_dir(): raise ValueError('Final source directory missing: '+folder)
        if directory.is_symlink() or directory.is_junction():
            raise ValueError('Linked top-level final source directory: '+folder)
        for parent, directories, files in os.walk(directory, followlinks=False):
            parent = Path(parent)
            for name in list(directories):
                child = parent/name
                if child.is_symlink() or child.is_junction():
                    raise ValueError('Linked directory in final code/config tree: '+str(child))
                if name in ('__pycache__', '.pytest_cache'): directories.remove(name)
            for name in files:
                path = parent/name
                if path.suffix not in ('.py', '.json', '.toml') and name not in ('.gitattributes', 'AGENTS.md'):
                    continue
                selected.add(path)
    selected.update(root.glob('*.py'))
    selected.update(root.glob('*LOCK*.json'))
    for name in FIXED_ROOT_FILES:
        if not (root/name).is_file(): raise ValueError('Final dependency or contract file missing: '+name)
        selected.add(root/name)
    if (root/'AGENTS.md').exists(): selected.add(root/'AGENTS.md')
    for path in selected:
        if not path.is_file() or path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
            raise ValueError('Final source member must be an ordinary in-project file: '+str(path))
    return sorted(selected)


def snapshot_sources(root):
    root = Path(root).resolve()
    files = {path.relative_to(root).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
             for path in source_files(root)}
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return {'files': files, 'file_count': len(files), 'content_set_sha256': hashlib.sha256(encoded).hexdigest(),
        'scope': 'All project Python sources/tests/entries, JSON configs/root locks, selected build/dependency files and local instructions. Runtime report/log outputs are separate evidence.'}


def assert_source_snapshot(root, expected):
    actual = snapshot_sources(root)
    if actual['files'] != expected.get('files') or actual['file_count'] != expected.get('file_count'):
        before = expected.get('files', {})
        added = sorted(set(actual['files'])-set(before)); removed = sorted(set(before)-set(actual['files']))
        changed = sorted(name for name in set(actual['files']) & set(before) if actual['files'][name] != before[name])
        raise RuntimeError(f'Final execution source set changed: added={added}; removed={removed}; changed={changed}')
    if actual['content_set_sha256'] != expected.get('content_set_sha256'):
        raise RuntimeError('Final source set digest is inconsistent')
    return actual

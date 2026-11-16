"""Read boundary for actual portable inference, installed before ML imports."""
import os
from pathlib import Path
import sys


def denied_path(path, payload, original=None, output=None):
    path = Path(path).resolve()
    payload = Path(payload).resolve()
    if path.is_relative_to(payload / 'data'):
        return 'Competition/cache data directory is forbidden for raw-free replay'
    if original is not None and path.is_relative_to(Path(original).resolve() / 'data'):
        return 'Original competition/cache data directory is forbidden'
    if path.is_relative_to(payload):
        return None
    if output is not None and path.is_relative_to(Path(output).resolve()):
        return None
    if original is not None:
        original = Path(original).resolve()
        if path.is_relative_to(original) and not path.is_relative_to(original / '.venv'):
            return 'Original workspace cannot supply missing portable payload assets'
    return None


def install(payload, *, original=None, output=None):
    payload = Path(payload).resolve()
    if original is not None and Path(original).resolve() == payload:
        raise ValueError('A fresh extracted workspace must differ from the original')
    denied = []
    opened = set()

    def audit(event, arguments):
        if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        reason = denied_path(path, payload, original, output)
        if reason:
            denied.append({'path': str(path), 'reason': reason})
            raise PermissionError(reason + ': ' + str(path))
        if path.is_relative_to(payload):
            opened.add(path.relative_to(payload).as_posix())

    sys.addaudithook(audit)
    return denied, opened


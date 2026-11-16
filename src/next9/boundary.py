"""Application checks for writes, RP-only reads and target-free NPZ members."""
from contextlib import contextmanager
from pathlib import Path
import os
import sys
from src.next9.common import ROOT, BASE

LABEL_MEMBERS = frozenset(('target', 'tau', 'breakOccurred', 'label', 'labels', 'true_break_age'))


@contextmanager
def target_free_npz():
    import numpy as np
    cls = np.lib.npyio.NpzFile
    previous = cls.__getitem__
    observed = []

    def checked(instance, key):
        normalized = str(key).removesuffix('.npy')
        if normalized in LABEL_MEMBERS:
            raise PermissionError('Target-free phase forbids NPZ member: ' + normalized)
        observed.append(normalized)
        return previous(instance, key)

    cls.__getitem__ = checked
    try:
        yield observed
    finally:
        cls.__getitem__ = previous


def install_workspace_boundary():
    allowed_write = tuple((ROOT / p).resolve() for p in ('src/next9', 'scripts/next9', 'artifacts/next9'))
    rp = (ROOT / 'data/next5/research_5d54e316ff5bf25d').resolve()
    violations = []

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        reason = None
        if writing and path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in allowed_write):
            reason = 'Legacy workspace write forbidden'
        if not writing and path.is_relative_to(ROOT / 'data') and not path.is_relative_to(rp):
            reason = 'Only isolated RP cache may supply observation values'
        if reason:
            violations.append(dict(path=str(path), reason=reason))
            raise PermissionError(reason + ': ' + str(path))

    sys.addaudithook(audit)
    return violations

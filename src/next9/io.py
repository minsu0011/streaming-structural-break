"""Exclusive NEXT9 array writes; interrupted outputs are never overwritten."""
from pathlib import Path
import os
import uuid
import numpy as np
from src.next9.common import BASE


def save_npz(path, **arrays):
    path = Path(path).resolve()
    assert path.is_relative_to(BASE.resolve()) and not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(tmp, path)

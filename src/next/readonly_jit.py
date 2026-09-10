"""Disable disk caching for already imported project Numba dispatchers.

This changes cache IO only, not compiled math, model parameters or source hashes.
The supported Numba version exposes cache.disable(); a missing contract fails.
"""
from pathlib import Path
import hashlib
import sys
from numba.core.registry import CPUDispatcher


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def disable_loaded_disk_caches():
    seen = set()
    for name,module in tuple(sys.modules.items()):
        if module is None or not name.startswith(('src.next.','src.features.','src.models.','src.streaming.')):
            continue
        for value in tuple(vars(module).values()):
            if isinstance(value,CPUDispatcher) and id(value) not in seen:
                seen.add(id(value))
                cache = getattr(value,'_cache',None)
                if cache is None or not callable(getattr(cache,'disable',None)):
                    raise RuntimeError('The pinned Numba disk-cache disabling contract changed')
                cache.disable()
    return len(seen)

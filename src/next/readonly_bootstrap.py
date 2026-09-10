"""Prevent project Numba disk-cache initialization before project imports.

The pinned Dispatcher constructor starts with NullCache. Skipping its cache
enable request retains that in-memory default; compiled math is unchanged.
This additive entry bootstrap does not edit any frozen feature/model sources.
"""
from pathlib import Path
import hashlib
import sys
sys.dont_write_bytecode = True
import numba
from numba.core.registry import CPUDispatcher

_requests = []
_installed = False


def install():
    global _installed
    if _installed:
        return
    if numba.__version__ != '0.67.0':
        raise RuntimeError('The reviewed read-only import adapter requires pinned Numba 0.67.0')
    original = CPUDispatcher.enable_caching
    def enable_without_project_disk_cache(dispatcher):
        function = dispatcher.py_func
        if function.__module__.startswith('src.'):
            _requests.append(function.__module__+'.'+function.__qualname__)
            return None
        return original(dispatcher)
    CPUDispatcher.enable_caching = enable_without_project_disk_cache
    _installed = True


def cache_requests_suppressed():
    return tuple(_requests)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

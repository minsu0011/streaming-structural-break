"""Bound math-library threads without changing the parent environment."""
import os

THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def environment(parent=None):
    result = dict(os.environ if parent is None else parent)
    for key in THREAD_KEYS:
        result[key] = '1'
    return result

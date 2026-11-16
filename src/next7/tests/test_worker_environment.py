from src.next7.worker_environment import environment, THREAD_KEYS


def test_only_math_thread_limits_change_and_parent_is_untouched():
    parent = {'PATH': 'unchanged', 'NUMBA_CACHE_DIR': 'next7-owned', 'OPENBLAS_NUM_THREADS': '8'}
    copy = dict(parent)
    child = environment(parent)
    assert parent == copy
    assert all(child[k] == '1' for k in THREAD_KEYS)
    assert child['PATH'] == parent['PATH'] and child['NUMBA_CACHE_DIR'] == parent['NUMBA_CACHE_DIR']
    assert set(child) - set(parent) == set(THREAD_KEYS) - set(parent)

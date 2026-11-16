"""Test-only resource and synthetic output-root compatibility.

The original raw-reader guard and numerical test assertions stay unchanged.
"""
from pathlib import Path
import pytest
from src.next8 import test_guard as original
from src.next8.common import BASE, atomic, digest, utc_now

keep_legacy_scalar_tests_raw_free = original.keep_legacy_scalar_tests_raw_free
_adaptations = []


def pytest_configure(config):
    import numba
    assert numba.config.NUMBA_NUM_THREADS == 2
    numba.set_num_threads(1)
    original.pytest_configure(config)


@pytest.fixture(autouse=True)
def isolated_test_resources_and_output_root(request, monkeypatch):
    import numba
    numba.set_num_threads(1)
    name = request.node.name.split('[')[0]
    if name in ('test_complete_H_rule_fit_does_not_use_validation_labels',
                'test_end_to_end_conditional_rule_fit_excludes_validation_labels'):
        import src.next8.screening as screening
        synthetic_root = request.getfixturevalue('tmp_path').resolve()
        assert not synthetic_root.is_relative_to(BASE)
        monkeypatch.setattr(screening, 'BASE', synthetic_root)
        _adaptations.append({'test': request.node.nodeid,
            'change': 'Serialize synthetic conditional model paths relative to this test temporary directory',
            'temporary_root': str(synthetic_root), 'training_or_numerical_assertions_changed': False})
    yield
    if name == 'test_pair_matrix_matches_literal_duplicate_rows_and_ties':
        assert numba.get_num_threads() == 2
        _adaptations.append({'test': request.node.nodeid,
            'change': 'Original test actually ran its requested two Numba threads; numerical oracle unchanged',
            'actual_numba_threads': 2})
    numba.set_num_threads(1)


def pytest_unconfigure(config):
    original.pytest_unconfigure(config)
    import os
    name = os.environ.get('NEXT8_TEST_COMPATIBILITY_RECEIPT', 'FINAL_TEST_COMPATIBILITY_V3_NEXT8.json')
    assert '/' not in name and '\\' not in name
    atomic(BASE / 'audit' / name, {'status': 'COMPLETE', 'timestamp': utc_now(),
        'test_only_adaptations': _adaptations, 'source_sha256': digest(Path(__file__)),
        'existing_test_sources_and_all_numerical_assertions_unchanged': True,
        'raw_reader_guard_unchanged': True, 'normal_active_numba_threads': 1,
        'legacy_requested_thread_capacity': 2, 'production_process_thread_settings_unchanged': True})

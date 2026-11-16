import datetime
import pytest
from scripts import prepare_next3_independent_reproduction as preparation
from src.next3.artifacts import atomic_json


def session(tmp_path, monkeypatch, elapsed, audit_offset=None):
    start = datetime.datetime(2026, 9, 10, tzinfo=datetime.timezone.utc)
    record = {'started_utc': start.isoformat()}
    if audit_offset is not None:
        record['audit_started_utc'] = (start+datetime.timedelta(seconds=audit_offset)).isoformat()
    atomic_json(tmp_path/'SESSION.json', record)
    monkeypatch.setattr(preparation, 'BASE', tmp_path)
    monkeypatch.setattr(preparation.time, 'time', lambda: start.timestamp()+elapsed)


@pytest.mark.parametrize('elapsed,audit_offset', [(40199, None), (40300, None), (40300, 40199), (40300, 40400)])
def test_final_raw_preparation_rejects_early_missing_or_invalid_audit(tmp_path, monkeypatch, elapsed, audit_offset):
    session(tmp_path, monkeypatch, elapsed, audit_offset)
    with pytest.raises(RuntimeError): preparation.time_gate()


def test_final_raw_preparation_accepts_separately_started_audit_at_boundary(tmp_path, monkeypatch):
    session(tmp_path, monkeypatch, 40200, 40200)
    result = preparation.time_gate()
    assert result['wall_seconds'] == result['research_minimum_seconds'] == 40200


def selection_root(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    monkeypatch.setattr(preparation, 'BASE', tmp_path/'artifacts/next3')
    return tmp_path


def test_validation_only_lean_prototype_cannot_replace_eligible_packed_entry(tmp_path, monkeypatch):
    root = selection_root(tmp_path, monkeypatch)
    atomic_json(root/'configs/next3/LEAN_RUNTIME_SELECTION.json', {'status': 'VALIDATION_ONLY', 'variant': 'LEAN_PACKED_SHARED'})
    atomic_json(root/'configs/next3/PACKED_RUNTIME_SELECTION.json', {'status': 'PROVISIONAL_ENGINEERING_ELIGIBLE', 'variant': 'PACKED_SHARED'})
    atomic_json(root/'artifacts/next3/packed_public_api_all_dev/RESULTS.json', {'status': 'PASS', 'identity': {'variant': 'PACKED_SHARED'}})
    entry, files = preparation.selected_performance()
    assert entry == 'next3_packed_performance_main'
    assert not any('LEAN_' in path.name for path in files)


@pytest.mark.parametrize('cold_pass,direct_pass', [(False, True), (True, False)])
def test_lean_selection_requires_both_actual_cost_gates(tmp_path, monkeypatch, cold_pass, direct_pass):
    root = selection_root(tmp_path, monkeypatch)
    atomic_json(root/'configs/next3/LEAN_RUNTIME_SELECTION.json', {'status': 'PROVISIONAL_ENGINEERING_ELIGIBLE', 'variant': 'LEAN_PACKED_SHARED'})
    atomic_json(root/'artifacts/next3/lean_public_api_all_dev/RESULTS.json', {'status': 'PASS', 'identity': {'variant': 'LEAN_PACKED_SHARED'}})
    atomic_json(root/'artifacts/next3/engineering/lean_shared/cold/RESULTS.json', {'status': 'PASS', 'cold_cost_gate_pass': cold_pass})
    atomic_json(root/'artifacts/next3/engineering/lean_shared/direct/RESULTS.json', {'status': 'PASS', 'direct_cost_gate_pass': direct_pass})
    with pytest.raises(RuntimeError, match='both actual cost gates'): preparation.selected_performance()


def test_eligible_lean_entry_copies_all_required_runtime_contracts(tmp_path, monkeypatch):
    root = selection_root(tmp_path, monkeypatch)
    atomic_json(root/'configs/next3/LEAN_RUNTIME_SELECTION.json', {'status': 'PROVISIONAL_ENGINEERING_ELIGIBLE', 'variant': 'LEAN_PACKED_SHARED'})
    atomic_json(root/'artifacts/next3/lean_public_api_all_dev/RESULTS.json', {'status': 'PASS', 'identity': {'variant': 'LEAN_PACKED_SHARED'}})
    atomic_json(root/'artifacts/next3/engineering/lean_shared/cold/RESULTS.json', {'status': 'PASS', 'cold_cost_gate_pass': True})
    atomic_json(root/'artifacts/next3/engineering/lean_shared/direct/RESULTS.json', {'status': 'PASS', 'direct_cost_gate_pass': True})
    entry, files = preparation.selected_performance()
    assert entry == 'next3_lean_performance_main'
    assert {path.name for path in files} == {'LEAN_RUNTIME_SELECTION.json', 'next3_lean_performance_main.py',
        'LEAN_FEATURE_CONTRACT.json', 'LEAN_NUMERIC_BUILD.json', 'LEAN_RUNTIME_INPUTS.json'}

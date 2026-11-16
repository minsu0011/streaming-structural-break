import pytest
from scripts import export_next3_progress as exporter
from src.next3.artifacts import atomic_json, digest


def test_partial_worker_table_is_not_published_until_committed(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'ROOT', tmp_path)
    path = tmp_path/'scores.csv'; path.write_text('candidate,score\nA,0.6\n', encoding='utf-8')
    assert exporter.completed_csvs([path]) == []
    atomic_json(tmp_path/'RESULTS.json', {'status': 'PASS', 'files_sha256': {path.name: digest(path)}})
    frame, = exporter.completed_csvs([path])
    assert frame.candidate.tolist() == ['A']
    assert frame.source_receipt_sha256.iloc[0] == digest(tmp_path/'RESULTS.json')


def test_completed_worker_table_hash_mismatch_or_missing_hash_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'ROOT', tmp_path)
    path = tmp_path/'scores.csv'; path.write_text('candidate,score\nA,0.6\n', encoding='utf-8')
    receipt = tmp_path/'RESULTS.json'
    atomic_json(receipt, {'status': 'PASS', 'files_sha256': {path.name: digest(path)}})
    path.write_text('candidate,score\nA,0.9\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='exact committed receipt'): exporter.completed_csvs([path])
    atomic_json(receipt, {'status': 'PASS', 'files_sha256': {}})
    with pytest.raises(RuntimeError, match='exact committed receipt'): exporter.completed_csvs([path])

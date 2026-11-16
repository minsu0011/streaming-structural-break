"""Check the boundary between handoff summaries and observation/target records."""
import json
from pathlib import Path
import pytest
from scripts import package_next3_handoff as package


@pytest.mark.parametrize('name,content', [
    ('rows.csv', b'dataset_id,target,prediction\n1,1,.4\n'),
    ('rows.csv', b'\xef\xbb\xbfid,tau\n1,10\n'),
    ('rows.json', b'{"records":[{"dataset_id":1,"tau":10}]}'),
    ('observations.json', b'{"record":{"historical":[0,1],"online":[2]}}'),
])
def test_real_record_shaped_payloads_are_excluded(name, content):
    assert package.raw_record_reason(name, content)


@pytest.mark.parametrize('name,content', [
    ('summary.csv', b'candidate,mean_delta,ci_low,ci_high\nR1,.001,-.001,.002\n'),
    ('thresholds.json', b'{"label":"O1","threshold":0.9,"seed_pairs":512}'),
    ('manifest.json', b'{"dataset_id_hash":"0123","target_hash":"0456"}'),
    ('config.json', b'{"ids":[1,2,3],"feature_names":["scale","mean"]}'),
])
def test_aggregate_statistics_and_id_only_contracts_are_preserved(name, content):
    assert package.raw_record_reason(name, content) is None


@pytest.mark.parametrize('relative', ['data/raw/X_train.parquet', '.venv/key.txt',
    'artifacts/point_predictions.parquet', 'archive.zip', 'model.pickle', 'service.env', 'access_token.txt'])
def test_forbidden_ordinary_payload_paths_are_rejected(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    target = tmp_path/relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'fixture')
    with pytest.raises(RuntimeError): package.check_ordinary(target)


def test_ordinary_scientific_source_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    source = tmp_path/'src/next3/example.py'; source.parent.mkdir(parents=True)
    source.write_bytes(b'VALUE = 1\n')
    package.check_ordinary(source)
    assert package.stable_bytes(source) == b'VALUE = 1\n'


def test_snapshot_never_overwrites_an_existing_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'collect_paths', lambda **kwargs: ([], {}, {}, []))
    before = tmp_path/'source.txt'; before.write_bytes(b'preserve')
    with pytest.raises(RuntimeError, match='earlier payload'): package.snapshot_payload(tmp_path, require_audit=False)
    assert before.read_bytes() == b'preserve'

"""Regression contracts for the reviewed final-suite metadata exception."""
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from src.next5.artifacts import ROOT, digest
from src.next5.final_test_compatibility import validate_metadata, classify_path


@pytest.mark.parametrize('extra', ['tau', 'target', 'value', 'prediction'])
def test_metadata_exception_rejects_observation_or_target_columns(tmp_path, extra):
    path = tmp_path / 'metadata.parquet'
    columns = {'dataset_id': [1], 'group_id': [1], 'role': ['fixture'], 'fold': [0], extra: [1]}
    pq.write_table(pa.table(columns), path)
    with pytest.raises(RuntimeError, match='Only ID/group/role/fold'):
        validate_metadata(path, digest(path))


def test_metadata_exception_requires_exact_frozen_bytes(tmp_path):
    path = tmp_path / 'metadata.parquet'
    pq.write_table(pa.table({'dataset_id': [1], 'group_id': [1], 'role': ['fixture'], 'fold': [0]}), path)
    with pytest.raises(RuntimeError, match='bytes changed'):
        validate_metadata(path, '0' * 64)


def test_similarly_named_legacy_array_and_canonical_raw_stay_denied():
    assert classify_path(ROOT / 'artifacts/next4/GROUP_ASSIGNMENTS.parquet') is not None
    assert classify_path(ROOT / 'artifacts/validation/GROUP_ASSIGNMENTS_COPY.parquet') is not None
    assert classify_path(ROOT / 'data/raw/X_train.parquet') is not None
    assert classify_path(ROOT / 'data/next/example/series.npz') is not None

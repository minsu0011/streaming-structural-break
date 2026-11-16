"""Reviewed test-only metadata admission and identical legacy model loading.

The original blanket reader guard stays installed. Only the pinned ID allocation
table is admitted; old observations, labels, features and OOF remain forbidden.
"""
import os
from pathlib import Path
import pytest
from src.next5.artifacts import ROOT, BASE, read_json, digest, atomic_json, utc_now
from src.next5 import test_access_guard as original

METADATA = ROOT / 'artifacts/validation/GROUP_ASSIGNMENTS.parquet'
METADATA_SHA = 'db0a562da579eac2251c2d82301de16d2211f7bcc5212bc52240f49464f5bbee'
METADATA_COLUMNS = {'dataset_id', 'group_id', 'role', 'fold'}
_base_classifier = original.classify_path
_metadata_checks = []
_model_loads = []


def validate_metadata(path, expected_sha):
    import pyarrow.parquet as pq
    path = Path(path)
    if digest(path) != expected_sha:
        raise RuntimeError('Pinned ID metadata bytes changed')
    if set(pq.read_schema(path).names) != METADATA_COLUMNS:
        raise RuntimeError('Only ID/group/role/fold metadata columns are permitted')


def classify_path(value):
    source = value if isinstance(value, (str, bytes, os.PathLike)) else getattr(value, 'name', None)
    try:
        path = Path(os.fsdecode(source)).resolve() if source is not None else None
    except (TypeError, ValueError, OSError):
        path = None
    if path == METADATA.resolve():
        validate_metadata(path, METADATA_SHA)
        _metadata_checks.append(str(path))
        return None
    return _base_classifier(value)


def legacy_reference_weights_only(training_scope):
    if training_scope != 'full_fold0':
        raise ValueError('Only the four existing full-fold-0 scalar contracts are adapted')
    from src.next3.frozen import candidate, J102, C205
    from src.next2.io import model_class
    from src.next2.blend import FixedBlendBundle, source_hash
    from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
    models, proofs, row_hashes = [], {}, []
    for eid in (J102, C205):
        spec, directory = candidate(eid)
        directory = directory / 'full'
        report_path = directory / 'fold_0.json'
        report = read_json(report_path)
        if report['status'] != 'PASS':
            raise RuntimeError('Original frozen model receipt is not PASS')
        for name, sha in report['files_sha256'].items():
            if digest(directory / name) != sha:
                raise RuntimeError('Original model or coordinate bytes changed')
        row_hashes.append(report['files_sha256']['fold_0_global_rows.npy'])
        model_path = directory / 'fold_0.joblib'
        models.append(model_class(spec).load(model_path))
        proofs[eid] = {'receipt_sha256': digest(report_path),
            'model_path': model_path.relative_to(ROOT).as_posix(),
            'model_sha256': digest(model_path), 'original_files_sha256': report['files_sha256']}
    if row_hashes[0] != row_hashes[1]:
        raise RuntimeError('Original fold coordinate bytes do not match')
    model = FixedBlendBundle(models[0], models[1], .8,
        implementation_sha256=source_hash())
    model.validate()
    proof = {'training_scope': training_scope, 'original_fold': 0,
        'identical_original_weights': proofs, 'matching_coordinate_bytes_sha256': row_hashes[0],
        'old_per_series_arrays_decoded': False, 'new_fits': 0,
        'scope': 'Test-only loading; original numerical streaming assertions are unchanged. This loader does not re-evaluate historical training membership from arrays.'}
    _model_loads.append(proof)
    return PreparedSharedBlendPredictor(model, resident=False), proof


@pytest.fixture(autouse=True)
def keep_legacy_scalar_tests_raw_free(request, monkeypatch):
    if Path(request.module.__file__).resolve() == ROOT / 'tests/test_next4_robust_finalist_runtime.py':
        monkeypatch.setattr(request.module, 'load_reference', legacy_reference_weights_only)


def pytest_configure(config):
    validate_metadata(METADATA, METADATA_SHA)
    original.classify_path = classify_path
    original.pytest_configure(config)


def pytest_unconfigure(config):
    original.pytest_unconfigure(config)
    original.classify_path = _base_classifier
    directory = os.environ.get('NEXT5_TEST_AUDIT_DIRECTORY')
    if directory:
        out = Path(directory).resolve()
        if not out.is_relative_to(BASE / 'final_audit'):
            raise RuntimeError('Owned test audit directory required')
        atomic_json(out / 'TEST_COMPATIBILITY.json', {
            'created_utc': utc_now(), 'metadata_file': METADATA.relative_to(ROOT).as_posix(),
            'metadata_sha256': METADATA_SHA, 'metadata_columns': sorted(METADATA_COLUMNS),
            'metadata_admission_checks': len(_metadata_checks),
            'legacy_model_loads': _model_loads,
            'original_test_source_changed': False, 'raw_and_old_array_guard_retained': True,
            'source_sha256': digest(Path(__file__)),
            'scope': 'Test infrastructure only; admission checks are not counts of observation rows.'}, immutable=True)

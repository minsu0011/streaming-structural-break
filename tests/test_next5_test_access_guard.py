from src.next5.artifacts import ROOT
from src.next5.test_access_guard import classify_path


def test_raw_and_old_derived_arrays_denied_but_model_weights_and_synthetic_fixtures_allowed(tmp_path):
    assert classify_path(ROOT/'data/raw/X_train.parquet') is not None
    assert classify_path(ROOT/'data/next5/example.npy') is not None
    assert classify_path(ROOT/'artifacts/next4/example/OOF.npz') is not None
    assert classify_path(ROOT/'artifacts/next5/bootstrap/counts/PRIMARY_fold_0.npz') is None
    assert classify_path(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2') is None
    assert classify_path(tmp_path/'X_train.parquet') is None

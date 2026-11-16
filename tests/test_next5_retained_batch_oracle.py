"""A fixed synthetic fixture checks the independent retained-feature oracle."""
from pathlib import Path
import numpy as np
from scripts.verify_next5_retained_artifacts import matrix, collect
from src.next5.artifacts import BASE, ROOT, read_json


def test_retained_original_batch_features_match_both_actual_entries():
    import next2_shared_performance_main as primary_entry
    import next4_robustness_main as robustness_entry
    from src.next2.shared_inference import load_artifact
    from src.next4.robust_finalist import load_prepared
    from src.next2.blend import combine
    roles = read_json(BASE / 'preflight/FROZEN_REFERENCES.json')['roles']
    primary_dir = (ROOT / roles['FROZEN_PRIMARY']['model_file']).parent
    robust_dir = (ROOT / roles['NEXT4_ROBUSTNESS']['model_file']).parent
    model, _ = load_artifact(primary_dir)
    expert = load_prepared(robust_dir).model
    grid = np.arange(256, dtype=float)
    h = (np.sin(grid * .37) + .2 * np.cos(grid * .11)).astype(np.float32)
    o = np.linspace(-1., 1., 80, dtype=np.float32)
    o[30] = 8.
    left = model.primary.predict(matrix(model.primary, h, o))
    right = model.complement.predict(matrix(model.complement, h, o))
    baseline = combine(left, right, .8)
    extra = expert.predict(matrix(expert, h, o))
    robust = combine(baseline, extra, .9)
    np.testing.assert_array_equal(collect(primary_entry, primary_dir, h, o), baseline.astype(np.float64))
    np.testing.assert_array_equal(collect(robustness_entry, robust_dir, h, o), robust.astype(np.float64))

from dataclasses import asdict
from src.validation.adaptive import Candidate, select_next


def test_crossscale_ablation_preserves_calibrated_reference_and_configuration():
    spec = Candidate('M4_lightgbm', families='ABCDEFQ', sampling='S3',
                     params={'num_leaves': 7}, feature_changes={'normalization': 'median_mad'},
                     representation='historical_median_mad')
    result = {'experiment_id': spec.experiment_id, 'candidate': asdict(spec), 'model': spec.model,
              'families': spec.families, 'status': 'COMPLETE', 'mean': .6, 'median': .6,
              'worst_fold': .59, 'std': .01, 'time_profile': {'0-9': .51}}
    attempted = {spec.identity}; found = []
    for _ in range(150):
        candidate = select_next([result], attempted, primary_id=spec.experiment_id)
        if candidate is None: break
        attempted.add(candidate.identity)
        if candidate.phase == 'memory_crossscale_ablation': found.append(candidate)
    assert {c.families for c in found} == {'ABCDE', 'ABCDF', 'E', 'F', 'EF'}
    for candidate in found:
        assert candidate.feature_changes == spec.feature_changes
        assert candidate.params == spec.params
        assert candidate.representation == spec.representation
        assert candidate.sampling == spec.sampling

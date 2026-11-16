"""Freeze actual raw-free runtime dependencies after prediction release."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.portable_replay import required_runtime_paths, REPLAY_CASES, HISTORICAL_CASES


def main():
    lock = read_json(BASE / 'PORTABLE_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    release = read_json(BASE / 'PREDICTION_FREEZE_NEXT8.json')
    assert release['status'] == 'FROZEN'
    with activity('Freeze portable actual-model artifact dependencies', 'PORTABLE_PREPARATION', 'B7_NORMALIZATION_AND_LINEAGE'):
        paths, scopes = required_runtime_paths()
        baseline = read_json(BASE / 'BASELINE_PROVENANCE_NEXT8.json')
        files = {}
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            sha = digest(path)
            if name in baseline['files_sha256']:
                assert sha == baseline['files_sha256'][name]
            if path.is_relative_to(BASE):
                key = path.relative_to(BASE).as_posix()
                if key in release['files_sha256']:
                    assert sha == release['files_sha256'][key]
            files[name] = sha
        atomic(BASE / 'PORTABLE_RUNTIME_MANIFEST_NEXT8.json', {
            'status': 'FROZEN', 'timestamp': utc_now(), 'files_sha256': files, 'scopes': scopes,
            'normal_research_folds': 5, 'early_only_research_folds': 5,
            'research_cases': REPLAY_CASES, 'short_H_research_head_cases': [0, 159],
            'canonical_historical_cases': HISTORICAL_CASES,
            'canonical_fixtures_are_synthetic_not_competition_OOF': True,
            'real_point_OOF_or_feature_cache_needed': False,
            'models_are_existing_actual_frozen_models_not_refitted_surrogates': True,
            'five_full_TRAIN_models_are_for_reproduction_not_a_new_ensemble_candidate': True,
            'source_dependency_lock_sha256': digest(BASE / 'PORTABLE_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json'),
            'original_prediction_freeze_sha256': digest(BASE / 'PREDICTION_FREEZE_NEXT8.json')})
    note('Portable actual-model manifest frozen', 'Fixture execution awaits separate final audit admission', 'B7_NORMALIZATION_AND_LINEAGE')


if __name__ == '__main__':
    main()

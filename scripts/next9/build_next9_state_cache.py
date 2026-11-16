"""All153 target-free movement states, built once from observed feature history."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, ROOT, atomic, read_json, digest, activity, utc_now
from src.next9.boundary import install_workspace_boundary, target_free_npz


def main(part):
    violations = install_workspace_boundary()
    lock = read_json(BASE / 'STATE_CACHE_IMPLEMENTATION_LOCK_NEXT9.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    with target_free_npz() as members:
        import numpy as np
        from src.next9.data import FeatureBank
        from src.next9.io import save_npz
        bank = FeatureBank()
        gate = read_json(BASE / 'TARGET_FREE_BANK_TEST_GATE_NEXT9.json')
        assert gate['status'] == 'PASS' and digest(BASE / 'state_cache/INDEX.npz') == gate['state_index_sha256']
        with np.load(BASE / 'state_cache/INDEX.npz') as z:
            rows = z['global_row']
            np.testing.assert_array_equal(z['dataset_id'], bank.row_ids[rows])
        features = list(range(153))[part::4]
        records = []
        for j in features:
            out = BASE / 'state_cache' / f'F{j:03d}'
            if (out / 'DONE.json').exists():
                done = read_json(out / 'DONE.json')
                assert done['implementation_lock_sha256'] == digest(BASE / 'STATE_CACHE_IMPLEMENTATION_LOCK_NEXT9.json')
                assert digest(out / 'STATES.npz') == done['states_sha256']
                records.append(done)
                continue
            with activity(f'Build all RP sampled pre-current movement states F{j:03d}', 'FEATURES', 'B9_NORMALIZATION_STATE_AVAILABILITY'):
                out.mkdir(parents=True, exist_ok=True)
                pieces = {}
                for start in range(0, len(rows), 32768):
                    state = bank.pre_state(rows[start:start+32768], j)
                    for name, values in state.items():
                        if name in ('lags', 'delta_prev', 'history_available'):
                            continue
                        dtype = np.float32 if name == 'slope8' else np.uint8
                        pieces.setdefault(name, []).append(np.asarray(values, dtype=dtype))
                save_npz(out / 'STATES.npz', **{name: np.concatenate(value) for name, value in pieces.items()})
                done = dict(status='COMPLETE', timestamp=utc_now(), feature=j, rows=len(rows),
                            states_sha256=digest(out / 'STATES.npz'), input_index_sha256=gate['state_index_sha256'],
                            feature_manifest_sha256=read_json(BASE / 'PREFLIGHT_NEXT9.json')['feature_manifest_sha256'],
                            implementation_lock_sha256=digest(BASE / 'STATE_CACHE_IMPLEMENTATION_LOCK_NEXT9.json'),
                            labels_tau_or_outer_scores_used=False, current_F_excluded_from_state=True,
                            population_fitted_state_mapping=False)
                atomic(out / 'DONE.json', done)
                records.append(done)
                print('PRESTATE', part, j, len(rows), flush=True)
        assert not violations
        atomic(BASE / 'state_cache' / f'PART_{part}_DONE.json', dict(status='COMPLETE', timestamp=utc_now(),
              features=features, records=records, boundary_violations=violations, accessed_NPZ_members=sorted(set(members)),
              no_break_label_or_tau_read=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--part', required=True, type=int, choices=range(4))
    main(p.parse_args().part)

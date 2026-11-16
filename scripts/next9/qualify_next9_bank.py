"""Read no labels; verify original partitions and scalar/cache state alignment."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, ROOT, atomic, read_json, digest, activity, utc_now
from src.next9.boundary import install_workspace_boundary, target_free_npz


def main():
    violations = install_workspace_boundary()
    with target_free_npz() as accesses, activity('Qualify target-free bank, all original nested partitions and pre-state scalar parity', 'GUARD', 'B9_NORMALIZATION_STATE_AVAILABILITY'):
        import numpy as np
        from src.next9.data import FeatureBank, outer_scope, inner_scope, selection_splits
        from src.next9.states import MovementState
        from src.next9.io import save_npz
        bank = FeatureBank()
        scopes = []
        for fold in range(5):
            for inner in [None, 0, 1, 2]:
                parent = outer_scope(bank, fold) if inner is None else inner_scope(bank, fold, inner)
                parts = selection_splits(bank, fold, inner)
                assert set().union(*(p.validation_ids for p in parts)) == parent.training_ids
                assert sum(len(p.validation_ids) for p in parts) == len(parent.training_ids)
                for part in parts:
                    assert part.training_ids | part.validation_ids == parent.training_ids
                scopes.append(dict(parent=parent.provenance(), fold=fold, inner=inner, parts=[p.provenance() for p in parts]))
        # Predetermined first ID of each original fold; all153 channels, prefix97.
        comparisons = 0
        for fold in range(5):
            index = int(np.flatnonzero(bank.folds == fold)[0])
            a, b = map(int, bank.offsets[index:index+2])
            rows = np.arange(a, min(b, a + 97), dtype=np.int64)
            for j in range(153):
                stream = MovementState(bank.series['seed_values'][index, :, j], float(bank.series['epsilon'][index, 0, j]), bool(bank.series['H_length'][index] >= 160))
                expected = bank.pre_state(rows, j)
                for k, row in enumerate(rows):
                    actual = stream.peek()
                    for key in actual:
                        np.testing.assert_array_equal(actual[key], expected[key][k])
                    stream.observe(float(bank.values[row, j]))
                    comparisons += 1
        assert not violations
        sample = bank.sampled_rows(bank.ids)
        save_npz(BASE / 'state_cache/INDEX.npz', global_row=sample, dataset_id=bank.row_ids[sample], time_online=bank.times[sample])
        atomic(BASE / 'ORIGINAL_NESTED_SCOPES_NEXT9.json', dict(status='FROZEN', timestamp=utc_now(), scopes=scopes,
              original_outer_inner_subinner_preserved=True, label_access=False))
        atomic(BASE / 'TARGET_FREE_BANK_TEST_GATE_NEXT9.json', dict(status='PASS', timestamp=utc_now(),
              state_scalar_vector_comparisons=comparisons, features=153, fixed_series=5,
              admitted_RP_series=6400, scope_count=len(scopes), selection_split_count=sum(len(s['parts']) for s in scopes),
              no_break_labels_decoded=True, no_tau_decoded=True, read_NPZ_members=sorted(set(accesses)),
              boundary_violations=violations, state_index_sha256=digest(BASE / 'state_cache/INDEX.npz'),
              sampled_rows=len(sample), protected_preflight_sha256=digest(BASE / 'PREFLIGHT_NEXT9.json')))
        print('Target-free bank and original nested partitions PASS', comparisons, flush=True)


if __name__ == '__main__':
    main()

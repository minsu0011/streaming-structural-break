"""Record the failed in-memory fit and preserve its pre-fit pair plan."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, digest, utc_now


def main():
    old = read_json(BASE / 'G3_IMPLEMENTATION_LOCK.json')['source_sha256']
    for name, snapshot in [('src/next7/pairwise.py', 'pairwise.py'),
        ('scripts/next7/fit_g3_pairwise.py', 'fit_g3_pairwise.py'), ('src/next7/tests/test_pairwise.py', 'test_pairwise.py')]:
        assert digest(BASE / ('audit/sources/G3_STANDARDIZATION_BEFORE_' + snapshot)) == old[name]
    assert not list((BASE / 'g3').glob('fold_*/model.joblib'))
    assert not list((BASE / 'g3').glob('fold_*/predictions.npz'))
    plan = BASE / 'cache/g3_pairs/fold_0.npz'
    receipt = BASE / 'g3/fold_0/PAIR_PLAN_FROZEN.json'
    assert digest(plan) == read_json(receipt)['file_sha256']
    save('G3_STANDARDIZATION_BUG_REPAIR_001.json', {'created_utc': utc_now(),
        'old_source_sha256': old, 'new_source_sha256': {n: digest(ROOT / n) for n in old},
        'completed_in_memory_optimizer_before_repair': 1, 'serialized_models_before_repair': 0,
        'outer_predictions_before_repair': 0, 'outer_scores_before_repair': 0,
        'observed_center_max_error': 3.59101637e-13,
        'cause': 'G3 reduced the C-contiguous input directly; G2 first selects all columns by advanced indexing and uses its resulting memory layout.',
        'repair': 'Use the identical advanced FULL-column indexing before G3 moments. Keep exact equality gate, objective, penalty, sampling and optimizer unchanged.',
        'test_xml_sha256': digest(BASE / 'audit/G3_PAIRWISE_TESTS_002.xml'), 'tests_passed': 8,
        'preserved_partial_pair_plan_sha256': digest(plan), 'preserved_pair_receipt_sha256': digest(receipt),
        'partial_files_archive': 'audit/G3_STANDARDIZATION_PARTIAL',
        'no_scientific_gate_relaxation': True})


if __name__ == '__main__':
    main()

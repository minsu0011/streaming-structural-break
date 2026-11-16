"""Check every serialized estimator envelope against original grouped scopes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.training_lineage_audit import audit_lineage


def main():
    lock = read_json(BASE / 'TRAINING_LINEAGE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    with activity('Independent original-group model lineage audit', 'TRAINING_LINEAGE_AUDIT', 'B9_NESTED_SELECTION'):
        result = audit_lineage()
        atomic(BASE / 'TRAINING_LINEAGE_AUDIT_NEXT8.json', {'timestamp': utc_now(), **result})
    note('Original-group model lineage audit PASS', 'Recorded model envelopes, exact feature selection membership and complete state crossfit', 'B9_NESTED_SELECTION')


if __name__ == '__main__':
    main()

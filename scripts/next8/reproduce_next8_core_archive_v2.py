"""Create and actually replay the first raw-free audit payload."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, note
from src.next8.runtime_accounting import audit_admitted
from src.next8.archive_reproduction_v2 import archive_and_reproduce


def main():
    assert not (BASE / 'CORE_ARCHIVE_REPRODUCTION_NEXT8.json').exists()
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    hb = read_json(BASE / 'runtime/HEARTBEAT.json')
    assert audit_admitted(admission, hb['productive_seconds'], digest(BASE / 'SESSION_NEXT8.json'), 36000.)
    closure = read_json(BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')
    for name, sha in closure['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    assert read_json(BASE / 'computation_audit/DONE.json')['status'] == 'COMPLETE'
    result = archive_and_reproduce('CORE')
    atomic(BASE / 'CORE_ARCHIVE_REPRODUCTION_NEXT8.json', result)
    note('Raw-free extracted core ZIP replay PASS', 'Actual models and canonical public entries; final publication ZIP remains separate',
         'B7_NORMALIZATION_AND_LINEAGE', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()

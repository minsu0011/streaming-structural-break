"""Inspect completed-work time or open a phase gate after every requirement."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic
from src.next8.strict_work_time import completed_work_snapshot, admit_research_completion, admit_final_handoff


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('preview', 'admit-research', 'admit-handoff'), default='preview')
    args = p.parse_args()
    lock = read_json(BASE / 'STRICT_WORK_TIME_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    assert read_json(BASE / 'STRICT_WORK_TIME_TEST_GATE_NEXT8.json')['status'] == 'PASS'
    if args.stage == 'preview':
        ap = BASE / 'runtime/FINAL_AUDIT_ADMISSION.json'
        result = completed_work_snapshot(read_json(ap) if ap.exists() else None)
        # Only a compact live diagnostic; full hashes are retained at admission.
        display = {k: v for k, v in result.items() if k not in ('ledger_sha256', 'counted_activities')}
        display['counted_completed_activities'] = len(result['counted_activities'])
        atomic(BASE / 'runtime/STRICT_WORK_HEARTBEAT_NEXT8.json', display, immutable=False)
        print(display)
    else:
        print(admit_research_completion() if args.stage == 'admit-research' else admit_final_handoff())


if __name__ == '__main__':
    main()

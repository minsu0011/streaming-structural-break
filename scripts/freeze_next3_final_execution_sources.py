"""Freeze the actual final-audit code/config set, never before the time gate."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_source_contract import snapshot_sources
from scripts.prepare_next3_independent_reproduction import time_gate


def main():
    timing = time_gate(); snapshot = snapshot_sources(ROOT)
    scientific = read_json(ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
    assert scientific['scientific_role_count'] == 2 and not scientific['new_scientific_search_authorized']
    atomic_json(BASE/'final_audit/FINAL_EXECUTION_SOURCE_LOCK.json', {
        'status': 'FROZEN_FINAL_EXECUTION_SOURCE', 'created_utc': utc_now(), 'time_gate': timing,
        **snapshot, 'scientific_role_lock_sha256': digest(ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json'),
        'new_fits': 0, 'source_sha256': digest(Path(__file__)),
        'repair_rule': 'Preserve this lock and any failing attempt. A source correction requires a new explicit lock and fresh affected evidence.'}, immutable=True)
    progress('Final execution source set frozen', str(snapshot['file_count'])+' current code/config/dependency files bound before final tests and raw reproduction', phase='FINAL_AUDIT')
    print('NEXT3 FINAL EXECUTION SOURCE LOCK', snapshot['file_count'], snapshot['content_set_sha256'], flush=True)


if __name__ == '__main__': main()

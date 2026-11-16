"""Final report assembly, fresh final-ZIP replay and exactly20 Desktop files."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.final_reports import build_final_reports
from src.next8.archive_reproduction import archive_and_reproduce
from src.next8.desktop_delivery import deliver_twenty


def main():
    admission = read_json(BASE / 'runtime/FINAL_HANDOFF_ADMISSION_NEXT8.json')
    assert admission['status'] == 'PASS' and admission['productive_seconds_observed'] >= 36000.
    assert admission['separate_audit_seconds_observed'] >= 3600.
    assert not (BASE / 'DELIVERY_RECEIPT_NEXT8.json').exists()
    closure = read_json(BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')
    for name, sha in closure['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    for task in read_json(BASE / 'runtime/TASK_QUEUE.json'):
        if task['id'] == 'FINAL_PUBLICATION':
            continue
        path = BASE / 'runtime/queue' / task['id'] / 'TERMINAL.json'
        assert path.exists() and read_json(path)['status'] == 'COMPLETE', 'Unfinished registered task: ' + task['id']
    with activity('Final Korean A-J reports, measured time and evidence manifest', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        directory = build_final_reports()
    result = archive_and_reproduce('FINAL', directory)
    atomic(BASE / 'FINAL_ARCHIVE_REPRODUCTION_NEXT8.json', result)
    with activity('Verified exactly20 Desktop handoff files and matching archives', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        delivery = deliver_twenty(directory, result)
    atomic(BASE / 'runtime/FINAL_COMPLETION_NEXT8.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
        'delivery_receipt_sha256': digest(BASE / 'DELIVERY_RECEIPT_NEXT8.json'),
        'Desktop_directory': delivery['Desktop_directory'], 'scientific_status': read_json(directory / 'FINAL_AUDIT_NEXT8.json')['scientific_status'],
        'scheduler_should_be_stopped_after_this_queue_terminal_is_committed': True})
    note('NEXT8 final verified Desktop delivery complete', delivery['Desktop_directory'],
         'B7_NORMALIZATION_AND_LINEAGE', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()

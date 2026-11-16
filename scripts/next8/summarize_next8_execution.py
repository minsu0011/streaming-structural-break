"""Assemble actual fit counts, aliases and measured runtime evidence."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.execution_summary import fit_inventory, summarize_latency
from src.next8.research_reports import table


def main():
    lock = read_json(BASE / 'EXECUTION_SUMMARY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    assert read_json(BASE / 'PREDICTION_FREEZE_NEXT8.json')['status'] == 'FROZEN'
    output = BASE / 'execution_summary'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('Existing execution summaries preserved')
    with activity('Inventory fit calls, reuse and measured runtime', 'EXECUTION_REPORTING', 'B3_SELF_HISTORY_SUFFICIENCY'):
        fitting = fit_inventory()
        latency = summarize_latency()
        atomic(output / 'FIT_ACCOUNTING_NEXT8.json', {'timestamp': utc_now(), **fitting})
        atomic(output / 'LATENCY_NEXT8.json', {'timestamp': utc_now(), **latency})
        table(output / 'FIT_CALL_EVIDENCE.csv', fitting['fit_count_evidence'])
        table(output / 'LATENCY_NEXT8.csv', latency['rows'])
        atomic(output / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
               'files_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
               'scientific_quality_or_candidate_settings_changed': False})
    note('Execution evidence assembled', 'Actual fits, aliases and measured latency remain separate', 'B3_SELF_HISTORY_SUFFICIENCY')


if __name__ == '__main__':
    main()

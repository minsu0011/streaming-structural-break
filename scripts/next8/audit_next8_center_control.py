"""Supplementary fixed H-center audit, without altering any fitted model."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.cache import FeatureCache
from src.next8.center_control import summarize_center_control, LIMIT_KO
from src.next8.research_reports import table


def main():
    lock = read_json(BASE / 'CENTER_CONTROL_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    output = BASE / 'center_control'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('Existing center-control evidence preserved')
    with activity('Separate persistence improvement from H-center prediction', 'CENTER_CONTROL_AUDIT', 'B3_SELF_HISTORY_SUFFICIENCY'):
        cache = FeatureCache()
        results = []
        for early in (False, True):
            result = summarize_center_control(cache, early)
            atomic(output / ('EARLY.json' if early else 'NORMAL.json'), {'timestamp': utc_now(), **result})
            results.append(result)
        table(output / 'SELF_HISTORY_CENTER_CONTROL_NEXT8.csv', [r for result in results for r in result['rows']])
        table(output / 'CENTER_DOMAIN_RESULTS.csv', [r for result in results for r in result['domain_rows']])
        atomic(output / 'SELF_HISTORY_BASELINE_LIMIT_KO.md', LIMIT_KO)
        atomic(output / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
               'files_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
               'original_models_gates_and_certificates_not_revised': True})
    note('H-center persistence audit complete', 'Do not equate persistence improvement with new memory information', 'B3_SELF_HISTORY_SUFFICIENCY')


if __name__ == '__main__':
    main()

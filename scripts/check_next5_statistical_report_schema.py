"""Check saved-draw/report consistency before the later raw-OOF audit."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.audit_next5_frozen_statistics import old_draws, close
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.bootstrap import paired_summary


def main():
    out = BASE / 'integrity_audits/statistical_report_schema_v1'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'source_sha256': digest(Path(__file__)),
        'scope': 'Saved aggregate draw arrays only; no raw observations, real point OOF, new fits or new draws'}, immutable=True)
    checked = []
    for path in sorted((BASE / 'experiments').glob('*/full/RESULTS.json')):
        cid = path.parent.parent.name
        forensic = BASE / 'forensic' / cid
        receipt = read_json(forensic / 'RESULTS.json')
        table = forensic / 'PAIRED_FRACTURES.csv'
        assert digest(table) == receipt['files_sha256'][table.name]
        with table.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 26
        for row in rows:
            scope = row['scope']
            old = np.stack([old_draws(cid, fold, scope, baseline=True) for fold in range(5)], axis=1)
            new = np.stack([old_draws(cid, fold, scope) for fold in range(5)], axis=1)
            assert old.shape == new.shape == (2001, 5)
            summary = paired_summary(old, new)
            for name, value in summary.items():
                close(value, row[name])
            close(new[0].mean(), row['candidate_mean'])
            close(old[0].mean(), row['baseline_mean'])
            if scope == 'all_time':
                assert summary == read_json(BASE / 'bootstrap' / cid / 'RESULTS.json')['paired_all_time']
            checked.append({'candidate': cid, 'scope': scope, 'saved_draws_and_CSV_summary_match': True})
        progress('Saved statistical report schema verified', cid + ': all 26 scopes use existing exact 2001-by-5 arrays')
    assert len(checked) == 19 * 26
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS', 'candidate_scopes': len(checked),
        'records': checked, 'new_fits': 0, 'new_draws': 0, 'real_point_OOF_read': False,
        'future_final_audit_source_sha256': digest(ROOT / 'scripts/audit_next5_frozen_statistics.py'),
        'limitation': 'This confirms saved-array and table consistency. Independent raw-OOF fixed-count recomputation and official scorer execution remain for the separate final audit.',
        'source_sha256': digest(Path(__file__))}, immutable=True)
    print('NEXT5 SAVED STATISTICAL SCHEMA PASS', len(checked), 'candidate/scopes', flush=True)


if __name__ == '__main__':
    main()

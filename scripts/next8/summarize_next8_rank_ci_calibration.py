"""Preserve every known-truth confidence-interval experiment in handoff."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, csv_write, utc_now, activity
from src.next8.rank_ci_calibration import cells, aggregate


def main():
    lock = read_json(BASE / 'RANK_CI_CALIBRATION_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    with activity('Interpret full fixed rank-interval calibration', 'RESULT_INTERPRETATION', 'B1_CONDITIONAL_DIRECTION'):
        rows, records, sources = [], [], {}
        for cell, (groups, rho, shift) in enumerate(cells()):
            out = BASE / 'rank_ci_calibration' / f'cell_{cell:02d}'
            done = read_json(out / 'DONE.json')
            assert done['status'] == 'COMPLETE' and (done['groups'], done['rho'], done['mean_shift']) == (groups, rho, shift)
            current = []
            for name, sha in sorted(done['files_sha256'].items()):
                assert digest(out / name) == sha
                current.append(read_json(out / name))
            rows.extend(aggregate(current))
            records.extend(current)
            sources[str(cell)] = digest(out / 'DONE.json')
        assert len(records) == 1200 and len(rows) == 24
        csv_write('RANK_CI_CALIBRATION_NEXT8.csv', rows)
        atomic(BASE / 'RANK_CI_CALIBRATION_NEXT8.json', {
            'status': 'COMPLETE', 'timestamp': utc_now(), 'summary': rows, 'all_replicates': records,
            'cell_DONE_sha256': sources, 'summary_table_sha256': digest(BASE / 'RANK_CI_CALIBRATION_NEXT8.csv'),
            'model_fits': 0, 'RP_score_bootstrap_or_interval_changes': 0,
            'limitations': [
                'Sampling calibration for a fixed synthetic score with known population AUC; it excludes fitted-model, feature-selection and historical adaptive DEV uncertainty.',
                'Wilson intervals quantify100-replicate Monte Carlo uncertainty per cell; they are not simultaneous across all24 method/cell summaries.',
                'Within-series correlation and group count are controlled. Actual conditional-state censoring, clipping, population mixtures and all time/H strata are not exhausted.',
                'This post-outer diagnostic can qualify interpretation but never changes the frozen RP score, selector, state, horizon, model or reported original CI.',
                'Both interval methods and every replicate are retained, including failures and insufficient bootstrap support.'
            ]})


if __name__ == '__main__':
    main()

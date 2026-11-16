"""Run original full TRAIN discovery plus H selection on one frozen raw law."""
from pathlib import Path
from collections import Counter
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, csv_write, utc_now, activity
from src.next8.raw_engine_gate_control import BANKS, load_fixture
from src.next8.screening import screen_split, summarize_screen
from src.next8.h_context import audit_h_split, combine_selection


def main(bank):
    lock = read_json(BASE / 'RAW_ENGINE_GATE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    out = BASE / 'raw_engine_gate_control' / bank
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Partial result retained; no silent repetition'
    with activity('Full original selector on synthetic raw law ' + bank, 'SYNTHETIC_GATE_CONTROL', 'B1_CONDITIONAL_DIRECTION'):
        fixture = load_fixture(BASE / 'raw_engine_gate_control/cache', bank)
        panels, corrections = [], []
        for k, scope in enumerate(fixture.scopes):
            p, c = screen_split(fixture.cache, scope, out / ('inner' + str(k)))
            panels.append(p)
            corrections.append(c)
        initial = summarize_screen(panels, corrections, fixture.parent)
        atomic(out / 'INITIAL_SELECTION.json', initial)
        h_rows, h_branches, h_corrections = [], [], []
        for k, scope in enumerate(fixture.scopes):
            r, b, c = audit_h_split(fixture.cache, scope, out / ('inner' + str(k)), out / ('H_inner' + str(k)), initial['stable_features'])
            h_rows.append(r)
            h_branches.append(b)
            h_corrections.append(c)
        final = combine_selection(fixture.cache, fixture.parent, initial, h_rows, h_branches, h_corrections)
        final['parent_full_initial_selection_sha256'] = digest(out / 'INITIAL_SELECTION.json')
        rows = [{'bank': bank, 'feature': j, 'name': r['name'], 'initial_classification': initial['classifications'][j]['classification'],
                 'final_classification': r['classification'], 'selected_stable': j in final['stable_features'],
                 'selected_variable': j in final['variable_features'], 'synthetic_only_not_RP_certification': True}
                for j, r in enumerate(final['classifications'])]
        table = 'RAW_ENGINE_GATE_CONTROL_' + bank + '_NEXT8.csv'
        csv_write(table, rows)
        initial_fits = sum(3 + (4 if r['conditional']['status'] == 'COMPLETE' else 0) for p in panels for r in p)
        h_fits = sum(int(r['reconstructed_O1'] is not None) + 4 * sum(c['status'] == 'COMPLETE' for c in r['conditional'].values())
                     for p in h_rows for r in p)
        result = {'status': 'COMPLETE', 'timestamp': utc_now(), 'bank': bank, 'series': 5120,
            'initial_selection': initial, 'final_selection': final, 'all_initial_inner_records': panels,
            'all_H_inner_records': h_rows, 'initial_corrections': corrections, 'H_corrections': h_corrections,
            'initial_classification_counts': dict(Counter(r['initial_classification'] for r in rows)),
            'final_classification_counts': dict(Counter(r['final_classification'] for r in rows)),
            'statistical_fixture_fits': initial_fits + h_fits, 'initial_fit_calls': initial_fits, 'H_fit_calls': h_fits,
            'H_map_quantile_fits_separate': 6 + sum(s['kind'] != 'OWN_F' for s in final['orientation_state_specs'].values()),
            'actual_RP_fit_prediction_score_bootstrap': 0, 'no_outer_B_JOINT_or_promotion_claim': True,
            'cache_DONE_sha256': digest(BASE / 'raw_engine_gate_control/cache/DONE.json'),
            'policy_sha256': digest(BASE / 'RAW_ENGINE_GATE_CONTROL_POLICY_NEXT8.json'),
            'table_sha256': digest(BASE / table),
            'inner_DONE_sha256': {str(k): digest(out / ('inner' + str(k)) / 'DONE.json') for k in range(3)},
            'H_inner_DONE_sha256': {str(k): digest(out / ('H_inner' + str(k)) / 'DONE.json') for k in range(3)},
            'execution_scope_note': 'Original H code records before_outer_evaluation because its inputs are TRAIN-only; this entire synthetic diagnostic is post-release and no synthetic outer set is scored.'}
        name = 'RAW_ENGINE_GATE_CONTROL_' + bank + '_NEXT8.json'
        atomic(BASE / name, result)
        atomic(out / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(), 'root_result_sha256': digest(BASE / name)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', choices=BANKS, required=True)
    main(parser.parse_args().bank)

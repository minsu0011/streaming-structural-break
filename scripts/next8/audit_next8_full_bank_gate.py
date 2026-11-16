"""Run the single preregistered full-bank diagnostic with unchanged gates."""
from pathlib import Path
import sys
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, csv_write, utc_now, activity, note
from src.next8.full_bank_gate_control import full_bank_fixture, full_bank_names, SIGNALS


def main():
    lock = read_json(BASE / 'FULL_BANK_GATE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    policy = read_json(BASE / 'FULL_BANK_GATE_CONTROL_POLICY_NEXT8.json')
    assert policy['bank_count'] == 1 and policy['features'] == 153
    assert policy['known_signal_mapping'] == {str(k): v for k, v in SIGNALS.items()}
    out = BASE / 'full_bank_gate_control'
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Partial diagnostic is retained; no silent repetition'
    with activity('Full153 joint-screen known-signal diagnostic', 'SYNTHETIC_GATE_CONTROL', 'B1_CONDITIONAL_DIRECTION'):
        fixture = full_bank_fixture()
        panels, corrections = [], []
        with full_bank_names() as screening:
            for k, scope in enumerate(fixture.scopes):
                panel, correction = screening.screen_split(fixture.cache, scope, out / ('inner' + str(k)))
                panels.append(panel)
                corrections.append(correction)
            selection = screening.summarize_screen(panels, corrections, fixture.parent)
        assert len(selection['classifications']) == 153 and len(panels) == 3
        rows = []
        for j, r in enumerate(selection['classifications']):
            rows.append({'feature': j, 'artificial_world': SIGNALS.get(j, 'NULL'),
                         'classification': r['classification'],
                         'invariant_inner_passes': sum(p[j]['invariant_direction_supported'] for p in panels),
                         'nonlinear_inner_passes': sum(p[j]['fixed_nonlinear_supported'] for p in panels),
                         'conditional_repeated_inner_folds': r['conditional_supported_folds'],
                         'selected_stable': j in selection['stable_features'],
                         'selected_variable': j in selection['variable_features'],
                         'actual_RP_feature_or_candidate': False})
        csv_write('FULL_BANK_GATE_CONTROL_RESULTS_NEXT8.csv', rows)
        fit_count = sum(3 + (4 if r['conditional']['status'] == 'COMPLETE' else 0) for p in panels for r in p)
        result = {'status': 'COMPLETE', 'timestamp': utc_now(), 'bank_count': 1,
            'series': 5120, 'features': 153, 'inner_folds': 3, 'screened_feature_inner_units': 459,
            'selection': selection, 'classification_counts': dict(Counter(r['classification'] for r in rows)),
            'actual_Holm_family_sizes': [len(c) for c in corrections],
            'conditional_corrections': corrections, 'all_inner_records': panels,
            'coordinate_sha256': fixture.coordinate_sha256, 'statistical_fixture_fits': fit_count,
            'new_real_RP_fit_prediction_score_bootstrap': 0,
            'table_sha256': digest(BASE / 'FULL_BANK_GATE_CONTROL_RESULTS_NEXT8.csv'),
            'policy_sha256': digest(BASE / 'FULL_BANK_GATE_CONTROL_POLICY_NEXT8.json'),
            'implementation_lock_sha256': digest(BASE / 'FULL_BANK_GATE_IMPLEMENTATION_LOCK_NEXT8.json'),
            'inner_DONE_sha256': {str(k): digest(out / ('inner' + str(k)) / 'DONE.json') for k in range(3)},
            'limits': ['One artificial bank, not a general power or familywise error rate estimate.',
                      'Original153 screening,12 detailed eligibility, actual shared Holm family and top6 budgets are exercised.',
                      'Oracle population H constants are used. Real H60/H40 extraction and second-stage H-context selection are not replayed.',
                      'No outer B or JOINT certification; no real candidate, model, gate or prediction changed.',
                      'The150 null coordinates share the same5120 target series; they are not150 independent experiments.']}
        atomic(BASE / 'FULL_BANK_GATE_CONTROL_RESULTS_NEXT8.json', result)
        atomic(out / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
            'aggregate_sha256': digest(BASE / 'FULL_BANK_GATE_CONTROL_RESULTS_NEXT8.json'),
            'statistical_fixture_fits': fit_count})
    note('Full153 joint-screen diagnostic complete', str(result['classification_counts']), 'B1_CONDITIONAL_DIRECTION')


if __name__ == '__main__':
    main()

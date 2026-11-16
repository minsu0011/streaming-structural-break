"""Second review: exact ranking mechanism and full G2 state evidence."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, note


def main():
    with activity('bottleneck cycle002: null admission versus detection loss', 'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        prior = read_json(BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_001.json')
        items = {r['id']: r for r in prior['items']}
        summaries = []
        protocol = read_json(BASE / 'NEXT7_PROTOCOL_LOCK.json')
        for case in protocol['evaluation']['cases']:
            directory = BASE / f'synthetic_g2_full/{case}'
            done = read_json(directory / 'DONE.json')
            assert done['status'] == 'COMPLETE' and done['seeds'] == 512
            assert digest(directory / 'SUMMARY.csv') == done['SUMMARY_sha256']
            with (directory / 'SUMMARY.csv').open(newline='', encoding='utf-8') as f:
                summaries.extend(csv.DictReader(f))
        bycase = {(r['case'], int(r['horizon'])): r for r in summaries}
        null_rates = {c: float(bycase[c, 16384]['any_latch']) for c in ('N0', 'N1', 'N2', 'N3')}
        scale_rates = {c: float(bycase[c, 1024]['first_detection']) for c in ('V1', 'V2', 'V3', 'V4')}
        items['B1'].update(status='G1_SCALE_REPRESENTATION_CLOSED_G2_REUSES_UNCHANGED',
            evidence='G2 uses exactly the same S1/C1 scale; no representation or cutoff selected after G1 failure.',
            next_test='Fixed H transport and actual RP mechanism strata; no threshold adjustment')
        items['B2'].update(status='PAIRED_SHOCK_UNCERTAINTY_NEXT',
            evidence='G2 full512 shock/control paths complete; state-only latches must remain separate from final BL80 detector alarms.',
            next_test='2000 paired whole-seed intervals, excess-memory trajectories and calibrated final detector comparison')
        items['B3'].update(status='EXACT_RANK_PAIR_CAUSE_IDENTIFIED',
            evidence='M_EVENT delta -.001374571894; P0/N1 false-negative-side latch contribution -.011999010879, P1/N0 true-positive-only contribution +.010647958159; exact additive decomposition.',
            confidence='Exact for frozen RP predictions; algebraic attribution, not an intervention effect',
            next_test='Qualified-event RP attribution, first-event age and observed-time fractures')
        items['B4'].update(status='G2_FULL_STATE_GATE_PASSES_EMPIRICALLY',
            evidence=f'G2 qualified 16384 null latch: {null_rates}; all below the preregistered .50 rejection threshold. V1..4 first detection1024: {scale_rates}.',
            severity='HIGH',
            confidence='512 fixed paired paths per environment; confidence intervals pending; H bootstrap has no anytime FWER guarantee',
            next_test='H-length transport, false-latch survival and support-conditioned calibration error',
            stop_rule='Do not lower thresholds after detection loss; no new qualified events beyond1024 observed blocks by explicit contract')
        items['B5'].update(status='NEXT_PRIMARY_H_SUPPORT_LIMIT', severity='HIGH',
            evidence='G2 uses only remaining40% H pseudo blocks, then resamples four-block segments. More resamples do not add historical information.',
            next_test='H1000/1500/3000/5000 at fixed seeds, table saturation and finite-support diagnostics')
        items['B6'].update(status='G2_STRICT_NESTED_CORRECTION_FITTING',
            evidence='G1 I4 Pearson>.995 and delta<.0005 on3/5 folds; no independent signal claim for those folds. G2 has no outer scores released yet.',
            next_test='All five G2 predictions frozen, one score release, paired2000 or predecided5000 draws')
        items['B7'].update(status='ENFORCED',
            evidence='Seven screens reserved/completed;14 full scientific candidates reserved,12 completed;4 learned recipes reserved;5 families reserved;2 full slots remain.',
            next_test='Distinct sixth structural family only if supported by causal failure analysis; otherwise forensic/runtime/reproduction')
        heartbeat = read_json(BASE / 'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 2, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': heartbeat['wall_seconds'], 'productive_seconds': heartbeat['productive_seconds'],
            'priority': ['B5', 'B6', 'B4', 'B3', 'B2', 'B1', 'B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'G2_RP_score_released': False,
            'full_G2_state_null_rates': null_rates, 'full_G2_state_scale_detection': scale_rates,
            'prior_cycle_sha256': digest(BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_001.json'),
            'G2_protocol_sha256': digest(BASE / 'G2_PROTOCOL_LOCK.json'),
            'screen64_is_subset_of_full512_not_extra_independent_evidence': True}
        assert not (BASE / 'G2_OUTER_SCORE_RELEASE.json').exists()
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_002.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = BASE / 'BOTTLENECK_HISTORY_NEXT7.csv'
        with history.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        rows += [{'cycle': 2, 'created_utc': result['created_utc'], 'id': r['id'], 'status': r['status'],
            'severity': r['severity'], 'evidence': r['evidence'], 'next_test': r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', rows, immutable=False)
        usage = read_json(BASE / 'runtime/CANDIDATE_USAGE.json')
        usage.update(screen_completed=7, G2_screen_completed=True, updated_utc=utc_now())
        save('runtime/CANDIDATE_USAGE.json', usage, False)
        save('runtime/CURRENT_RESEARCH_STATUS.json', {'best': 'I4_EVENT_SCALE_TREE',
            'best_mean_delta': -.0003624879784099866, 'bottleneck': 'B5_H_REFERENCE_SUPPORT',
            'next_task': 'G2 nested result, H transport and exact paired uncertainty',
            'screen_count': 7, 'full_count': 12, 'created_utc': utc_now()}, False)
    note('Bottleneck cycle002 complete', 'Null growth improved; test information and detection costs', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

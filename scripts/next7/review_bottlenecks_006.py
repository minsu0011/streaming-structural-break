"""Review finite support, first complete classifier cohort, and reproducibility."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, csv_write, digest, utc_now, activity, note


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    previous_path = BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_005.json'
    items = {row['id']: row for row in read_json(previous_path)['items']}
    with activity('bottleneck cycle006: empirical support and calibrated classifier false alarms',
                  'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        classifier = [row for row in rows(BASE / 'classifier_statistics/N0/PAIRED_CONTRASTS.csv')
                      if row['horizon'] == '16384' and row['metric'] == 'any_alarm']
        own = {row['candidate']: row for row in classifier if row['reference'] == 'M0'}
        assert float(own['AMPLITUDE_0_9_M0']['mean_delta']) == 0
        for method in ('M_EVENT', 'G2_EVENT', 'I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL'):
            assert float(own[method]['CI_low']) > 0
        support = rows(BASE / 'BOOTSTRAP_EMPIRICAL_SUPPORT_NEXT7.csv')
        saturation = []
        for background in ('N0', 'N1', 'N2', 'N3'):
            for h_length in (1000, 2048, 5000):
                cohort = [row for row in support if row['case'] == background and int(row['H_length']) == h_length]
                assert len(cohort) == 512
                n_equal = sum(float(row['envelope_minus_final_cut']) == 0 for row in cohort)
                if h_length < 5000:
                    assert n_equal == 512
                saturation.append({'case': background, 'H_length': h_length,
                                   'terminal_cut_equals_exact_support_envelope': n_equal, 'histories': 512})
        partition = []
        for h_length in (1000, 2048):
            partition += [row for row in rows(BASE / f'support_partition/N0/H_{h_length}/SUMMARY.csv')
                          if row['horizon'] == '16384']
        gradient = rows(BASE / 'g3_full_gradient/fold_0/OBJECTIVE_SUMMARY.csv')
        items['B2'].update(status='N0_COMPLETE_OWN_CALIBRATED_CLASSIFIER_FALSE_ALARMS_EXCEED_M0',
            evidence='N0,16384: M0 any-alarm=.240625; I4=.30390625; G2_FULL=.315625; G3=.3265625. Every five NEXT7 classifier candidate has positive paired pointwise interval for false-alarm increase versus M0. Same own-calibrated0.9M0 is exactly identical to M0; holding the unscaled M0 cutoff gives zero alarms and is an amplitude artifact.',
            next_test='Complete the remaining22 locked classifier cohorts and their paired uncertainty. No cutoff or recipe changes.',
            confidence='512 shared seeds and five fixed fitted folds; intervals conditional on fitted models and cutoffs, not independent-fold inference.')
        items['B4'].update(status='FINITE_SUPPORT_EXCEEDANCE_EXPLAINS_MOST_N0_SHORT_H_LONG_FALSE_MEMORY',
            evidence='N0,H1000/H2048 at16384: empirical-support exceedance accounts for93.56%/78.05% of G2 latches; within-support-above-final-cut component is exactly zero. Earlier retained-only crossings account for the remainder. Every cell and existing bootstrap draw satisfies the exact partition.',
            next_test='Complete the remaining10 crossed cohorts; distinguish fixed support failures from early-boundary retention and unsaturated finite-simulation cuts.')
        items['B5'].update(status='EXACT_BOOTSTRAP_SUPPORT_ENUMERATED_ALL6144_HISTORIES',
            evidence='At16384 the original terminal cutoff equals the exact empirical-resampler support envelope for all512 histories at H1000 and H2048 in each of four backgrounds. Increasing bootstrap replication or path length under the same support/order-statistic rule cannot move the cutoff above that envelope. This is a scoped support obstruction, not a universal impossibility theorem.',
            next_test='Complete H5000 and non-N0 crossed risk; retain H-specific risk distribution and approximate same-seed-pair uncertainty.')
        items['B6'].update(status='G3_FOLD0_SAMPLE_OPTIMIZER_CONVERGED_EXACT_GRADIENT_ERROR_MEASURED',
            evidence='All1,347,881,453 eligible fold0 training pairs evaluated at frozen coefficients. G3 sampled gradient norm6.38e-7 versus exact norm.00234947; exact regularized objective improves over G2 parent by only.0000147238. The full-gradient result measures sample approximation, without identifying the cause of held-out AUC loss. Other four folds remain queued.',
            next_test='Complete full-gradient diagnostics for all five folds; execute the frozen quiet scalar latency workload during admitted final audit.')
        items['B7'].update(status='RAW_FREE_STREAM_REPRODUCTION_AND_OWNED_CRASH_RECOVERY_PASS',
            evidence='New isolated payload process reproduced92,160 scalar outputs across90 paths bitwise while original project reads were denied. Real owned process interruption verified one retry, checkpoint reuse, and quarantine after a second failure. All6,333 protected prior files rehashed unchanged. Calibration BOM and missing reproduction configs repaired with original failure evidence preserved.',
            next_test='Continue productive research to36000 verified seconds, finish all queued science diagnostics, then final audit, actual ZIP extraction reproduction, and final handoff. No additional model search.')
        heartbeat = read_json(BASE / 'runtime/HEARTBEAT.json')
        evidence = ('BOOTSTRAP_SUPPORT_DONE.json', 'classifier_statistics/N0/DONE.json',
            'g3_full_gradient/fold_0/DONE.json', 'reproduction/research_002/DONE.json',
            'audit/OWNED_CRASH_RECOVERY_TESTS_001.json', 'audit/PROTECTED_AUTHORITY_RESEARCH_001.json',
            'CLASSIFIER_CALIBRATION_REPAIR_CLOSED.json', 'visuals/v3/BROWSER_REVIEW.json',
            'SCORE_EVIDENCE_CONSOLIDATED.json')
        result = {'created_utc': utc_now(), 'cycle': 6, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': heartbeat['wall_seconds'], 'productive_seconds': heartbeat['productive_seconds'],
            'priority': ['B5', 'B4', 'B2', 'B6', 'B3', 'B1', 'B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': not heartbeat['research_minimum_met'],
            'prior_cycle_sha256': digest(previous_path), 'N0_long_null_classifier_paired_results': classifier,
            'exact_support_saturation': saturation, 'N0_short_H_support_partition': partition,
            'G3_fold0_training_gradient_results': gradient,
            'evidence_sha256': {name: digest(BASE / name) for name in evidence}}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_006.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = rows(BASE / 'BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle': 6, 'created_utc': result['created_utc'], 'id': row['id'],
                     'status': row['status'], 'severity': row['severity'], 'evidence': row['evidence'],
                     'next_test': row['next_test']} for row in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', history, immutable=False)
        save('audit/VIEWER_V3_VISUAL_REVIEW_001.json', {'status': 'PASS', 'created_utc': utc_now(),
            'reviewed': {name: digest(BASE / 'visuals/v3' / name) for name in
                         ('viewer_desktop_review.png', 'viewer_narrow_review.png')},
            'observations': 'Desktop and narrow screenshots visually checked: Korean labels legible, no overlap or horizontal overflow; current normal, historical memory retained, target1 and classifier current alarm absent clearly distinguished.',
            'automated_browser_receipt_sha256': digest(BASE / 'visuals/v3/BROWSER_REVIEW.json'),
            'font_repair': 'Static raster/SVG literal U+2212 replaced by ASCII hyphen because Malgun Gothic lacks the glyph. Original17 generated files retained under pre_font_repair; no numeric data or inference changes.'})
    note('Bottleneck cycle006 complete', 'Exact finite support obstruction; N0 classifier candidate false alarms remain higher than BL80', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

"""Review all N0 crossed histories and background-specific classifier evidence."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, csv_write, digest, utc_now, activity, note


def rows(name):
    with (BASE / name).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    previous = BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_006.json'
    items = {row['id']: row for row in read_json(previous)['items']}
    with activity('bottleneck cycle007: matched H contrasts and reference transport',
                  'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        contrasts = [r for r in rows('crossed_H_length_comparison/N0/PAIRED_H_LENGTH_CONTRASTS.csv')
                     if r['horizon'] == '16384']
        qualified = [r for r in contrasts if r['metric'] == 'G2_QUALIFIED_EVER_LATCH']
        assert len(qualified) == 3
        assert all(float(r['CI_low_approx_pointwise']) < 0 < float(r['CI_high_approx_pointwise'])
                   for r in qualified)
        risk_order = [r for r in rows('crossed_H_length_comparison/N0/RISK_ORDER_STABILITY.csv')
                      if r['horizon'] == '16384']
        normalizers = []
        for length in (1000, 2048, 5000):
            normalizers += [r for r in rows(f'normalizer_transport/correlations/N0/H_{length}/CORRELATIONS.csv')
                            if r['horizon'] == '16384']
        classifiers = []
        for case in ('N0', 'N1', 'N2'):
            classifiers += [r for r in rows(f'classifier_statistics/{case}/PAIRED_CONTRASTS.csv')
                            if r['horizon'] == '16384' and r['metric'] == 'any_alarm'
                            and r['reference'] == 'M0']
        gradient = []
        for fold in (0, 1):
            gradient += rows(f'g3_full_gradient/fold_{fold}/OBJECTIVE_SUMMARY.csv')
        partition = [r for r in rows('support_partition/N0/H_5000/SUMMARY.csv')
                     if r['horizon'] == '16384']
        items['B2'].update(status='CLASSIFIER_LONG_NULL_FAILURE_DEPENDS_ON_BACKGROUND',
            evidence='N0/N1: every NEXT7 candidate increases own-calibrated long-null alarms versus M0 with positive paired pointwise intervals. N2: M0 itself alarms .67578125; old M3 .50078125 is lower, while I4 .678515625 and G2_FULL/G3 .67109375 have differences whose intervals include zero. A global all-background deterioration claim is unsupported.',
            next_test='Complete N3 and all19 fixed shock/break classifier cohorts without changing calibration or features.',
            confidence='Three complete backgrounds,512 original seeds and five fixed model folds. No future target or background-specific gate enters inference.')
        items['B4'].update(status='LONGER_H_CHANGES_FALSE_MEMORY_COMPONENTS_WITHOUT_RESOLVING_N0_TOTAL_RISK',
            evidence='N0 at16384: G2 crossed risk is .357342/.340984/.332447 for H1000/2048/5000. All three matched rate-difference intervals include zero. H5000 beyond-support=.159874, within-support-above-terminal-cut=.0752335, early-retained-only=.0973396. Within-support exceedance is not automatically Monte Carlo error.',
            next_test='Finish nine remaining crossed cohorts and all fixed paired H-length comparisons. Keep support, terminal-tail and earlier-memory components separate.')
        items['B5'].update(status='H_REFERENCE_TRANSPORT_ASSOCIATION_AND_LOW_RISK_ORDER_STABILITY',
            evidence='All6144 H-only prefix/full normalizers reconstructed. N0 long-risk versus log prefix/full rank-tail spread correlation is positive, reaching .660693 at H5000 (pointwise interval .610624,.706482); signed-log-MAD spread correlations are about .216-.225 across the three H lengths. Matched conditional H-risk order Spearman is only .034-.107 across lengths. All23 covariates are retained, with no causal attribution or H selection.',
            next_test='Complete the same normalizer correlations across all12 locked cohorts. Audit the near-constant rank-second-moment covariate before interpreting its numerical correlations.',
            confidence='Posthoc descriptive H associations; same-seed-pair product bootstrap is approximate and not an independent-factor bootstrap theorem.')
        items['B6'].update(status='FULL_GRADIENT_REPEATS_SMALL_PARENT_OBJECTIVE_GAIN_IN_TWO_FOLDS',
            evidence='Fold1 evaluates1,346,531,864 exact training pairs: G3 sampled gradient1.93e-7, exact gradient .002312968. Exact penalized objective improves over G2 by only .0000116777; fold0 improvement .0000147238. These are training approximation diagnostics and do not bound held-out AUC. Remaining three folds are queued.',
            next_test='Complete the remaining full-gradient diagnostics; preserve frozen weights. Run measured scalar runtime only after admitted final audit.')
        items['B1'].update(status='BOUNDED_RULE_EVENT_EQUIVALENCE_PROVEN_IN_APPLICABLE_H_BINS',
            evidence='For four values bounded in[-4,4], exactly two positive values cannot make the signed mean exceed2. Therefore S1 C1 and C2 confirmation events coincide when their shared strict cutoff is at least2 (H1000/1500/2048). All23 original H2048 case summaries agree across18 event fields. Subthreshold current scores can differ; no full-prediction equivalence is claimed.',
            next_test='Retain all originally tested rules and the analytic scope. Do not spend a new recipe on algebraically duplicate event tests.')
        items['B7'].update(status='FINAL_AUDIT_DRIVERS_PREPARED_AND_FAILURE_REGISTRY_RESOLVED',
            evidence='Read-only failure registry covers seven software failures and the repaired BOM queue quarantine, unresolved0. Final regression, causal prefix/order/serialization audit, and RP boundary audit drivers are prepared but not executed. No final audit time or PASS status is claimed before admission.',
            next_test='Finish productive research and all queued diagnostics; assemble explicit allowlisted handoff, then run final audit and extracted-ZIP reproduction.')
        evidence = ['crossed_H_length_comparison/N0/DONE.json', 'NORMALIZER_H_PARAMETERS_DONE.json',
                    'CONFIRMATION_RULE_EQUIVALENCE_DONE.json', 'CONFIRMATION_RULE_REPORT_DONE.json',
                    'COMPLETED_CORE_TABLES_NEXT7.json', 'audit/failure_registry/research_001/DONE.json']
        evidence += [f'normalizer_transport/correlations/N0/H_{length}/DONE.json' for length in (1000,2048,5000)]
        evidence += [f'classifier_statistics/{case}/DONE.json' for case in ('N0','N1','N2')]
        evidence += [f'g3_full_gradient/fold_{fold}/DONE.json' for fold in (0,1)]
        heartbeat = read_json(BASE / 'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 7, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': heartbeat['wall_seconds'], 'productive_seconds': heartbeat['productive_seconds'],
            'priority': ['B5','B4','B2','B6','B1','B3','B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': not heartbeat['research_minimum_met'],
            'prior_cycle_sha256': digest(previous), 'N0_matched_H_length_long_results': contrasts,
            'N0_conditional_H_risk_order': risk_order, 'N0_H5000_support_partition': partition,
            'N0_all_H_long_normalizer_associations': normalizers,
            'three_background_classifier_results': classifiers,
            'two_fold_full_gradient_results': gradient,
            'evidence_sha256': {name: digest(BASE / name) for name in evidence}}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_007.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = rows('BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle': 7, 'created_utc': result['created_utc'], 'id': row['id'],
                     'status': row['status'], 'severity': row['severity'], 'evidence': row['evidence'],
                     'next_test': row['next_test']} for row in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', history, immutable=False)
    note('Bottleneck cycle007 complete', 'Conditional H transport and background-specific null failure',
         'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

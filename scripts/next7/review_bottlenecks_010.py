"""Review complete four-case scale alarms and the next conditional null cohort."""
from pathlib import Path
import csv
import datetime as dt
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note


def rows(name):
    with (BASE/name).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    previous = BASE/'bottlenecks/NEXT7_BOTTLENECK_CYCLE_009.json'
    prior = read_json(previous)
    elapsed = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(prior['created_utc'])).total_seconds()
    assert elapsed >= 45*60, 'Keep the requested periodic cadence; this is not a padding task'
    items = {r['id']: r for r in prior['items']}
    with activity('Bottleneck cycle010: fixed scale classifier power and conditional AR risk',
                  'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        items['B1'].update(status='STRONG_CONTRACTION_CLASSIFIER_GAIN_WEAK_CONTRACTION_STILL_LIMITED',
            evidence='All four original variance-change classifier cohorts are complete. At1024, V4 G3 correct first alarm=.23203125 versus M0=.1109375: delta+.12109375, paired95% interval[+.100771484375,+.142578125]. V3 G3=.023828125 versus M0=.0203125: delta+.003515625, interval[-.0015625,+.009375]. At4096, V3 G3 gains+.0328125 with interval[+.020703125,+.045712890625]. These classifier endpoints differ from the previously audited state-memory failure partitions. V4 is variance0.5, not standard deviation0.5.',
            confidence='Complete original512 seeds and five fixed model folds; paired intervals are pointwise, with no background-specific threshold changes.',
            next_test='Finish other classifier mechanisms and retain the overall RP and long-null gates. Do not turn V4-only power into a new robustness role.')
        items['B2'].update(confidence='All1536 original shock/control state pairs have exact local/current equality checks; final classifier shock comparisons are still separate pending work.',
            next_test='Complete the original classifier shock, pulse and recovery cases. Compare persistent state memory with own-calibrated classifier alarms and delays without changing thresholds.')
        items['B3'].update(status='LOCAL_SCALE_ALARM_GAIN_DOES_NOT_REVERSE_RP_RANKING_FAILURE',
            evidence='V4 classifier power improves under the frozen own cutoffs, but all15 NEXT7 candidates retain negative mean matched RP TS-AUC and negative worst-fold delta. Current recovery, preserved event memory and final classifier alarms remain separately displayed in the reviewed v4 viewer.',
            next_test='Complete the remaining fixed comparisons and final nested lineage audit. No candidate refitting or selection from the local synthetic improvements.')
        items['B4'].update(status='AR_H2048_LONG_FALSE_MEMORY_REMAINS_ABOVE_A_QUARTER',
            evidence='N1,H2048,16384 crossed G2 memory=.2753181457519531, interval[.24744749069213867,.3036998748779297]. Beyond empirical support=.210205078125,76.3499% of G2 latches; earlier-retained-only=.06511306762695312; within-support above terminal cut=0. Across the512 H references, median conditional risk=.1689453125 and19.7266% of H have risk above.5. H-main descriptive mean-square fraction=.3911344915. These are complete fixed-cohort diagnostics, not population/anytime guarantees.',
            next_test='Finish N1H5000 and all six N2/N3 crossed cohorts. Preserve the exact support partitions and original matched H-length contrasts.')
        items['B5'].update(status='AR_NORMALIZER_ASSOCIATION_AND_FINITE_RESAMPLING_AUDIT_SEPARATED',
            evidence='N1 long-risk association with log(prefix/full tail spread) is Pearson.44664899 atH1000 and.47287607 atH2048, with respective pointwise intervals[.38266332,.50810208] and[.39032688,.54204685]. These are posthoc associations, not causal features. A new algebra-only audit of exact finite-multinomial moments is registered for all12 completed grids. Eight tests enumerate small sample spaces exactly; no original draws or intervals are changed.',
            next_test='Complete all H-length and normalizer comparisons, then run the registered108-row exact moment audit. Keep arithmetic exactness distinct from population interval coverage.')
        items['B6'].update(status='EXACT_TRAINING_OBJECTIVE_DIAGNOSTICS_COMPLETE_IN_ALL_FIVE_FOLDS',
            evidence='All five full-pair diagnostics and their consolidated tables are complete. G3 improves the exact penalized training objective relative to its G2 parent in every fold, without refitting. Fold4 evaluates1,379,709,289 training pairs: G3=.6417935995896087 versus G2=.6418170417701139, an improvement of.0000234421805052. G3 sampled-gradient L2=3.65068e-7, exact-gradient L2=.0026849237438. Improved surrogate values do not bound or rescue held-out TS-AUC, and the sampled fit is not claimed to minimize the exact objective.',
            confidence='Exact enumeration for all five frozen fold/head pairs, with original sampled fits unchanged.',
            next_test='Preserve the completed exact objective/gradient evidence. Measure scalar runtime during the quiet admitted final audit; do not optimize another head from this diagnosis.')
        items['B7'].update(status='PUBLIC_REFERENCE_PORTABILITY_FIXED_FINAL_CLOSURE_STILL_GATED',
            evidence='Required BL80/J102 model.json and D2 RESULTS.json dependencies were added to the planned ZIP before export. A fresh isolated public-entry replay checks9 paths and9216 scalar outputs bitwise, with zero original-workspace read attempts. Both four-case classifier scale figures have passed visual review. Final closure now requires all114 queued tasks, complete-cohort figures, the exact moment audit and actual productive time. No final audit or delivery is claimed.',
            next_test='Finish complete-cohort research and required productive time, then admit final audit, run quiet scalar benchmarks/full checks, and verify both replay entry points from the final ZIP before copying20 desktop files.')
        paths = ['conditional_statistics/N1/H_2048/DONE.json', 'support_partition/N1/H_2048/DONE.json',
            'normalizer_transport/correlations/N1/H_1000/DONE.json',
            'normalizer_transport/correlations/N1/H_2048/DONE.json',
            'g3_full_gradient/fold_3/DONE.json', 'g3_full_gradient/fold_4/DONE.json',
            'COMPLETE_DIAGNOSTICS_GRADIENTS_NEXT7.json', 'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json',
            'audit/FROZEN_REFERENCE_ISOLATED_REPLAY_001.json', 'CROSSED_MOMENT_AUDIT_LOCK_NEXT7.json',
            'visuals/classifier_scale/original_cases_002/DONE.json',
            'audit/CLASSIFIER_SCALE_VISUAL_REVIEW_002.json']
        paths += [f'classifier_statistics/{case}/DONE.json' for case in ('V1','V2','V3','V4')]
        h = read_json(BASE/'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 10, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': h['wall_seconds'], 'productive_seconds': h['productive_seconds'],
            'priority': ['B5','B4','B1','B2','B6','B3','B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': not h['research_minimum_met'],
            'prior_cycle_sha256': digest(previous), 'evidence_sha256': {name:digest(BASE/name) for name in paths},
            'no_new_candidate_fit_or_threshold_or_random_draw': True}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_010.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = rows('BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle':10, 'created_utc':result['created_utc'], 'id':r['id'], 'status':r['status'],
            'severity':r['severity'], 'evidence':r['evidence'], 'next_test':r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', history, immutable=False)
    note('Bottleneck cycle010 complete', 'V4 power gain, V3 limits, AR null transport and portable reference replay; no promotion', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

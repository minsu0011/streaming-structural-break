"""Review exact scale failure partitions, finite shock influence and viewer clarity."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note


def rows(name):
    with (BASE / name).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    previous = BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_008.json'
    items = {r['id']: r for r in read_json(previous)['items']}
    with activity('Bottleneck cycle009: scale failure location and finite shock influence',
                  'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        scale = [r for r in rows('SCALE_FIRST_EVENT_FAILURE_PARTITION_NEXT7.csv') if r['case'] in ('V3', 'V4')]
        shocks = rows('SHOCK_PERSISTENT_MEMORY_DIFFERENCES_NEXT7.csv')
        n1 = [r for r in rows('conditional_statistics/N1/H_1000/SUMMARY.csv') if r['horizon'] == '16384']
        support = [r for r in rows('support_partition/N1/H_1000/SUMMARY.csv') if r['horizon'] == '16384']
        rp = [r for r in rows('RP_SCALE_ADMISSION_PARTITION_NEXT7.csv') if r['group'] == 'ALL_RP']
        gradients = rows('g3_full_gradient/fold_2/OBJECTIVE_SUMMARY.csv')
        items['B1'].update(status='SCALE_MISSES_MOSTLY_PRECEDE_RAW_CONFIRMATION_AT1024',
            evidence='Original V3: correct G2 first detection27/512, prebreak memory6, postbreak raw confirmation rejected14, no postbreak raw confirmation465. V4:180,12,44,276. Fully postbreak V4 scale windows are80.28% negative-coherent but below raw confirmation strength; only3.15% confirm negative scale. A hard shared-cut ceiling exists but intersects only11 of332 V4 first misses by1024. RP scale confirmations partition1223 pass/596 subceiling reject/456 hard-ceiling reject;322 rejected scale blocks still qualify through core. Counts are dependent block events, not detection rates.',
            next_test='Complete the remaining fixed classifier scale cohorts and compare their own-calibrated alarms/delays. Preserve all thresholds and the no-promotion decision; do not mistake sign agreement or local gains for whole-RP benefit.')
        items['B2'].update(status='ALL1536_SHOCK_PAIRS_HAVE_FINITE_LOCAL_CURRENT_INFLUENCE_BUT_MEMORY_CAN_PERSIST',
            evidence='All1536 original paired paths satisfy prefix equality, the lag4/four-block local equality bound and two additional recovery-block current-state equality. At4096, G2 intervention-only latch rates: O1 .00390625, OB16 .013671875, repeated .0078125. Repeated control-only latch=.001953125, so one-way excess and net risk difference differ. The broader any-history-coordinate difference includes shock timestamps and is not an event-latch rate.',
            next_test='Complete the original classifier shock/pulse/recovery cases under frozen own cutoffs. Keep current evidence equality, event-memory equality and final classifier alarms as distinct endpoints.')
        items['B3'].update(status='MECHANICAL_TARGET_PERSISTENCE_DOES_NOT_REPAIR_OVERALL_RANK_TRADEOFF',
            evidence='All15 fixed NEXT7 candidates still have negative mean matched RP delta and negative worst-fold delta. Exact scale evidence and finite shock-memory diagnostics locate missing or harmful information without modifying candidate scores. The v4 viewer now separates raw scale confirmation, shared H admission, core fallback, fresh qualification and retained memory on the same eight examples.',
            next_test='Finish all fixed synthetic classifier mechanisms and final nested lineage checks. No new recipe or score-driven reweighting is authorized by a local visual explanation.')
        items['B4'].update(status='SECOND_BACKGROUND_CONFIRMS_SHORT_H_EMPIRICAL_SUPPORT_OBSTRUCTION',
            evidence='N1,H1000,16384 crossed G2 risk=.3887519836425781 CI[.35747528076171875,.4228507041931152]. Beyond empirical support=.3669395446777344,94.389% of all G2 latches; within-support above terminal cutoff=0; earlier-retained-only=.02181243896484375. This repeats the short-H support problem in a second background. N1H2048/H5000 remain unfinished; no all-H generalization.',
            next_test='Complete all remaining crossed null cohorts and registered matched H-length comparisons, retaining every background and exact support decomposition.')
        items['B5'].update(status='H_TRANSPORT_AND_SCALE_SHARED_CUT_OBSTRUCTIONS_DISTINGUISHED',
            evidence='H transport, empirical support and scale-channel bounded feasibility are separate mechanisms. Original6144 null H grids and23 actual synthetic H cohorts were audited without new fitting. All3,203,157 RP cached cut/support/fresh decisions match the original H calibrations. RP caches have original manifest hashes; synthetic H files lacked original DONE hashes, so this audit froze their bytes and cross-checked all final cuts against originally hashed SERIES files.',
            next_test='Finish the remaining normalizer associations and H-length contrasts. Interpret near-constant rank moments through the proven tie/rounding geometry; do not convert posthoc correlations into an inference gate.')
        items['B6'].update(status='EXACT_GRADIENT_DIAGNOSIS_COMPLETE_IN_THREE_FOLDS',
            evidence='Fold2 evaluates1,353,424,464 exact training pairs. G3 exact objective=.6399141599540578 versus G2=.6399191896956316, a .00000502974 training-objective improvement. G3 sampled gradient L2=4.42e-7 but exact gradient=.00283626244. This is approximation evidence, not a bound on held-out AUC. Original fold0/fold1 findings remain; folds3/4 are pending.',
            next_test='Complete both remaining exact-gradient diagnostics without refitting. Measure scalar runtime only in the quiet admitted final audit.')
        items['B7'].update(status='REVIEWED_FINAL_VIEWER_AND_DELIVERY_GATES_READY_RESEARCH_STILL_ACTIVE',
            evidence='V4 explanation flags use the original16 exact paired synthetic state arrays and preserve every original classifier curve/boolean alarm. Four prefix/equality/core-fallback tests and64 scenario/classifier browser combinations pass, with extra boundary frames and full desktop/narrow visual review. Three handoff-plan amendments include the new completed diagnostics. Productive research minimum and final audit remain pending; no final delivery is claimed.',
            next_test='Complete remaining science and productive time, render the final complete-cohort figures, then execute the frozen audit/quiet scalar runtime and exact ZIP extraction/replay with20 copied desktop outputs.')
        names = ['SCALE_ADMISSION_CEILING_DONE_NEXT7.json', 'OBSERVED_SCALE_ADMISSION_DONE_NEXT7.json',
            'SCALE_EVIDENCE_PATH_AUDIT_DONE_NEXT7.json', 'SHOCK_INFLUENCE_AUDIT_DONE.json',
            'conditional_statistics/N1/H_1000/DONE.json', 'support_partition/N1/H_1000/DONE.json',
            'normalizer_transport/correlations/N1/H_1000/DONE.json', 'g3_full_gradient/fold_2/DONE.json',
            'visuals/scale_evidence/original_cases_001/DONE.json', 'audit/SCALE_EVIDENCE_VISUAL_REVIEW_001.json',
            'visuals/v4/BUILD_DONE.json', 'visuals/v4/BROWSER_REVIEW.json', 'audit/VIEWER_V4_VISUAL_REVIEW_001.json',
            'audit/SCALE_EVIDENCE_TEST_DEVELOPMENT_REPAIR_001.json']
        heartbeat = read_json(BASE / 'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 9, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': heartbeat['wall_seconds'], 'productive_seconds': heartbeat['productive_seconds'],
            'priority': ['B5', 'B4', 'B1', 'B2', 'B6', 'B3', 'B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': not heartbeat['research_minimum_met'],
            'prior_cycle_sha256': digest(previous), 'scale_first_event_partitions': scale,
            'shock_finite_influence_memory_results': shocks, 'N1_H1000_long_crossed_results': n1,
            'N1_H1000_long_support_results': support, 'RP_scale_admission_census': rp,
            'fold2_full_gradient': gradients,
            'board_scope_clarification': 'B1 keeps its canonical scale-contraction name; B2 keeps its shock-memory name. Earlier cross-cutting evidence is preserved in its original cycles.',
            'evidence_sha256': {name: digest(BASE / name) for name in names}}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_009.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = rows('BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle': 9, 'created_utc': result['created_utc'], 'id': r['id'], 'status': r['status'],
                     'severity': r['severity'], 'evidence': r['evidence'], 'next_test': r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', history, immutable=False)
    note('Bottleneck cycle009 complete', 'Exact scale misses, finite shock influence and short-H null support; no promotion', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()

"""Review completed ARCH contrasts and separate paired shock alarm endpoints."""
from pathlib import Path
import csv
import datetime as dt
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,activity,note


def main():
    previous=BASE/'bottlenecks/NEXT7_BOTTLENECK_CYCLE_012.json'
    prior=read_json(previous)
    elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(prior['created_utc'])).total_seconds()
    assert elapsed>=45*60
    items={r['id']:r for r in prior['items']}
    with activity('Bottleneck cycle013: completed ARCH contrasts and joint shock endpoints',
                  'BOTTLENECK_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        items['B1'].update(status='SHORT_TRUE_CHANGE_PULSE_REMAINS_LOW_POWER',
            evidence='TEMP_SCALE_PULSE uses32 observations multiplied by sqrt2 and then returns to its original process, while the target remains1 after the first change. Correct first detection by1024 is M0=.076171875 versus G3=.07109375, paired delta-.005078125 with interval[-.011328125,+.000390625]. By4096 M0=.28125 and G3=.276171875, delta-.005078125 with interval[-.016015625,+.00625]. Fewer alarms in this true-change condition are not a robustness improvement. Local V4, ARCH and directional tail gains do not establish overall ranking or transient-change improvement.',
            next_test='Complete RETURN_NORMAL, LATE_PERSISTENT and SCALE_PLUS_SHOCK with the already frozen models, cuts and original seeds. Keep the permanent target separate from current-state recovery.')
        items['B2'].update(status='ONE_WAY_SHOCK_METRIC_CAN_FALL_WHILE_TREATMENT_AND_CONTROL_ALARMS_RISE',
            evidence='For repeated shocks by4096, M0 treatment any=.98671875, control any=.090625 and treatment-only=.89609375. G3 treatment any=.990625, control any=.112890625 and treatment-only=.877734375. Paired G3 deltas are treatment any+.00390625 with interval[+.001171875,+.007041015625], control any+.022265625 with interval[+.0125,+.0328125], and treatment-only-.018359375 with interval[-.028916015625,-.0078125]. Control-only is0 for both, so both-alarm probability grows more than treatment-any probability. This is not classifier robustness improvement. The frozen joint audit reconstructs48 joint rows,42 comparisons and114 original mean identities; six historical M3 joint rows remain unmeasured.',
            confidence='All original512 seeds, fixed five folds and2000 paired draws retained. The joint decomposition adds algebra, not new uncertainty draws or feature-level causal attribution.',
            next_test='Preserve treatment-any, control-any, treatment-only and missing fields together in the final guide. No endpoint switching or gate relaxation; retain the O1 zero-memory-but-alarming example and OB16 worsening.')
        items['B4'].update(status='ARCH_H_LENGTH_CONTRAST_NONMONOTONIC_POINT_ESTIMATE_NOT_SIGNIFICANT',
            evidence='N2 long qualified false-memory risk for H1000/H2048/H5000 is.39958953857421875/.2843475341796875/.3023796081542969. H5000 minus H2048 is+.018032073974609375 with original matched interval[-.01250476837158203,+.0488873481750488], which crosses0. H2048 minus H1000=-.11524200439453125 with interval[-.1541903495788574,-.07702522277832036]. H5000 minus H1000=-.09720993041992188 with interval[-.13746204376220703,-.056477642059326205]. The positive N1 H5000-minus-H2048 interval does not generalize to ARCH. H selection and a causal sample-size effect are not licensed.',
            next_test='Complete all three N3 crossed cohorts and their registered postprocessors before final all-background figures; retain all original H comparisons.')
        items['B5'].update(status='ARCH_H5000_HAS_WITHIN_SUPPORT_AND_EARLY_RETAINED_RISK_TOO',
            evidence='N2,H5000 long qualified risk=.3023796081542969 partitions into beyond-support=.0922698974609375, within-support above terminal cut=.13340377807617188 and earlier-retained-only=.0767059326171875. Their shares are30.51458993%,44.11798100%,25.36742907%. Thus a finite-support ceiling explains only part of this cohort. H5000 rank-tail spread mismatch association r=.3139068570305402, interval[.2196324652072147,.40423624886420184]; signed-log-MAD spread mismatch r=.5321950055519867, interval[.45021294194155104,.6034032694555544]. These posthoc associations do not establish causal contributions.',
            next_test='Finish N3 and the registered108-row exact finite-multinomial audit. Retain support and normalizer mechanisms separately, without new draws, calibration or selected thresholds.')
        items['B7'].update(status='FOUR_WORKERS_STABLE_FINAL_AUDIT_NOT_YET_ADMITTED',
            evidence='Four owned workers continue to advance CPU time and required output; the latest heartbeat has no accounting warnings or hang suspicion and RAM about62%. Nine of12 crossed cohorts and20 of23 classifier cases are complete. The joint-shock audit, final-guide endpoint tables and closure bindings are complete. All15 new candidates still have negative mean and worst-fold RP deltas, and inherited core raw-lag overlap still violates strict nonoverlap. Productive research is approaching but has not yet reached36000 seconds; final audit and desktop delivery are not claimed.',
            next_test='Finish all115 tasks and productive minimum, review all-cohort plots, admit the final audit, execute quiet runtime measurement, then verify both extracted replay paths and exactly20 desktop copies.')
        paths=['conditional_statistics/N2/H_5000/DONE.json','support_partition/N2/H_5000/DONE.json',
            'normalizer_transport/correlations/N2/H_5000/DONE.json','crossed_H_length_comparison/N2/DONE.json',
            'classifier_statistics/REPEATED_SHOCK/DONE.json','classifier_statistics/TEMP_SCALE_PULSE/DONE.json',
            'SHOCK_CLASSIFIER_JOINT_AUDIT_LOCK_NEXT7.json','SHOCK_CLASSIFIER_JOINT_AUDIT_DONE_NEXT7.json',
            'SHOCK_CLASSIFIER_JOINT_PROBABILITIES_NEXT7.csv','SHOCK_CLASSIFIER_ENDPOINT_DECOMPOSITION_NEXT7.csv',
            'research_notes/PAIRED_CONTROL_ALARM_ENDPOINT_INTERPRETATION_KO.md',
            'FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json']
        h=read_json(BASE/'runtime/HEARTBEAT.json')
        result={'created_utc':utc_now(),'cycle':13,'status':'RESEARCH_CONTINUES',
            'wall_seconds':h['wall_seconds'],'productive_seconds':h['productive_seconds'],
            'priority':['B2','B5','B4','B3','B1','B6','B7'],'items':list(items.values()),
            'current_primary':'FROZEN_BL80','learned_candidate_result':'NO_PROMOTION',
            'research_duration_requirement_not_yet_met':not h['research_minimum_met'],
            'prior_cycle_sha256':digest(previous),'evidence_sha256':{name:digest(BASE/name) for name in paths},
            'no_new_candidate_fit_or_threshold_or_random_draw':True}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_013.json',result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json',result,False)
        with (BASE/'BOTTLENECK_HISTORY_NEXT7.csv').open(newline='',encoding='utf-8-sig') as handle:
            history=list(csv.DictReader(handle))
        history += [{'cycle':13,'created_utc':result['created_utc'],'id':r['id'],'status':r['status'],
            'severity':r['severity'],'evidence':r['evidence'],'next_test':r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv',history,immutable=False)
    note('Bottleneck cycle013 complete','ARCH matched contrasts and shock endpoint decomposition; no promotion','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    main()

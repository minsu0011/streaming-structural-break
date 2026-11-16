"""Review completed classifier shock failures and the ARCH conditional cohort."""
from pathlib import Path
import csv
import datetime as dt
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,activity,note


def main():
    previous=BASE/'bottlenecks/NEXT7_BOTTLENECK_CYCLE_011.json'
    prior=read_json(previous)
    elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(prior['created_utc'])).total_seconds()
    assert elapsed>=45*60
    items={r['id']:r for r in prior['items']}
    with activity('Bottleneck cycle012: classifier shock boundary and ARCH null transport',
                  'BOTTLENECK_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        items['B1'].update(status='LOCAL_POWER_GAINS_RETAIN_STRONG_DIRECTIONAL_ASYMMETRY',
            evidence='Completed T1 Gaussian-to-t3 and T2 t3-to-Gaussian classifier cases use variance factor1 and fixed AR.3. G3 correct first detection by1024 is.86484375 inT1 versus M0.740625, delta+.12421875 with interval[+.107421875,+.14296875]. InT2 G3=.1390625 versus M0=.064453125, delta+.074609375 with interval[+.059765625,+.090625]. Gains in both directions do not make their absolute detection rates symmetric. MAD scale and variance are distinct quantities; tail-shape effects must not be called variance-contraction repairs. Prior V3/V4 evidence remains unchanged.',
            next_test='Finish repeated shock, temporary scale pulse, recovery, late persistence and combined scale/shock classifier cases. Retain the full case table and local-power versus global-ranking distinction.')
        items['B2'].update(status='CONFIRMATION_MEMORY_DOES_NOT_PROTECT_FINAL_CLASSIFIER_FROM_SHOCK_ALARMS',
            evidence='Complete O1 at1024: M0 any alarm=.79140625, G3=.782421875, paired delta-.008984375 with interval[-.018359375,+.000390625]. Treatment alarm/control never rates are.767578125 and.759375. Prior M3 any alarm=.665625; its historical paired-control endpoint remains missing. Complete OB16: M0 any=.8890625, G3=.905859375, delta+.016796875 with interval[+.010546875,+.0234375]. Thus classifier burst-shock risk worsens despite protected event-memory updates. In the already fixed O1 seed0 viewer, raw and G2 memory remain exactly zero throughout1024, while BL80 alarms at17 points starting129 and G3 at12 points starting193; controls do not alarm. All8 exact alarm arrays match the reviewed viewer.',
            confidence='Complete original512-seed cohorts, five fixed folds and original2000 paired draws. The fixed example proves separation of contracts, not a whole-cohort feature-level causal attribution.',
            next_test='Complete repeated and combined interventions. Preserve separate state-memory and classifier-alarm tables, explicit missing M3 fields and the original gate feasibility audit. Do not infer classifier robustness from memory protection.')
        items['B3'].update(status='RAW_NONOVERLAP_REQUIREMENT_UNMET_AND_RP_PROMOTION_STILL_FAILS',
            evidence='All15 NEXT7 candidates still have negative mean and worst-fold matched RP deltas. The inherited core lag4 branch still reuses up to four raw observations across feature blocks; the new scale raw16 blocks are disjoint. This unmet requirement is explicit in closure and final audit. The next-session plan describes an unevaluated within-block lag1/lag4 design with15/12 pairs and the need to recalibrate changed statistics; no replacement model is implemented or claimed.',
            next_test='Keep NO_PROMOTION and complete the nested lineage audit. Preserve the distinction between a reviewed negative-research handoff and a production model satisfying every design requirement.')
        items['B4'].update(status='ARCH_LONG_QUALIFIED_FALSE_MEMORY_REMAINS_HIGH',
            evidence='N2,H1000,16384 crossed G2 false memory=.39958953857421875, interval[.36604509353637693,.4340644836425781]. N2,H2048 is.2843475341796875, interval[.25759401321411135,.31098232269287107]. AtH2048 the median H-conditional risk=.1953125 and21.09375% of H references have risk above.5; the H-main descriptive variance fraction=.373485497230501. The remaining H5000 and N3 cohorts are still required; no new H selection or conditional5% guarantee follows.',
            next_test='Finish the remaining four crossed cohorts and all original paired H-length contrasts before final all-cohort figures and closure.')
        items['B5'].update(status='ARCH_SUPPORT_AND_NORMALIZER_MISMATCH_REMAIN_DISTINCT',
            evidence='N2,H2048 long G2 risk partitions exactly into beyond empirical support.214080810546875, within-support above terminal cut0, and earlier-retained-only.0702667236328125. Beyond-support accounts for75.2884357392% of latches, with interval[71.4603315334%,78.5441606442%]. AtH1000 the beyond-support share is94.7198090692%. H2048 rank-tail spread mismatch has Pearson association.4681651312176267, interval[.3927440431791053,.5364133385659169]; signed-log-MAD spread mismatch r=.3313388308902635, interval[.24430584065652824,.4103751773488186]. These posthoc associations do not allocate causal effects.',
            next_test='Complete every background and the registered108-row exact finite-multinomial audit. More draws from the same finite support cannot remove the demonstrated support ceiling.')
        items['B6'].update(next_test='All five original full-pair objective diagnostics are complete. Keep the exact-versus-sampled gradient evidence and negative held-out deltas; run the quiet scalar benchmark after research closure.')
        items['B7'].update(status='FOUR_OWNED_WORKERS_AND_CONCRETE_SHOCK_HANDOFF_REVIEW',
            evidence='Capacity increased from3 to4 after observing8 physical/logical CPUs,50.7% CPU and59.5% RAM. Nine ownership/reservation tests pass with the reviewed reader guard. A monitor-only clean restart adopts all three prior live jobs; the recorded epoch gap is28.1576526165 seconds and no clock reset or unbounded backfill is used. The subsequent resource snapshot is62.5% CPU and60.4% RAM. Existing failed-work count is8 with no new failed scientific work. The complete O1 cohort and all8 fixed viewer alarm arrays have a new PASS review; final guide shock tables now separate memory and classifier endpoints. No final audit, ZIP or desktop delivery is claimed.',
            next_test='Finish all115 queued tasks and actual productive time, render/review complete-cohort figures, then admit final audit and verify the final ZIP through both portable replay entry points before20-file desktop copy.')
        paths=['conditional_statistics/N2/H_1000/DONE.json','support_partition/N2/H_1000/DONE.json',
            'conditional_statistics/N2/H_2048/DONE.json','support_partition/N2/H_2048/DONE.json',
            'normalizer_transport/correlations/N2/H_2048/DONE.json','classifier_statistics/T1/DONE.json',
            'classifier_statistics/T2/DONE.json','classifier_statistics/O1/DONE.json','classifier_statistics/OB16/DONE.json',
            'audit/O1_CLASSIFIER_MEMORY_BOUNDARY_REVIEW_001.json','research_notes/CLASSIFIER_SHOCK_WITHOUT_EVENT_MEMORY_KO.md',
            'audit/FOUR_WORKER_MONITOR_RESTART_001.json','audit/FOUR_WORKER_RESOURCE_FOLLOWUP_001.json',
            'FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json','ROLE_GATE_FEASIBILITY_LOCK_NEXT7.json']
        h=read_json(BASE/'runtime/HEARTBEAT.json')
        result={'created_utc':utc_now(),'cycle':12,'status':'RESEARCH_CONTINUES',
            'wall_seconds':h['wall_seconds'],'productive_seconds':h['productive_seconds'],
            'priority':['B2','B5','B4','B3','B1','B6','B7'],'items':list(items.values()),
            'current_primary':'FROZEN_BL80','learned_candidate_result':'NO_PROMOTION',
            'research_duration_requirement_not_yet_met':not h['research_minimum_met'],
            'prior_cycle_sha256':digest(previous),'evidence_sha256':{name:digest(BASE/name) for name in paths},
            'no_new_candidate_fit_or_threshold_or_random_draw':True}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_012.json',result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json',result,False)
        with (BASE/'BOTTLENECK_HISTORY_NEXT7.csv').open(newline='',encoding='utf-8-sig') as handle:
            history=list(csv.DictReader(handle))
        history += [{'cycle':12,'created_utc':result['created_utc'],'id':r['id'],'status':r['status'],
            'severity':r['severity'],'evidence':r['evidence'],'next_test':r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv',history,immutable=False)
    note('Bottleneck cycle012 complete','Classifier shock failures and ARCH support mismatch; no promotion','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    main()

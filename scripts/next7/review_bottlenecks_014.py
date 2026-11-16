"""Review final fixed-cohort evidence before research closure, without new fits."""
from pathlib import Path
import csv
import datetime as dt
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,activity,note


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def main():
    previous=BASE/'bottlenecks/NEXT7_BOTTLENECK_CYCLE_013.json'
    prior=read_json(previous)
    elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(prior['created_utc'])).total_seconds()
    assert elapsed>=45*60
    assert not (BASE/'runtime/FINAL_AUDIT_ADMISSION.json').exists()
    items={r['id']:r for r in prior['items']}
    paths=['COMPLETE_DIAGNOSTICS_CLASSIFIERS_NEXT7.json','COMPLETE_DIAGNOSTICS_GRADIENTS_NEXT7.json',
        'FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json','ROLE_GATE_FEASIBILITY_DONE_NEXT7.json',
        'SHOCK_CLASSIFIER_JOINT_AUDIT_DONE_NEXT7.json','CLASSIFIER_STRESS_RESULTS_NEXT7.csv',
        'CLASSIFIER_STRESS_PAIRED_NEXT7.csv','visuals/classifier_stress/full_suite_001/MANIFEST.json',
        'audit/CLASSIFIER_FULL_SUITE_VISUAL_REVIEW_001.json','audit/FULL_SUITE_PLOT_REPAIR_001.json',
        'audit/failure_registry/research_003/DONE.json','audit/BOTTLENECK_CYCLE_013_COUNT_CORRECTION_001.json']
    rates=[];contrasts=[];partitions=[]
    for case in ('N0','N1','N2','N3'):
        for length in (1000,2048,5000):
            directory=BASE/f'conditional_statistics/{case}/H_{length}'
            if (directory/'DONE.json').exists():
                assert read_json(directory/'DONE.json')['status']=='COMPLETE'
                row=next(r for r in rows(directory/'SUMMARY.csv') if r['horizon']=='16384' and r['metric']=='G2_QUALIFIED_EVER_LATCH')
                rates.append({'case':case,'H_length':length,'rate':row['mean_crossed_rate'],
                    'CI':[row['CI_low_approx_pointwise'],row['CI_high_approx_pointwise']]})
                paths += [(directory/p).relative_to(BASE).as_posix() for p in ('DONE.json','SUMMARY.csv')]
            support=BASE/f'support_partition/{case}/H_{length}'
            if length==5000 and (support/'DONE.json').exists():
                partitions.extend(r for r in rows(support/'SUMMARY.csv') if r['horizon']=='16384')
                paths += [(support/p).relative_to(BASE).as_posix() for p in ('DONE.json','SUMMARY.csv')]
        directory=BASE/f'crossed_H_length_comparison/{case}'
        if (directory/'DONE.json').exists():
            row=next(r for r in rows(directory/'PAIRED_H_LENGTH_CONTRASTS.csv') if r['horizon']=='16384' and
                r['metric']=='G2_QUALIFIED_EVER_LATCH' and r['H_candidate']=='5000' and r['H_reference']=='2048')
            contrasts.append({'case':case,'delta':row['mean_rate_delta'],
                'CI':[row['CI_low_approx_pointwise'],row['CI_high_approx_pointwise']]})
            paths += [(directory/p).relative_to(BASE).as_posix() for p in ('DONE.json','PAIRED_H_LENGTH_CONTRASTS.csv')]
    moments_path=BASE/'CROSSED_MOMENT_AUDIT_DONE_NEXT7.json'
    moments=read_json(moments_path) if moments_path.exists() else None
    if moments:
        assert moments['status']=='COMPLETE' and moments['rows']==108
        paths += ['CROSSED_MOMENT_AUDIT_DONE_NEXT7.json','CROSSED_RESAMPLING_EXACT_MOMENTS_NEXT7.csv']
    with activity('Bottleneck cycle014: completed classifier suite and final crossed-cohort synthesis',
                  'BOTTLENECK_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        items['B1'].update(status='ALL23_CLASSIFIER_CASES_COMPLETE_LOCAL_GAINS_DO_NOT_REPAIR_GLOBAL_RANKING',
            evidence='All23 original512-seed cases and eight methods are consolidated into6400 summary rows and12000 paired rows. RETURN_NORMAL G3 first-detection delta at1024 is+.0421875 with interval[+.029287109375,+.055478515625], while TEMP_SCALE_PULSE delta is-.005078125 with interval crossing0. LATE_PERSISTENT at4096 improves+.0625 with interval[+.045693359375,+.079306640625]; its1024 endpoint remains unobserved, not zero. SCALE_PLUS_SHOCK at1024 improves+.048828125 with interval[+.035146484375,+.062890625]. These fixed local gains coexist with all15 negative mean and worst-fold RP deltas.',
            next_test='No additional candidate or threshold. Complete final lineage, online-contract and runtime audits, retaining complete positive and negative scenario evidence.')
        items['B2'].update(status='CLASSIFIER_SHOCK_FAILURES_AND_ENDPOINT_DECOMPOSITION_RETAINED',
            evidence='The complete23-case figures show negative-case any alarms separately from true-change first detection. O1 and burst/repeated shock classifier alarm rates remain high. The original fixed O1 seed0 has zero raw/qualified memory but classifier alarms. In repeated shocks at4096, G3 treatment-any and control-any both increase even though treatment-only decreases. The48-row joint audit and114 original paired-mean identities preserve the distinction; six historical M3 joint rows remain missing.',
            next_test='Keep confirmation-only memory, final classifier alarms, overall error and paired treatment-only error separate. No robustness claim from a decreasing one-way endpoint alone.')
        items['B3'].update(status='NO_PROMOTION_AND_STRICT_CORE_NONOVERLAP_UNMET',
            evidence='All15 new candidates fail the independently user-required nonworsening worst-fold criterion. The24-row role-gate feasibility audit also exposes the agent-quantified absolute.10 floor limitation and preserves endpoint ambiguity rather than selecting an advantageous reference. New scale raw blocks are disjoint; inherited lag4 raw overlap is still an unmet requirement. Completed negative-research delivery is not approval of a strict-disjoint production detector.',
            next_test='Preserve existing BL80/D2/J102 and no new operating role. Describe the unevaluated within-block lag design and required recalibration only as a next-session hypothesis.')
        items['B4'].update(status=f'COMPLETED_{len(rates)}_OF12_CROSSED_COHORTS',
            evidence='Long qualified risk and original pointwise intervals by completed background/H: '+str(rates)+
                '. Original matched H5000-minus-H2048 contrasts: '+str(contrasts)+
                '. N1 positive matched increase does not generalize automatically; N2 interval crosses0. No new H selection, causal sample-size attribution or conditional5% guarantee.',
            next_test='Require all12 completed cohorts and all four paired H comparisons for research closure, with complete-cohort figure review.')
        items['B5'].update(status='FINITE_SUPPORT_NORMALIZER_TRANSPORT_AND_RESAMPLING_SCOPE_SEPARATED',
            evidence='Completed H5000 long-risk three-way partitions: '+str([{k:r[k] for k in ('case','component','crossed_rate','fraction_of_all_G2_latches')} for r in partitions])+
                '. The exact finite-multinomial moment audit is '+('COMPLETE with108 rows and no replacement of the original2000-draw intervals.' if moments else 'PENDING; required before closure.')+
                ' Support exceedance, within-support terminal exceedance and earlier retained memory are distinct. H-normalizer correlations remain posthoc associations.',
            next_test='Keep original counts and intervals unchanged; include all-cohort diagnostics and exact-moment checks without making a population coverage claim.')
        items['B6'].update(status='FULL_PAIR_TRAINING_OBJECTIVE_AUDIT_COMPLETE_HELDOUT_GAIN_ABSENT',
            evidence='All five exact full-pair audits are complete. G3 improves its penalized training objective relative to the frozen G2 parent in each fold, but held-out mean ranking is worse. Small sampled gradients do not establish an optimum of the full-pair objective, and training-objective bounds do not guarantee held-out TS-AUC.',
            next_test='Execute the frozen quiet scalar benchmark after research closure; no loss refinement or refitting from the final diagnostics.')
        h=read_json(BASE/'runtime/HEARTBEAT.json')
        failures=read_json(BASE/'audit/failure_registry/research_003/DONE.json')
        assert failures['status']=='PASS' and not failures['unresolved_engineering_failures']
        items['B7'].update(status='PRODUCTIVE_MINIMUM_MET_AND_REMAINING_CLOSURE_CHECKS_TRACKED',
            evidence=f"Observed productive seconds={h['productive_seconds']}; audit seconds={h['audit_seconds']}. All23 classifier cases and{len(rates)} crossed cohorts complete. Both all-scenario PNGs reviewed,368 cells bound to original results, eight unobserved late-change cells retained. Nine main failed work units are repaired and reproduced; the latest was a Matplotlib keyword compatibility issue, with the original failure and empty failed directory preserved. No new scientific fit or cutoff and no final delivery claimed.",
            next_test='Finish remaining required cross-cohort artifacts and visual reviews, freeze the negative decision, admit final audit, then verify both exact extracted replay entry points and20 desktop copies.')
        result={'created_utc':utc_now(),'cycle':14,'status':'RESEARCH_CONTINUES',
            'wall_seconds':h['wall_seconds'],'productive_seconds':h['productive_seconds'],
            'priority':['B5','B4','B2','B3','B1','B6','B7'],'items':list(items.values()),
            'current_primary':'FROZEN_BL80','learned_candidate_result':'NO_PROMOTION',
            'research_duration_requirement_not_yet_met':not h['research_minimum_met'],
            'completed_crossed_cohorts':len(rates),'completed_H_length_comparisons':len(contrasts),
            'prior_cycle_sha256':digest(previous),'evidence_sha256':{name:digest(BASE/name) for name in paths},
            'no_new_candidate_fit_or_threshold_or_random_draw':True}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_014.json',result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json',result,False)
        history=rows(BASE/'BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle':14,'created_utc':result['created_utc'],'id':r['id'],'status':r['status'],
            'severity':r['severity'],'evidence':r['evidence'],'next_test':r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv',history,immutable=False)
    note('Bottleneck cycle014 complete','Original full classifier suite and completed crossed cohorts; no promotion','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    main()

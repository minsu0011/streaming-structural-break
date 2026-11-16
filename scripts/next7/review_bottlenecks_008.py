"""Review background-specific alarm power and H arithmetic interpretation."""
from pathlib import Path
import csv
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,activity,note


def rows(name):
    with (BASE/name).open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def main():
    previous=BASE/'bottlenecks/NEXT7_BOTTLENECK_CYCLE_007.json'
    items={r['id']:r for r in read_json(previous)['items']}
    with activity('Bottleneck cycle008: null heterogeneity, mean-change power and H arithmetic',
                  'BOTTLENECK_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        n3=[r for r in rows('classifier_statistics/N3/PAIRED_CONTRASTS.csv') if
            r['horizon']=='16384' and r['metric']=='any_alarm' and r['reference']=='M0']
        mean_changes=[]
        for case in ('M1','M2'):
            mean_changes += [r for r in rows(f'classifier_statistics/{case}/PAIRED_CONTRASTS.csv') if
                r['horizon']=='4096' and r['metric']=='postbreak_first_detection' and r['reference']=='M0']
        geometry=rows('RANK_REFERENCE_GEOMETRY_SUMMARY_NEXT7.csv')
        assert max(float(r['max_formula_absolute_error']) for r in geometry)<2e-16
        items['B2'].update(status='FOUR_NULL_CLASSIFIER_COHORTS_COMPLETE_WITH_MIXED_BACKGROUND_EFFECTS',
            evidence='N3,16384: M0 alarm=.64921875; G2_EVENT=.615625, paired delta-.03359375 CI[-.057421875,-.009365234375]. I4=.678125 worsens, while G2_FULL/G3 differences cross zero. N0/N1 candidate false alarms are higher; N2/N3 old M3 reduces false alarms. All four backgrounds are displayed, without averaging away the differences.',
            next_test='Complete the remaining17 original classifier cases including every variance/shock/temporary/late case. No calibration or selection change.')
        items['B3'].update(status='STRONG_SYNTHETIC_MEAN_CHANGE_POWER_DOES_NOT_TRANSFER_TO_OVERALL_RP_RANKING',
            evidence='M1/M2 at4096: G2_FULL first correct alarm increases .28203125/.2546875 versus M0; G3 increases .30703125/.271484375. All corresponding pointwise paired intervals are positive. These local gains coexist with negative overall RP TS-AUC; only16 RP series have the frozen posthoc mean-dominant signature. That signature is descriptive, not a true DGP label or an inference gate.',
            next_test='Finish all fixed synthetic mechanisms and report their gains as well as losses. Preserve the original RP mix and promotion gates; do not manufacture a favorable reweighted champion.')
        items['B5'].update(status='RANK_MOMENT_TIES_AND_TAIL_IQR_GEOMETRY_PROVEN_ALL6144_HISTORIES',
            evidence='Exact H-rank second moment equals(n-1)/(3(n+1))-sum(m^3-m)/(3n(n+1)^2), with m float32 tie multiplicities. All6144 stored prefix/full differences agree within1.67e-16; N0H1000 has no ties and only rounding variation. Tail block means are on1/16, IQRs on1/64 before the fixed floor. The near-constant moment correlation is not an independent distributional feature. All23 original correlations remain visible with an explicit geometry label.',
            next_test='Complete the remaining nine crossed cohorts and all normalizer/H-length comparisons. Keep transport association separate from causal attribution or new threshold design.')
        items['B1'].update(status='FINITE_SHOCK_INFLUENCE_PROOF_TESTED_FULL_PAIRED_AUDIT_QUEUED',
            evidence='Boundary perturbation tests verify finite local-evidence influence under lag4 and four complete blocks, followed by at most two recovery-block steps for current states. Lifetime memory has no equality/erasure requirement. A fixed audit of all1536 original O1/OB16/repeated-shock pairs is queued; no completed full-pair result is claimed yet.',
            next_test='Execute the queued original-pair audit and quantify persistent memory differences after local/current convergence using the unchanged2000 seed weights.')
        items['B7'].update(status='CONCRETE_FINAL_AUDIT_AND_DELIVERY_DRIVERS_PREPARED_NOT_EXECUTED',
            evidence='Prepared strict closure gate, nested row/OOF lineage audit, full regression, online contracts, raw membership audit, scalar runtime,45-question Korean report generator, byte-manifest ZIP extraction/replay and exact20 desktop copy. Twenty-one archive-boundary tests pass; reviewed src/data loader source is allowed while raw data remains blocked. Productive minimum and remaining science are still pending.',
            next_test='Complete required productive research and science. Then freeze candidates, admit audit, execute all checks and direct runtime, render final complete-cohort figures, write documents, freeze the ledger and verify final ZIP/desktop delivery.')
        evidence=['classifier_statistics/N3/DONE.json','classifier_statistics/M1/DONE.json','classifier_statistics/M2/DONE.json',
            'RANK_REFERENCE_GEOMETRY_AUDIT_DONE.json','SHOCK_INFLUENCE_AUDIT_LOCK.json','SHOCK_INFLUENCE_QUEUE_SUBMISSION.json',
            'audit/SHOCK_INFLUENCE_TESTS_002.xml','audit/HANDOFF_ARCHIVE_TESTS_003.xml',
            'audit/HANDOFF_AVAILABLE_CONTENT_PREFLIGHT_002.json','audit/TARGETED_TEST_CACHE_LOCATION_NOTE_001.json',
            'visuals/classifier_nulls/four_backgrounds_001/DONE.json','audit/CLASSIFIER_NULL_VISUAL_REVIEW_001.json',
            'visuals/normalizer_transport/n0_all_h_geometry_002/MANIFEST.json','audit/NORMALIZER_VISUAL_REVIEW_002.json',
            'HANDOFF_CONTENT_PLAN_NEXT7.json']
        heartbeat=read_json(BASE/'runtime/HEARTBEAT.json')
        result={'created_utc':utc_now(),'cycle':8,'status':'RESEARCH_CONTINUES',
            'wall_seconds':heartbeat['wall_seconds'],'productive_seconds':heartbeat['productive_seconds'],
            'priority':['B5','B4','B2','B3','B1','B6','B7'],'items':list(items.values()),
            'current_primary':'FROZEN_BL80','learned_candidate_result':'NO_PROMOTION',
            'research_duration_requirement_not_yet_met':not heartbeat['research_minimum_met'],
            'prior_cycle_sha256':digest(previous),'N3_long_classifier_paired_results':n3,
            'mean_change_4096_first_detection_results':mean_changes,'H_arithmetic_geometry':geometry,
            'evidence_sha256':{name:digest(BASE/name) for name in evidence}}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_008.json',result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json',result,False)
        history=rows('BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle':8,'created_utc':result['created_utc'],'id':r['id'],'status':r['status'],
                     'severity':r['severity'],'evidence':r['evidence'],'next_test':r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv',history,immutable=False)
    note('Bottleneck cycle008 complete','Synthetic mean-change gains coexist with RP rank loss and background-dependent false alarms','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':main()

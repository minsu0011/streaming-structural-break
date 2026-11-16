"""First evidence-driven review; close G1 tuning, register diagnostic priorities."""
from pathlib import Path
import csv
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,save,read_json,csv_write,utc_now,digest,activity,note


def main():
    with activity('bottleneck cycle001: synthetic gain versus RP loss','BOTTLENECK_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        board=read_json(BASE/'BOTTLENECK_BOARD_INITIAL_NEXT7.json')
        scores=read_json(BASE/'G1_OUTER_SCORES.json')
        table={r['method']:r for r in scores['summaries']}
        byid={r['id']:r for r in board['items']}
        byid['B1'].update({'status':'G1_DIRECT_REPRESENTATION_CLOSED_NO_TUNING',
            'evidence':'V4 first detection .05078125 -> .4375 for fixed S1/C1 union; RP scale-only TS-AUC .514803988134479; BL80+scale delta -.007765916685936602.',
            'severity':'MEDIUM','expected_TS_AUC_impact':'Negative for tested direct G1 usage',
            'confidence':'High for these fixed paths/OOF; not a universal scale impossibility theorem',
            'next_test':'Mechanism and pairwise attribution; no S1/S2 threshold or blend retuning'})
        byid['B2'].update({'status':'G1_QUANTIFIED_NOT_ZERO',
            'evidence':'S1/C1 O1 overall1024 latch .078125, excess paired-control latch .00390625; OB16 .099609375 and excess .025390625; repeated .080078125 and excess .009765625.',
            'severity':'MEDIUM','next_test':'Compare final detector calibrated alarms and event contamination separately; audit every excess-latch trajectory'})
        byid['B3'].update({'status':'OPEN_HIGH_PRIORITY',
            'evidence':'CURRENT_ONLY TS-AUC .5356774697472892; EVENT_ONLY .5206821084384895; M_EVENT delta -.001374571894001675.',
            'severity':'HIGH','expected_TS_AUC_impact':'Observed memory addition harms ranking',
            'next_test':'Exact rank-pair decomposition by positive/negative latch status, first-latch age and recovery'})
        byid['B4'].update({'status':'TOP_PRIORITY_HARD_GATE_FAIL',
            'evidence':'S1/C1 16384 null latch N0 .615234375, N1 .486328125, N2 .609375, N3 .54296875; frozen hard4 .423828125/.361328125/.427734375/.341796875.',
            'severity':'CRITICAL','expected_TS_AUC_impact':'Uncorrected event memory is ineligible under the fixed >.50 long-null gate',
            'confidence':'High for fixed synthetic cohort; 2000 paired seed CI will be reported',
            'next_test':'Register H-only/observed-block-count null-growth memory correction before evaluating it',
            'stop_rule':'Do not relabel or weaken the existing long-null gate; no nominal probability guarantee without calibration proof'})
        byid['B5'].update({'status':'OPEN','evidence':'G1 uses H-only block reference with fit/cal shrinkage and frozen H routing; independent H-transport not yet completed.',
            'next_test':'Fixed H1000/1500/3000/5000 transport, support/finite-sample limits, reference uncertainty'})
        byid['B6'].update({'status':'OPEN_HIGH_PRIORITY','evidence':'Best G1 I4 event/scale tree delta -.0003624879784099866; no positive incremental effect from the preregistered corrections.',
            'next_test':'Within-time rank/Pearson/Spearman and exact correction contribution; no extra learner capacity search'})
        byid['B7'].update({'status':'ENFORCED','evidence':'Six synthetic screen variants;12 full candidates including3 standalone diagnostics;5 G1 finalists;4 full slots left.',
            'next_test':'Only distinct preregistered structural hypotheses; otherwise forensic/runtime/reproduction',
            'stop_rule':'No score-driven micro-search; retain frozen BL80'})
        heartbeat=read_json(BASE/'runtime/HEARTBEAT.json')
        result={'created_utc':utc_now(),'cycle':1,'status':'RESEARCH_CONTINUES',
            'wall_seconds':heartbeat['wall_seconds'],'productive_seconds':heartbeat['productive_seconds'],
            'priority':['B4','B3','B5','B6','B2','B1','B7'],'items':list(byid.values()),
            'current_primary':'FROZEN_BL80','best_observed_G1':table['I4_EVENT_SCALE_TREE'],
            'G1_threshold_and_weight_search_closed':True,
            'next_structural_family':'F4_NULL_GROWTH_CORRECTION',
            'new_structure_must_be_registered_before_its_scores':True,
            'G1_outer_score_sha256':digest(BASE/'G1_OUTER_SCORES.json')}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_001.json',result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json',result,False)
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv',[{'cycle':1,'created_utc':result['created_utc'],
            'id':r['id'],'status':r['status'],'severity':r['severity'],'evidence':r['evidence'],
            'next_test':r['next_test']} for r in byid.values()])
    note('Bottleneck cycle001 complete','G1 tuning closed; investigate false latch, target alignment and H reference uncertainty','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    main()

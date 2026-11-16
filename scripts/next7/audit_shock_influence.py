"""Replay original paired shocks to separate finite local influence and memory."""
from pathlib import Path
import argparse
import os
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,activity,note,require_guard
from src.next7.shock_influence import compare_paths,LOCAL_COLUMNS,CURRENT_STATE_COLUMNS,MEMORY_COLUMNS
from src.next7.fast_state import FastQualifiedState
from src.next7.calibration import load_calibration,cutoff
from src.next6_extension.synthetic import generated


def freeze():
    names=['scripts/next7/audit_shock_influence.py','src/next7/shock_influence.py',
        'src/next7/fast_state.py','src/next7/event.py','src/next7/scale.py','src/next7/null_growth.py',
        'src/next6_extension/fresh_state.py','src/next6_extension/synthetic.py']
    sources={name:digest(ROOT/name) for name in names}
    path=BASE/'SHOCK_INFLUENCE_AUDIT_LOCK.json'
    if not path.exists():
        tests=BASE/'audit/SHOCK_INFLUENCE_TESTS_002.xml'
        xml=ET.parse(tests)
        assert len(xml.findall('.//testcase'))==5 and not xml.findall('.//failure') and not xml.findall('.//error')
        save(path.name,{'status':'FROZEN','created_utc':utc_now(),'source_sha256':sources,
            'tests_sha256':digest(tests),'guarded_test_receipt_sha256':digest(BASE/'audit/SHOCK_INFLUENCE_TEST_ACCESS_002.json'),
            'origin':'Posthoc mechanistic audit of why finite shock evidence and lifetime event memory can diverge; not a new candidate or threshold search.',
            'cases':['O1','OB16','REPEATED_SHOCK'],'seed_indices':list(range(512)),
            'partition':31,'H_length':2048,'horizon':4096,'new_random_draws':0,'new_fits':0,'new_candidates':0,
            'local_columns':LOCAL_COLUMNS,'current_state_columns':CURRENT_STATE_COLUMNS,'memory_columns':MEMORY_COLUMNS,
            'analytic_bound':'Let L be the last different raw point and b=floor((L+4)/16). At point16*(b+5)-1 four fully unaffected core blocks have displaced all affected blocks. Local fresh evidence then agrees. At most32 further points resolve both current-state recovery automata. No equality bound is imposed on event memories.',
            'inference_scope':'Same H and current scalar only. Intervention positions are used after replay to audit the bound, never to initialize or update a detector.',
            'statistics':'All512 original paired seeds. Original2000x512 multiplicities only for descriptive paired memory differences at4096; no extra draws, independent-fold or causal-graph claim.',
            'limits':'A known paired synthetic intervention can have persistent memory consequences after local evidence equality. This does not establish real-data causality or equality of BL80-based classifier predictions.'})
    assert read_json(path)['source_sha256']==sources
    return digest(path)


def main(lock_only=False):
    identity=freeze()
    if lock_only:return
    require_guard()
    suite=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    calibration=load_calibration()
    counts=np.load(BASE/'synthetic_stats/COUNTS_2000x512.npy',allow_pickle=False)
    assert counts.shape==(2000,512)
    count_lock=read_json(BASE/'synthetic_stats/COUNTS_LOCK.json')
    assert digest(BASE/'synthetic_stats/COUNTS_2000x512.npy')==count_lock['sha256']
    records,inputs=[],{}
    for case in ('O1','OB16','REPEATED_SHOCK'):
        with activity('Finite shock influence and persistent memory '+case,'SHOCK_MEMORY_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
            for seed in range(512):
                relative=f'shock_influence/{case}/seed_{seed:03d}.json'
                p=BASE/relative
                if p.exists():
                    row=read_json(p)
                    assert row['identity']==identity and row['status']=='PASS'
                else:
                    h,a,_,info=generated(case,seed,partition=31,horizon=4096,control=False,lock=suite)
                    hc,b,_,_=generated(case,seed,partition=31,horizon=4096,control=True,lock=suite)
                    np.testing.assert_array_equal(h,hc)
                    assert len(h)==2048 and len(a)==len(b)==4096
                    changed=np.flatnonzero(a!=b)
                    state=FastQualifiedState(h,core_cut=calibration['core_cut'],scale_cut=cutoff(calibration,len(h)))
                    control_state=state.clone()
                    out,control=state.replay(a),control_state.replay(b)
                    comparison=compare_paths(out,control,changed)
                    row={'status':'PASS','identity':identity,'case':case,'seed':seed,'horizon':4096,
                        'changed_raw_points':len(changed),**comparison,
                        'raw_memory_diff_at_end':bool(out[-1,3]!=control[-1,3]),
                        'qualified_memory_diff_at_end':bool(out[-1,30]!=control[-1,30]),
                        'any_memory_coordinate_diff_after_current_convergence':comparison['memory_max_difference_after_current_convergence']>0,
                        'raw_intervention_latched':bool(out[-1,3]>0),'raw_control_latched':bool(control[-1,3]>0),
                        'qualified_intervention_latched':bool(out[-1,30]>0),'qualified_control_latched':bool(control[-1,30]>0)}
                    save(relative,row)
                records.append(row)
                inputs[relative]=digest(p)
                if seed%64==0:print(utc_now(),case,seed,'/512',flush=True)
    metrics=('raw_intervention_only_latch_at_end','raw_control_only_latch_at_end',
        'qualified_intervention_only_latch_at_end','qualified_control_only_latch_at_end',
        'raw_memory_diff_at_end','qualified_memory_diff_at_end','any_memory_coordinate_diff_after_current_convergence',
        'raw_ever_score_difference_at_end','qualified_ever_score_difference_at_end','qualified_episode_difference_at_end')
    summaries=[]
    for case in ('O1','OB16','REPEATED_SHOCK'):
        cohort=[r for r in records if r['case']==case]
        assert len(cohort)==512 and [r['seed'] for r in cohort]==list(range(512))
        for metric in metrics:
            values=np.array([r[metric] for r in cohort],dtype=np.float64)
            draws=counts@values/512
            low,high=np.quantile(draws,[.025,.975])
            summaries.append({'case':case,'horizon':4096,'metric':metric,'mean':float(values.mean()),
                'CI_low_pointwise':float(low),'CI_high_pointwise':float(high),'draws':2000,
                'fixed_original_seed_pairs':512,'new_draws':0,'local_evidence_all512_bitwise_equal_after_bound':True,
                'current_state_all512_bitwise_equal_after_bound':True,'classifier_prediction_identity_claim':False})
    csv_write('SHOCK_LOCAL_INFLUENCE_AND_MEMORY_NEXT7.csv',records)
    csv_write('SHOCK_PERSISTENT_MEMORY_DIFFERENCES_NEXT7.csv',summaries)
    save('SHOCK_INFLUENCE_AUDIT_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'lock_sha256':identity,
        'original_pairs_checked':1536,'scalar_observations_replayed':1536*4096*2,
        'all_prefix_and_local_evidence_and_current_bounds_bitwise':True,
        'original_count_sha256':digest(BASE/'synthetic_stats/COUNTS_2000x512.npy'),
        'per_pair_receipts_sha256':inputs,'outputs_sha256':{name:digest(BASE/name) for name in
            ('SHOCK_LOCAL_INFLUENCE_AND_MEMORY_NEXT7.csv','SHOCK_PERSISTENT_MEMORY_DIFFERENCES_NEXT7.csv')},
        'new_candidates':0,'new_draws':0,'new_fits':0})
    note('Shock influence audit complete','All1536 original pairs: finite local evidence contrasted with lasting memory','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock-only',action='store_true')
    main(parser.parse_args().lock_only)

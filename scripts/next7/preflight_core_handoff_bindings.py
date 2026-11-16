"""Validate completed handoff bindings against the original scope definitions."""
from pathlib import Path
import math
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.next7.common import BASE,read_json,save,digest,utc_now,activity
from scripts.next7.write_final_handoff import rows,choose
from src.next5.fractures import masks


def main():
    sources=['MODEL_COMPARISON_NEXT7.csv','SHOCK_EVENT_CONTAMINATION_NEXT7.csv',
        'SCALE_UP_DOWN_RESULTS_NEXT7.csv','LONG_NULL_RESULTS_NEXT7.csv',
        'H_LENGTH_CALIBRATION_NEXT7.csv','REGIME_RESULTS_NEXT7.csv',
        'TIME_FRACTURE_NEXT7.csv','BREAK_AGE_NEXT7.csv','SCALE_FIRST_EVENT_FAILURE_PARTITION_NEXT7.csv']
    checks=0
    with activity('Corrected completed core handoff data-binding preflight',
                  'HANDOFF_ENGINEERING','REPRODUCTION_AND_RUNTIME'):
        models={r['method']:r for r in rows(sources[0])}
        for method in ('M0','I4_EVENT_SCALE_TREE','G2_FULL','G3_PAIRWISE_FULL'):
            for fold in range(5):
                assert math.isfinite(float(models[method][f'fold_{fold}_TS_AUC']))
                checks+=1
        for key in ('median_matched_fold_delta_M0','difference_of_fold_score_medians_M0',
                    'worst_matched_fold_delta_M0','mean_delta_CI_low','mean_delta_CI_high',
                    'probability_mean_delta_positive'):
            assert math.isfinite(float(models['I4_EVENT_SCALE_TREE'][key]))
            checks+=1
        for name,cases,horizons,metrics in (
            (sources[1],('O1','OB16','REPEATED_SHOCK'),(1024,4096),('any_latch','excess_latch_vs_paired_control')),
            (sources[2],('V1','V2','V3','V4'),(1024,),('first_detection',)),
            (sources[3],('N0','N1','N2','N3'),(1024,4096,16384),('any_latch',))):
            data=rows(name)
            for case in cases:
                for horizon in horizons:
                    for metric in metrics:
                        for method in ('NEXT6_HARD4_EVENT','G1_RAW_EVENT','G2_QUALIFIED_EVENT'):
                            row=choose(data,case=case,horizon=horizon,metric=metric,method=method)
                            assert 0<=float(row['estimate'])<=1
                            checks+=1
        # Enumerate names with empty synthetic arrays: no RP observation or
        # outcome is read and no score is computed to discover scope names.
        empty=np.array([],dtype=np.int64)
        text_empty=np.array([],dtype='U64')
        scope_names=set(masks(empty,empty,empty,empty,text_empty,text_empty))
        selected_scopes={}
        for name,prefix in ((sources[5],'regime_'),(sources[6],'time_'),(sources[7],'age_')):
            records=[r for r in rows(name) if r['candidate']=='G2_FULL' and r['reference']=='M0' and r['scope'].startswith(prefix)]
            expected={name for name in scope_names if name.startswith(prefix)}
            assert len(records)==len(expected) and {r['scope'] for r in records}==expected
            checks+=len(records)
            selected_scopes[prefix]=sorted(expected)
        save('audit/CORE_HANDOFF_BINDING_PREFLIGHT_002.json',{'status':'PASS','created_utc':utc_now(),
            'bindings_checked':checks,'input_sha256':{name:digest(BASE/name) for name in sources},
            'source_sha256':digest(__file__),'original_scope_source_sha256':digest(ROOT/'src/next5/fractures.py'),
            'writer_source_sha256':digest(ROOT/'scripts/next7/write_final_handoff.py'),
            'exact_original_scope_names':selected_scopes,
            'scope':'Completed core bindings only. No final audit status, runtime, pending classifier results or final document rendering is claimed.'})
    print('PASS',checks,'bindings',selected_scopes,flush=True)


if __name__=='__main__':main()

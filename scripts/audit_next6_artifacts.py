"""End-of-run byte integrity and finite model tables; no model selection."""
from pathlib import Path
import sys,os,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from src.next6.common import BASE,read_json,save,digest,activity,checkpoint,utc_now
from src.next6.data import source
from src.next6.graph import REL_NAMES,STATE_NAMES
from src.next6.evaluation import csv_write

def catalog():
    if (BASE/'STREAMING_GRAPH_FEATURE_CATALOG.csv').exists():return
    rows=[]
    for j,name in enumerate((*REL_NAMES,*STATE_NAMES)):
        rows.append({'column':j,'feature_name':name,'method':'M2/M4' if j<12 else 'M3/M4','kind':'conditional regression' if j<12 else 'factorized transition',
            'runtime_inputs':'current and past H-normalized nodes; five actual lags; current online counter; training-frozen graph',
            'label_tau_final_length_input':False,'normal_and_post_scored_concurrently':True,'probability_calibrated':False,
            'state_memory':'16x16 categorical ring +9 scalar summaries, independent of online length',
            'confirmation_caveat':'distinct derived nodes may share raw/filter ancestry; confirmation is operational, not independent economic evidence',
            'formula_source':'src/next6/graph.py::graph_replay','missing':'state3 skipped in probability terms; missing fraction explicit'})
    csv_write('STREAMING_GRAPH_FEATURE_CATALOG.csv',rows)
    directions=[('TS_AUC','higher','same time-pair support required'),('mean_relative_MSE_gain','higher','own-history denominator floor .01; equal series'),
        ('parent_vs_own_log_probability_gain','higher','predicting feature states, not calibrated break probability'),('null_eventual_alarm_rate','lower','fixed finite horizon'),
        ('prebreak_alarm_rate','lower','official tau, no-break separate'),('postbreak_first_detection_rate','higher','prebreak first alarms counted separately'),
        ('conditional_detection_delay_median','lower','always report detection rate and censoring'),('conditional_recovery_delay_median','lower','16 points within .01 of paired control; censoring separate'),
        ('paired_post_peak_increase','context dependent','amplitude shrinkage alone is not a detection improvement'),('scalar_latency','lower','same hardware/API/JIT treatment')]
    csv_write('METRIC_DIRECTIONS_NEXT6.csv',[dict(metric=m,favorable_direction=d,condition=c) for m,d,c in directions])

def main():
    catalog();data=source();protected=read_json(BASE/'PROTECTED_BASELINE_SHA256.json');changed=[];count=0;graphcount=0
    with activity('source_artifact_boundary_integrity','FINAL_AUDIT'):
        for name,sha in protected.items():
            if not (ROOT/name).is_file() or digest(ROOT/name)!=sha:changed.append(name)
        assert not changed,changed
        for name,sha in read_json(BASE/'IMPLEMENTATION_LOCK.json')['source_sha256'].items():assert digest(ROOT/name)==sha
        cache=read_json(BASE/'data/MANIFEST.json')
        for name,sha in cache['sha256'].items():assert digest(BASE/'data'/name)==sha
        receipts=list(BASE.glob('fold_*/**/DONE.json'))+list(BASE.glob('strict_baseline/fold_*/DONE.json'))
        for name,sha in read_json(BASE/'M1_INPUT_PARITY_CORRECTION_LOCK.json')['source_sha256'].items():assert digest(ROOT/name)==sha
        for name,sha in read_json(BASE/'EVALUATION_IMPLEMENTATION_LOCK.json')['source_sha256'].items():assert digest(ROOT/name)==sha
        assert digest(BASE/'strict_baseline/extras.npy')==read_json(BASE/'strict_baseline/EXTRAS_DONE.json')['sha256']
        for i,path in enumerate(receipts):
            rec=read_json(path)
            for field,name in [('model_sha256','model.joblib'),('prediction_sha256','predictions.npz'),('graph_sha256','graph.joblib'),('features_sha256','features.npy')]:
                if field in rec:assert digest(path.parent/name)==rec[field],str(path.parent/name);count+=1
            if 'graph_sha256' in rec:
                g=joblib.load(path.parent/'graph.joblib');assert len(g.selected)<=24 and max(map(len,g.parents))<=2
                assert np.isfinite(g.coef).all() and np.isfinite(g.tables).all() and np.min(g.tables)>0
                assert np.max(abs(g.tables.sum(axis=-1)-1))<1e-12 and np.max(abs(g.own_tables.sum(axis=-1)-1))<1e-12;graphcount+=1
            if i%20==0:checkpoint('artifact hashes',f'{i}/{len(receipts)} receipts')
        data.guard.check_state()
        save('final_audit/ARTIFACT_INTEGRITY.json',{'status':'PASS','timestamp':utc_now(),'protected_files_unchanged':len(protected),'changed_protected_files':changed,
            'fitted_artifact_hashes_checked':count,'graph_models_checked':graphcount,'probabilities_positive_normalized':True,
            'frozen_implementation_unchanged':True,'derived_node_cache_hashes_valid':True})
        save('FINAL_DATA_ACCESS_AUDIT_NEXT6.json',{'status':'PASS','timestamp':utc_now(),'initial_audit_sha256':digest(BASE/'DATA_ACCESS_AUDIT_NEXT6.json'),
            'ResearchPool_series':len(data.ids),'ResearchPool_online_rows':len(data.online),'split_and_shadow_seal_boundaries_unchanged':True,
            'Shadow_rows_returned':0,'Seal_rows_returned':0,'new_raw_scan':False,'raw_original_files_byte_hash_only':True,
            'synthetic_control_rows_are_generated_not_competition_test':True,'public_cloud_or_reduced_test_requests':0,'application_guard_not_OS_sandbox':True})

if __name__=='__main__':main()

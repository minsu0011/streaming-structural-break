"""Audit block timestamps and the extra lag footprint at a known training boundary.

This is one fixed training-support sensitivity check, not a new detector recipe.
The exported observer and all original graphs remain unchanged.
"""
from pathlib import Path
import sys, os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np, joblib
from src.next6_extension.common import *
from src.next6.data import source
from src.next6_extension.factorized_coherent import discover_and_fit

def main():
    lock=BASE/'BLOCK_BOUNDARY_AUDIT_LOCK.json'
    if not lock.exists():
        save(lock.name,{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),
            'question':'Original clean labels cover two blocks of current timestamps, while lag4 uses four earlier raw observations. Quantify and exclude only this residual boundary footprint.',
            'fixed_rule':'For a changed stream, a post pair is raw-footprint-pure only if time_end-35 >= tau. Keep normal labels unchanged; exclude residual mixed pairs with phase=-1.',
            'analysis':'Same coherent graph procedure, all five original outer-training sets only. No thresholds, features, lags, heads or exported model changes. Sensitivity results are adaptive development evidence.',
            'graph_source_sha256':digest(ROOT/'src/next6_extension/factorized_coherent.py')})
    assert read_json(lock)['source_sha256']==digest(Path(__file__))
    data=source();manifest=read_json(BASE/'block_cache/MANIFEST.json')
    for name,sha in manifest['files_sha256'].items():assert digest(BASE/'block_cache'/name)==sha
    v={Path(name).stem:np.load(BASE/'block_cache'/name,mmap_mode='r',allow_pickle=False) for name in manifest['files_sha256']}
    stricter=np.array(v['phase_clean'],copy=True);records=[]
    with activity('block_boundary_raw_footprint_audit','TRAINING_SUPPORT_SENSITIVITY'):
        cursor=0
        for i,sid in enumerate(data.ids):
            n=int(data.series['online_offsets'][i+1]-data.series['online_offsets'][i]);count=n//16;a=cursor;b=a+count;cursor=b
            end=np.arange(15,n,16);tau=int(data.series['tau'][i]);fold=int(data.series['fold'][i])
            assert np.all(v['dataset_id'][a:b]==sid) and np.all(v['fold'][a:b]==fold)
            np.testing.assert_array_equal(v['time_end'][a:b],end)
            np.testing.assert_array_equal(v['global_row'][a:b],data.series['online_offsets'][i]+end)
            expected=np.zeros(count,dtype=np.int8) if tau<0 else np.where(end<tau,0,np.where(end-31>=tau,1,-1)).astype(np.int8)
            np.testing.assert_array_equal(v['phase_clean'][a:b],expected)
            np.testing.assert_array_equal(v['target_end'][a:b],np.zeros(count,dtype=np.int8) if tau<0 else (end>=tau).astype(np.int8))
            if count>1:np.testing.assert_array_equal(v['previous'][a+1:b],v['states'][a:b-1])
            mixed=(expected==1)&(end-35<tau) if tau>=0 else np.zeros(count,dtype=bool)
            stricter[a:b][mixed]=-1
            records.append({'fold':fold,'series_blocks':count,'original_post_pairs':int(np.sum(expected==1)),'residual_lag_boundary_pairs':int(mixed.sum()),'series_has_residual_lag_boundary':bool(mixed.any()),'maximum_prebreak_raw_points':int(np.max(tau-(end[mixed]-35))) if mixed.any() else 0})
        assert cursor==manifest['blocks']==len(stricter)
        summary=[]
        for fold in range(5):
            rr=[r for r in records if r['fold']==fold]
            summary.append({'fold':fold,'series':len(rr),'blocks':sum(r['series_blocks'] for r in rr),'original_post_pairs':sum(r['original_post_pairs'] for r in rr),'residual_lag_boundary_pairs':sum(r['residual_lag_boundary_pairs'] for r in rr),'series_with_residual_boundary':sum(r['series_has_residual_lag_boundary'] for r in rr),'maximum_prebreak_raw_points':max(r['maximum_prebreak_raw_points'] for r in rr)})
        csv_write('BLOCK_BOUNDARY_FOOTPRINT_SUMMARY.csv',summary)
        fits=[]
        for fold in range(5):
            train=data.ids[data.series['fold']!=fold];valid=data.ids[data.series['fold']==fold];data.guard.partition(set(map(int,train)),set(map(int,valid)));tr=v['fold']!=fold
            graph,details=discover_and_fit(v['states'][tr],v['previous'][tr],stricter[tr],v['dataset_id'][tr],train,data.guard.split.groups)
            old=joblib.load(BASE/f'coherent_graphs/fold_{fold}/graph.joblib');old.validate()
            dest=BASE/f'boundary_sensitivity/fold_{fold}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'graph.joblib').exists();joblib.dump(graph,dest/'graph.joblib')
            csv_write(f'boundary_sensitivity/fold_{fold}/INNER_DISCOVERY.csv',details)
            same=graph.selected==old.selected
            fits.append({'fold':fold,'training_series':len(train),'removed_training_pairs':int(np.sum(tr&(v['phase_clean']==1)&(stricter==-1))),'original_selected':len(old.selected),'raw_footprint_pure_selected':len(graph.selected),'same_selected_connections':same,'same_graph_probabilities':bool(np.array_equal(graph.full,old.full) and np.array_equal(graph.null,old.null)),'training_ids_sha256':graph.provenance['training_ids_sha256'],'graph_sha256':digest(dest/'graph.joblib'),'exported_model_unchanged':True})
            checkpoint('block_boundary_raw_footprint_audit',f'fold {fold} complete; selected {len(graph.selected)}')
        csv_write('BLOCK_BOUNDARY_TRAINING_SENSITIVITY.csv',fits)
    data.guard.check_state()
    save('BLOCK_BOUNDARY_AUDIT_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'blocks_audited':cursor,'series_audited':len(data.ids),'residual_lag_boundary_pairs':sum(r['residual_lag_boundary_pairs'] for r in records),'original_clean_contract':'Two current-observation blocks have the same official phase; not all lagged raw inputs necessarily do.','strict_sensitivity':'Entire two-block plus lag4 footprint within phase; training only.','all_five_selection_unchanged':all(r['same_selected_connections'] for r in fits),'no_inference_or_export_changes':True,'source_sha256':digest(Path(__file__))})

if __name__=='__main__':main()

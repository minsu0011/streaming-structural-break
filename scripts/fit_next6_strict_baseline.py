"""Repair input-completeness of fixed M1 comparator; never change graph candidates."""
from pathlib import Path
import sys,os,gc,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from src.next6.common import BASE,save,read_json,digest,activity,checkpoint,ids_hash,utc_now
from src.next6.data import arrays
from src.next6.strict_baseline import extras,StrictModel
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training

def main():
    lock=BASE/'M1_INPUT_PARITY_CORRECTION_LOCK.json'
    if not lock.exists():
        assert not (BASE/'OUTER_SCORE_RELEASE.json').exists()
        save(lock.name,{'created_utc':utc_now(),'classification':'INPUT_EQUIVALENCE_AUDIT_CORRECTION_BEFORE_OUTER_RELEASE','new_comparator':'M1_STRICT',
            'original_M1_retained':'245 continuous tabular features; graph categorical states and explicit time were not direct inputs',
            'inputs':'original245 +current16 states+5x16 lagged states+2x16 categorical EWMA(8,32)+16 own state log run lengths(capped4096)+log1p(current_time)/8 =390',
            'same_graph_input_information_available':True,'no_cross_node_edges_or_transition_tables':True,'state_EWMA_initialization':'zero, lag ring from H final16, own run length zero',
            'learner':'unchanged fixed100 trees,15 leaves,maxdepth4,W3,training-only constant columns; no tuning','outer_fits':5,'inner_fits':0,
            'inner_fits_reason':'fixed nonselected comparator; M4 eligibility depends only on its own M2/M3 comparisons and is unchanged',
            'learned_recipes_total_including_controls':8,'candidate_families':3,'no_existing_candidate_refits':True,
            'decision':'original gates retained; additionally require delta>=.0002 and paired CI lower>0 versus M1_STRICT; never relax any gate',
            'source_sha256':{p:digest(ROOT/p) for p in ('src/next6/strict_baseline.py','scripts/fit_next6_strict_baseline.py')}})
    else:
        for name,sha in read_json(lock)['source_sha256'].items():assert digest(ROOT/name)==sha
    assert not (BASE/'OUTER_SCORE_RELEASE.json').exists(),'Freeze strict comparator before opening scores'
    data,meta,v=arrays();d=BASE/'strict_baseline';d.mkdir(exist_ok=True);path=d/'extras.npy';n=len(meta['target'])
    if not (d/'EXTRAS_DONE.json').exists():
        with activity('input_complete_nongraph_features','CONTROL_RESEARCH'):
            out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,145))
            for ix,sid in enumerate(data.ids):
                a,b=data.series['online_offsets'][ix:ix+2];out[a:b]=extras(v['states'][a:b],v['H_states'][ix].copy(),np.zeros((2,16)),np.zeros(16),np.zeros(1))
                if ix%1000==0:checkpoint('strict control inputs',str(ix)+'/6400')
            out.flush();save('strict_baseline/EXTRAS_DONE.json',{'sha256':digest(path),'columns':145,'row_count':n});del out
    extra=np.load(path,mmap_mode='r')
    for fold in range(5):
        directory=d/f'fold_{fold}';directory.mkdir(exist_ok=True)
        if (directory/'DONE.json').exists():continue
        assert (BASE/f'fold_{fold}/OUTER_PREDICTIONS_FROZEN.json').exists()
        tr=np.flatnonzero(meta['fold']!=fold);va=np.flatnonzero(meta['fold']==fold);data.guard.partition(np.unique(meta['dataset_id'][tr]),np.unique(meta['dataset_id'][va]))
        with activity('fit_input_complete_M1_fold'+str(fold),'CONTROL_RESEARCH'):
            # One training matrix only; avoid a second multi-GB retained-column copy.
            spread=np.zeros(390)
            low=np.full(390,np.inf);high=np.full(390,-np.inf)
            for a in range(0,len(tr),50000):
                rows=tr[a:a+50000];z=np.column_stack((v['tabular'][rows],extra[rows]));low=np.minimum(low,z.min(axis=0));high=np.maximum(high,z.max(axis=0))
            cols=np.flatnonzero(high-low>1e-12).astype(np.int32);x=np.empty((len(tr),len(cols)),dtype=np.float32)
            for a in range(0,len(tr),50000):
                rows=tr[a:a+50000];z=np.column_stack((v['tabular'][rows],extra[rows]));x[a:a+len(rows)]=z[:,cols]
            weight=weights_from_training(meta['target'][tr],meta['time_online'][tr],meta['dataset_id'][tr],'W3');est=make_estimator('M4_lightgbm',read_json(BASE/'NEXT6_PROTOCOL_LOCK.json')['learner_params'])
            with threadpool_limits(limits=4):est.fit(x,meta['target'][tr],sample_weight=weight)
            compact=export_lightgbm(est)
            with threadpool_limits(limits=1):native=est.booster_.predict(x[:2048],num_threads=1)
            assert np.max(abs(native-compact.predict(x[:2048])))<1e-12
            del x,est;gc.collect()
            model=StrictModel(cols,compact,{'training_series':5120,'training_ids_sha256':ids_hash(np.unique(meta['dataset_id'][tr])),'weighting':'W3 training only','constant_columns_training_only':True},
                {p:digest(ROOT/p) for p in ('src/next6/nodes.py','src/next6/strict_baseline.py','artifacts/next6/M1_INPUT_PARITY_CORRECTION_LOCK.json')})
            assert not (directory/'model.joblib').exists();joblib.dump(model,directory/'model.joblib');prediction=np.empty(len(va))
            for a in range(0,len(va),50000):
                r=va[a:a+50000];prediction[a:a+len(r)]=model.predict(np.column_stack((v['tabular'][r],extra[r])))
            np.savez(directory/'predictions.npz',global_row=va,dataset_id=meta['dataset_id'][va],prediction=prediction)
            save(str((directory/'DONE.json').relative_to(BASE)),{'status':'FROZEN_NOT_SCORED','fold':fold,'method':'M1_STRICT','columns':390,'retained_columns':len(cols),
                'training_ids_sha256':model.training['training_ids_sha256'],'model_sha256':digest(directory/'model.joblib'),'prediction_sha256':digest(directory/'predictions.npz'),
                'score_opened':False,'equal_learner_budget':True})
    save('M1_STRICT_PREDICTIONS_FROZEN.json',{'status':'FROZEN_BEFORE_ANY_OUTER_SCORE','created_utc':utc_now(),'folds':5,'no_selection_or_tuning':True})

if __name__=='__main__':main()

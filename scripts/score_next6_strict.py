"""Additional stricter input-parity comparison, conditional on frozen predictions."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np
from numba import set_num_threads
from src.next6.common import BASE,read_json,save,digest,activity,checkpoint
from src.next6.data import arrays
from src.next6.evaluation import csv_write,frozen_predictions
from src.next5.bootstrap import PreparedAUC,paired_summary
from src.scoring.ts_auc import ts_auc

def main():
    assert (BASE/'M1_STRICT_PREDICTIONS_FROZEN.json').exists();assert (BASE/'OUTER_SCORES.json').exists()
    data,meta,_=arrays();p=np.empty(len(meta['target']));score=[];boot=np.empty((1001,5));rng=np.random.default_rng(20260913);set_num_threads(min(8,os.cpu_count() or 1))
    with activity('strict_input_control_scoring_and_uncertainty','UNCERTAINTY'):
        for f in range(5):
            d=BASE/f'strict_baseline/fold_{f}';r=read_json(d/'DONE.json');assert digest(d/'predictions.npz')==r['prediction_sha256']
            with np.load(d/'predictions.npz',allow_pickle=False) as z:
                data.guard.admit(np.unique(z['dataset_id']),purpose='strict input comparator evaluation');rows=z['global_row'];assert np.array_equal(meta['dataset_id'][rows],z['dataset_id']);assert np.all(meta['fold'][rows]==f);p[rows]=z['prediction']
            y=meta['target'][rows];t=meta['time_online'][rows];v=p[rows];score.append(float(ts_auc(y,v,t)));ids,codes=np.unique(meta['dataset_id'][rows],return_inverse=True);n=len(ids)
            w=np.vstack((np.ones((1,n),dtype=np.int32),rng.multinomial(n,np.full(n,1/n),size=1000)));boot[:,f]=PreparedAUC.create(y,v,t,codes,n).evaluate(w);assert abs(boot[0,f]-score[-1])<1e-12
            checkpoint('strict paired bootstrap','fold'+str(f))
        np.save(BASE/'evaluation_cache/M1_STRICT_bootstrap.npy',boot,allow_pickle=False)
        scores=read_json(BASE/'OUTER_SCORES.json');scores['M1_STRICT']=score;comparison=[];paired=[]
        for method,vals in scores.items():
            valid=np.array([v is not None for v in vals]);a=np.array([v for v in vals if v is not None]);b=np.array(scores['M0'])[valid];strict=np.array(score)[valid]
            comparison.append({'method':method,'outer_folds':len(a),'mean_TS_AUC':float(a.mean()),'median_TS_AUC':float(np.median(a)),'worst_TS_AUC':float(a.min()),
                'delta_M0':float((a-b).mean()),'delta_input_complete_M1_STRICT':float((a-strict).mean()),
                'M1_original_missing_explicit_state_time_inputs':method=='M1','input_complete_primary_comparator':'M1_STRICT','partial_M4_not_full_candidate':len(a)<5})
            if method!='M1_STRICT':
                original=np.load(BASE/f'evaluation_cache/{method}_bootstrap.npy');summary=paired_summary(boot[:,valid],original[:,valid]);paired.append({'candidate':method,'reference':'M1_STRICT','folds':int(valid.sum()),**summary})
        csv_write('STRICT_INPUT_MODEL_COMPARISON_NEXT6.csv',comparison);csv_write('STRICT_INPUT_PAIRED_NEXT6.csv',paired)
        save('M1_STRICT_SCORES_NEXT6.json',{'fold_TS_AUC':score,'mean_TS_AUC':float(np.mean(score)),'input_equivalence_correction_before_outer_release':True,
            'comparison_label':'M1_STRICT is the input-complete nongraph comparator; original M1 is a continuous-only ablation',
            'candidate_models_or_M4_eligibility_changed':False,'additional_gate_is_stricter_never_weaker':True})

if __name__=='__main__':main()

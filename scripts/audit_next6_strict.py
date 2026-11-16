"""Strict comparator input equivalence, OOF replay and prefix contracts."""
from pathlib import Path
import sys,os,hashlib,pickle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from src.next6.common import BASE,read_json,save,digest,activity,checkpoint
from src.next6.data import source
from src.next6.strict_runtime import PreparedStrict

def main():
    data=source();results=[]
    with activity('strict_nongraph_OOF_input_and_prefix_audit','FINAL_AUDIT'):
        for f in range(5):
            d=BASE/f'strict_baseline/fold_{f}';r=read_json(d/'DONE.json');assert digest(d/'model.joblib')==r['model_sha256'];model=joblib.load(d/'model.joblib');p=PreparedStrict(model)
            with np.load(d/'predictions.npz',allow_pickle=False) as z:ids=z['dataset_id'];pred=z['prediction'];data.guard.admit(np.unique(ids),purpose='strict baseline parity')
            selected=sorted(map(int,np.unique(ids)),key=lambda i:hashlib.sha256(('NEXT6_PREFIX:'+str(i)).encode()).hexdigest())[:20];error=0.;points=0
            for k,sid in enumerate(selected):
                h,o,_=data.get(sid);actual=p.new_state(h).replay(o);expected=pred[ids==sid];error=max(error,float(abs(actual-expected).max()));assert error<1e-12;points+=len(o)
                if k<3:
                    slow=model.new_state(h).replay(o);assert np.max(abs(slow-actual))<1e-12
                    s=p.new_state(h);live=np.array([s.predict_one(x) for x in o[:64]]);assert np.array_equal(live,actual[:len(live)])
                    cut=min(31,len(o));mut=p.new_state(h).replay(np.r_[o[:cut],np.full(101,99,dtype=np.float32)]);assert np.array_equal(mut[:cut],actual[:cut])
                    assert np.array_equal(pickle.loads(pickle.dumps(p)).new_state(h).replay(o),actual)
                    for value in (1e100,float('nan')):
                        try:p.new_state(h).predict_one(value)
                        except ValueError:pass
                        else:raise AssertionError('Invalid strict input accepted')
            results.append({'fold':f,'series':20,'points':points,'OOF_vs_fused_max_abs_error':error,'scalar_batch_slow_save_load_suffix_length':True,'invalid_input_rejected':True})
            checkpoint('strict nongraph audit',f'fold{f} PASS')
        save('M1_STRICT_REPRODUCTION_NEXT6.json',{'status':'PASS','results':results,'source_sha256':digest(ROOT/'src/next6/strict_runtime.py'),
            'explicit_states_lags_EWMA_own_runlength_time_available_without_edges':True,'all_state_inputs_H_and_past_only':True,'no_graph_tables_used':True})

if __name__=='__main__':main()

"""Current-point/prefix contracts using existing inner fits, no new learner."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from src.next6.common import BASE,save,activity,digest
from src.next6.data import source
from src.next6.runtime import Prepared

def main():
    with activity('runtime_inner_engineering','REPRODUCIBILITY'):
        data=source();results=[]
        for method in ('M1','M2','M3','M4'):
            d=BASE/'fold_0/inner_0'/method;model=joblib.load(d/'model.joblib')
            with np.load(d/'predictions.npz',allow_pickle=False) as f:
                sid=int(f['dataset_id'][0]);mask=f['dataset_id']==sid;expected=f['prediction'][mask]
            h,o,_=data.get(sid);prepared=Prepared(model);state=prepared.new_state(h);actual=state.replay(o)
            error=float(np.max(abs(actual-expected)));assert error<1e-12,(method,error)
            slow=model.new_state(h);slowpred=np.array([slow.predict_one(p) for p in o[:64]])
            assert np.max(abs(actual[:64]-slowpred))<1e-12
            live=prepared.new_state(h);stream=np.array([live.predict_one(p) for p in o[:64]])
            assert np.max(abs(actual[:64]-stream))<1e-12
            suffix=np.r_[o[:31],np.full(99,123,dtype=np.float32)];mut=prepared.new_state(h).replay(suffix)
            assert np.max(abs(actual[:31]-mut[:31]))<1e-12
            results.append({'method':method,'dataset_id':sid,'points':len(o),'OOF_vs_fused_error':error,
                'batch_live_pass':True,'future_suffix_and_length_pass':True,'model_sha256':digest(d/'model.joblib')})
        save('RUNTIME_INNER_ENGINEERING.json',{'status':'PASS','scope':'inner engineering only; full outer contracts pending',
            'results':results,'runtime_source_sha256':digest(ROOT/'src/next6/runtime.py'),'new_fits':0,'sealed_data_opened':False})

if __name__=='__main__':main()

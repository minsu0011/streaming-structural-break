"""Portable verification; explicitly rejects raw/array dataset file access."""
from pathlib import Path
import sys,os,json,time,importlib.metadata,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'artifacts/next6_extension/numba_cache'))
blocked=[]
def file_guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]))
        if p.suffix.lower() in ('.parquet','.npy','.npz'):
            blocked.append(str(p));raise RuntimeError('Portable synthetic verification forbids raw/array dataset access')
sys.addaudithook(file_guard)
import numpy as np
from src.next6_extension.observer import load
from src.next6_extension.validated_runtime import Prepared
from src.next6_extension.synthetic import generated

def main():
    lock=json.loads((ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json').read_text(encoding='utf-8'));results=[];start=time.perf_counter()
    for fold in range(5):
        model=load(ROOT/f'artifacts/next6_extension/observable_models/fold_{fold}/model.joblib');prepared=Prepared(model)
        for case in ('D2','O1','V4'):
            h,o,tau,info=generated(case,0,partition=31,horizon=1024,control=False,lock=lock);state=prepared.new_state(h);batch=state.replay(o);scalar=prepared.new_state(h);actual=np.array([scalar.predict_one(x) for x in o]);np.testing.assert_array_equal(actual,batch);np.testing.assert_allclose(model.new_state(h).replay(o),batch,rtol=1e-12,atol=1e-12);assert np.isfinite(batch).all();alarm=np.flatnonzero(batch[:,1]>model.threshold)
            results.append({'fold':fold,'case':case,'points':len(o),'scalar_batch_max_error':float(np.max(np.abs(actual-batch))),'first_persistent':int(alarm[0]) if len(alarm) else None,'final_state':int(batch[-1,0]),'final_normal_transition_probability':float(batch[-1,20]),'final_post_transition_probability':float(batch[-1,21])})
        invalid=prepared.new_state(np.ones(1024));out=invalid.replay(np.full(256,np.finfo(np.float32).max));assert np.all(out[:,0]==6) and np.all(out[:,1]==0) and np.all(out[:,11]==0) and np.all(out[:,24]==0)
    imported={name:Path(module.__file__).resolve().as_posix() for name,module in sys.modules.items() if (name=='src' or name.startswith('src.')) and getattr(module,'__file__',None)}
    for name,path in imported.items():assert Path(path).is_relative_to(ROOT),'Source import escaped portable root: '+name
    assert not blocked
    print(json.dumps({'status':'PASS','python':platform.python_version(),'platform':platform.platform(),'source_root':str(ROOT),'all_source_imports_within_copy':True,'source_modules':imported,'raw_or_array_dataset_open_attempts':blocked,'new_environment_packages':{p:importlib.metadata.version(p) for p in ('numpy','numba','llvmlite','joblib','cloudpickle','psutil')},'results':results,'insufficient_reference_all_folds':'PASS','elapsed_seconds_including_initial_JIT':time.perf_counter()-start},indent=2))
if __name__=='__main__':main()

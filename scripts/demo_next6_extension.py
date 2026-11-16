"""Portable synthetic demonstration; does not read ResearchPool or hidden data."""
from pathlib import Path
import sys,os,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'artifacts/next6_extension/numba_cache'))
import numpy as np
from src.next6_extension.observer import load,COLUMNS
from src.next6_extension.validated_runtime import Prepared
from src.next6_extension.fresh_state import STATES
from src.next6_extension.synthetic import generated,CASES

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,default='D2');p.add_argument('--fold',type=int,choices=range(5),default=0);p.add_argument('--seed',type=int,default=0);p.add_argument('--horizon',type=int,default=1024);a=p.parse_args()
    model=load(ROOT/f'artifacts/next6_extension/observable_models/fold_{a.fold}/model.joblib');lock=json.loads((ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json').read_text(encoding='utf-8'));h,o,tau,info=generated(a.case,a.seed,partition=31,horizon=a.horizon,control=False,lock=lock)
    prepared=Prepared(model);batch=prepared.new_state(h).replay(o);state=prepared.new_state(h);scalar=np.array([state.predict_one(x) for x in o]);np.testing.assert_array_equal(batch,scalar)
    reference=model.new_state(h).replay(o);np.testing.assert_allclose(batch,reference,rtol=1e-12,atol=1e-12)
    alarm=np.flatnonzero(batch[:,1]>model.threshold);end=batch[-1]
    print(json.dumps({'status':'PASS','case':a.case,'points':len(o),'batch_scalar_max_error':float(np.abs(batch-scalar).max(initial=0)),'first_persistent_observation':int(alarm[0]) if len(alarm) else None,
        'final_state':STATES[int(end[0])],'normal_transition_probability':float(end[20]),'post_transition_probability':float(end[21]),'both_conditions_supported':bool(end[25]),
        'bounded_mutable_state_bytes':state.bounded_mutable_state_bytes,'interpretation':'Descriptive state transitions, not a calibrated probability of any past break. tau never passed to the model.'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

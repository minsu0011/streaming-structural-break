"""Portable synthetic scalar demonstration. Never accesses competition data."""
from pathlib import Path
import sys,os,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from src.next6.runtime import Prepared

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['M1','M2','M3'],default='M3');p.add_argument('--fold',type=int,choices=range(5),default=0);args=p.parse_args()
    model=joblib.load(ROOT/f'artifacts/next6/fold_{args.fold}/outer/{args.method}/model.joblib');detector=Prepared(model)
    rng=np.random.default_rng(20260913);historical=rng.normal(size=512).astype(np.float32);online=rng.normal(size=256).astype(np.float32);online[128:]*=1.5
    state=detector.new_state(historical);actual=np.array([state.predict_one(float(point)) for point in online]);batch=detector.new_state(historical).replay(online)
    assert np.array_equal(actual,batch)
    print({'status':'PASS','synthetic_points':len(actual),'method':args.method,'fold':args.fold,'current_point_batch_parity':True,
        'score_range':[float(actual.min()),float(actual.max())],'scope':'DEV research artifact; no champion replacement'},flush=True)

if __name__=='__main__':main()

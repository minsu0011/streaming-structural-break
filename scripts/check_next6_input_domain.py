"""Engineering probe of batch/scalar float32 input domain consistency."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from src.next6.runtime import Prepared
from src.next6.common import BASE,save,activity

def main():
    with activity('runtime_float32_domain_probe','REPRODUCIBILITY'):
        model=joblib.load(BASE/'fold_0/inner_0/M3/model.joblib');p=Prepared(model);r=np.random.default_rng(6);h=r.normal(size=320).astype(np.float32)
        out=[]
        for value in (1e100,1.00000006):
            row={'value':value}
            for api in ('scalar','batch'):
                try:
                    s=p.new_state(h);v=s.predict_one(value) if api=='scalar' else s.replay(np.array([value],dtype=np.float64))[0];row[api]={'accepted':True,'prediction':float(v)}
                except (ValueError,OverflowError) as e:row[api]={'accepted':False,'error':type(e).__name__}
            out.append(row)
        same=all(r['scalar']['accepted']==r['batch']['accepted'] for r in out)
        save('RUNTIME_INPUT_DOMAIN_RECHECK.json',{'results':out,'same_domain':same,'status':'PASS' if same else 'FAIL'})
        assert same
        print(out,flush=True)

if __name__=='__main__':main()

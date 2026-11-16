"""Execute unchanged official scorer on >=100 randomized synthetic cases."""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.scoring.ts_auc import score_frames
from src.utils.artifacts import write_json,sha256

def load_official():
    spec=importlib.util.spec_from_file_location('official_scoring',ROOT/'vendor/official/scoring.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def official_score(module,pred,target):
    old_p,old_y=module._load_prediction,module._load_y_test
    try:
        module._load_prediction=lambda _:pred.copy()
        module._load_y_test=lambda _:target.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            result=module.score('unused','unused',[(SimpleNamespace(),[SimpleNamespace(name='ts-auc',id='metric')])])['metric']
        return float(result.value)
    finally:
        module._load_prediction,module._load_y_test=old_p,old_y

def main():
    official=load_official()
    rng=np.random.default_rng(20260908)
    errors=[]
    raw_shuffle_deltas=[]
    for case in range(128):
        keys=[]; labels=[]; scores=[]
        for sid in range(int(rng.integers(3,25))):
            size=int(rng.integers(1,32))
            tau=int(rng.integers(0,size+1))
            for t in range(size):
                keys.append((sid,1000+sid*7+t))
                labels.append(int(t>=tau))
                scores.append(float(rng.integers(0,7))/6)
        index=pd.MultiIndex.from_tuples(keys,names=['id','time'])
        y=pd.DataFrame({'target':labels},index=index)
        p=pd.DataFrame({'prediction':scores},index=index)
        if case % 8 == 0: p['prediction']=0.5
        if case % 8 == 1: p['prediction']=y.target.astype(float)
        if case % 8 == 2: p['prediction']=1-y.target.astype(float)
        if case % 8 == 3: y['target']=0
        if case % 8 == 4: y['target']=1
        local=score_frames(p,y)
        ref=official_score(official,p.sort_index(),y.sort_index())
        errors.append(abs(local-ref))
        shuffled=p.sample(frac=1,random_state=case)
        assert score_frames(shuffled,y.sample(frac=1,random_state=case+1))==local
        errors.append(abs(score_frames(shuffled,y)-official_score(official,shuffled.sort_index(),y.sort_index())))
        raw_shuffle_deltas.append(abs(local-official_score(official,shuffled,y)))
    report={'status':'PASS' if max(errors)<1e-12 else 'FAIL','random_cases':128,'canonical_parity_comparisons':len(errors),
        'max_abs_error':max(errors),'tolerance':1e-12,'official_source_sha256':sha256(ROOT/'vendor/official/scoring.py'),
        'official_code_executed':True,'io_only_monkeypatched':True,'raw_official_unsorted_max_delta':max(raw_shuffle_deltas),
        'ordering_policy':'sort (id,time) before both scorers; raw official score() trusts row order',
        'data_scope':'SYNTHETIC_ENGINEERING_ONLY'}
    write_json(ROOT/'artifacts/validation/SCORER_PARITY.json',report)
    print(report)
    assert report['status']=='PASS'

if __name__=='__main__':
    main()

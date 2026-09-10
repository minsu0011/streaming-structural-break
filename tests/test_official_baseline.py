import ast
import json
import math
import os
from pathlib import Path
from typing import Iterable,Tuple,List
import joblib
import numpy as np
from src.models.baselines import OfficialEWMA

def test_exact_official_baseline(tmp_path):
    notebook=json.loads((Path(__file__).resolve().parents[1]/'vendor/official/baseline.ipynb').read_text(encoding='utf-8'))
    code=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('def infer('))
    namespace={'np':np,'math':math,'os':os,'joblib':joblib,'Iterable':Iterable,'Tuple':Tuple,'List':List}
    exec(compile(ast.parse(code),'<pinned_official_baseline>','exec'),namespace)
    joblib.dump({},tmp_path/'model.joblib')
    rng=np.random.default_rng(20260908)
    for h in [[],[0],[1,1,1],rng.normal(size=300),rng.normal(size=1000)*3+12]:
        online=rng.normal(size=100)
        output=list(namespace['infer'](iter([(h,iter(online))]),str(tmp_path)))
        assert output[0] is None
        state=OfficialEWMA(h)
        np.testing.assert_array_equal(output[1:],[state.predict_one(x) for x in online])

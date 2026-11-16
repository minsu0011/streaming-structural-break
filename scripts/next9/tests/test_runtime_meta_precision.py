import src.next9
import numpy as np
from types import SimpleNamespace
from src.next9 import streaming
from src.next9.runtime_heads_v3 import FrozenHeads
from src.next9.meta_correction import OffsetCorrection
from src.next9.learning import Classifier

def test_qualified_M7_shared_dedicated_and_batch_dtype_boundaries_identical(monkeypatch):
    monkeypatch.setattr(streaming,'assemble',lambda *args:(np.zeros((1,1)),('x',)))
    p=np.float32(.73123456)
    model=Classifier(1,np.array([0]),np.array([0.]),np.array([1.]),2,None,np.array([1-float(p),float(p)]),{'method':'LOGISTIC'})
    correction=OffsetCorrection(.02,.37,.85,-.1,{})
    spec=dict(candidates=[],rule_specs={},run_maps={},input_names=['x'],selected_columns={'M3':[0]},aliases={f'M{k}':'M3' for k in range(1,6)})
    args=(spec,{'M3':model},{},{},None,None,correction)
    original=streaming.FrozenHeads(*args);repaired=FrozenHeads(*args);bank=SimpleNamespace(time=0);b=.6123456789123
    old=original.predict(bank,b,include_tabular=False)['M7']
    expected=float(correction.predict(np.array([b]),np.array([p],dtype=np.float32))[0])
    assert old!=expected  # Reproduce the actual precision mismatch before repair.
    assert repaired.predict(bank,b,include_tabular=False)['M7']==expected
    assert repaired.predict_M7(bank,b)==expected

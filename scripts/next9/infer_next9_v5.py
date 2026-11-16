"""Research demonstration entry: H initialization then scalar-only Online input.

The prospective demo uses outer0 NORMAL, not the best-scoring fold and not a
new canonical deployment or an ensemble. M7 falls back exactly to its matched
excluded-group BL80 when its original inner qualification gate did not pass.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,digest


def infer(data,model_directory):
    if Path(model_directory).resolve()!=BASE.resolve():raise ValueError('Use the hash-bound NEXT9 run directory in this payload')
    config=read_json(BASE/'DEMO_INFERENCE_V5_NEXT9.json')
    assert config['fold']==0 and config['mode']=='NORMAL' and config['canonical_role_changed'] is False
    from src.next9.runtime_loading_v5 import load_heads,load_baseline
    from src.next9.integrated_selected import IntegratedStream
    heads=load_heads(config['fold'],config['mode']);baseline=load_baseline(config['fold'],config['mode'])
    for historical,online in data:
        state=IntegratedStream(heads,baseline,historical)
        yield None
        for point in online:yield state.predict_one(point)

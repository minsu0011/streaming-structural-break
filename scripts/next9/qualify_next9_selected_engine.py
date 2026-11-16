"""Actual frozen-model parity before admitting exact schema compilation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


def main():
    import copy
    import time
    import numpy as np
    from src.next9.runtime_loading_v4 import load_heads,load_baseline
    from src.next9.runtime_heads_v6 import FrozenHeads
    from src.next9.streaming import ResearchStream
    from src.next9.integrated_selected import IntegratedStream
    from src.next9.negative_controls_v2 import generate,SEEDS
    results=[]
    with activity('Verify selected-engine M7 against full reference on all ten frozen procedures','RUNTIME_QUALIFICATION','B9_NORMALIZATION_STATE_AVAILABILITY'):
        for fold in range(5):
            for mode in ['NORMAL','EARLY']:
                heads=load_heads(fold,mode);baseline=load_baseline(fold,mode)
                fast=FrozenHeads(heads.spec,heads.models,heads.self_laws,heads.cross_laws,heads.m6,heads.tabular,heads.correction)
                for background,length in [('GAUSSIAN',159),('GAUSSIAN',512),('AR_0.8',2048)]:
                    h,o,_=generate(background,'NONE',SEEDS[0],max(512,length))
                    h=h[-length:]
                    reference=ResearchStream(heads,h);optimized=ResearchStream(fast,h)
                    dedicated=IntegratedStream(fast,baseline,h);base=baseline.new_state(h)
                    timings=[0.,0.]
                    for t,point in enumerate(o):
                        b=base.predict_one(float(point))
                        start=time.perf_counter();expected=reference.predict_one(point,b,include_tabular=False);timings[0]+=time.perf_counter()-start
                        start=time.perf_counter();actual=optimized.predict_one(point,b,include_tabular=False);timings[1]+=time.perf_counter()-start
                        assert actual==expected,(fold,mode,background,t,actual,expected)
                        assert dedicated.predict_one(float(point))==expected['M7']
                        if t==31:dedicated=IntegratedStream.restore(dedicated.snapshot())
                    results.append(dict(fold=fold,mode=mode,background=background,H_length=length,points=len(o),
                        all_shared_heads_bitwise=True,dedicated_M7_bitwise=True,reference_seconds=timings[0],optimized_seconds=timings[1]))
                    print(results[-1],flush=True)
        atomic(BASE/'SELECTED_ENGINE_QUALIFICATION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),results=results,
            source_sha256={p:digest(ROOT/p) for p in ['src/next9/fast_design_v2.py','src/next9/runtime_heads_v6.py','src/next9/integrated_selected.py']},
            no_new_fit=True,not_latency_gate_measurement=True))


if __name__=='__main__':main()

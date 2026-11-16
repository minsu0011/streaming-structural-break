"""Exact dedicated M7 for frozen empty-rule candidates in the first engine."""
import numpy as np
from types import SimpleNamespace
from src.next9.streaming import IntegratedStream as ReferenceIntegratedStream,ResearchStream,PrefixBank


class SelectedFeatureState:
    def __init__(self,original,candidates):
        self.columns=np.asarray(candidates,dtype=np.int64)
        self.engine=original.engines[0] if original.engines else None
        self.center=original.center[self.columns].copy()
        self.spread=original.spread[self.columns].copy()

    def update(self,point):
        value=np.zeros(153,dtype=np.float32)
        if self.engine is not None:
            raw=self.engine.update(float(point))
            value[self.columns]=np.clip((raw[self.columns].astype(np.float64)-self.center)/self.spread,-12,12).astype(np.float32)
        return (value,)


class IntegratedStream(ReferenceIntegratedStream):
    def __init__(self,heads,baseline,historical):
        self.baseline=baseline.new_state(historical);self.research=None
        if heads.correction is None:return
        candidates=heads.spec['candidates']
        if heads.spec['rule_specs'] or heads.cross_laws or not candidates or not all(0<=j<93 for j in candidates):
            self.research=ResearchStream(heads,historical)
            return
        from src.next8.features import FeatureState
        original=FeatureState(historical)
        self.research=SimpleNamespace(heads=heads,features=SelectedFeatureState(original,candidates),
            bank=PrefixBank(original.history,original.epsilon,original.context,len(historical)))

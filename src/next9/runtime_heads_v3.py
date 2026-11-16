"""Preserve the fitted float32 M5 boundary before the fixed M7 correction."""
import numpy as np
from src.next9.streaming import FrozenHeads as OriginalFrozenHeads

class FrozenHeads(OriginalFrozenHeads):
    def predict(self,bank,baseline=None,include_tabular=True):
        result=super().predict(bank,baseline=None,include_tabular=include_tabular)
        if baseline is not None:
            result['M0']=float(baseline)
            result['M7']=float(baseline) if self.correction is None else float(self.correction.predict(
                np.asarray([baseline]),np.asarray([result['M5']],dtype=np.float32))[0])
        return result

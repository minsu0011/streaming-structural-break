"""Keep the unchanged kernel arguments resident in a typed in-memory object.

Numba boxes one object per call instead of unboxing both nested array tuples.
Only mutable online state is resident; no online iterable is buffered. The
owning Python detector keeps all original state/tree arrays visible to audits.
No jitclass is serialized or cached to disk; each process warms before ready.
"""
from functools import lru_cache
from numba import typeof
from numba.experimental import jitclass
from src.streaming.fused_cross_order import fused_equal_average


@lru_cache(maxsize=8)
def _resident_class(argument_type):
    @jitclass([('args',argument_type)])
    class ResidentCrossOrder:
        def __init__(self,args):
            self.args=args
        def predict_one(self,point):
            return fused_equal_average(point,self.args[0],self.args[1])
    return ResidentCrossOrder


def resident_state(call_args):
    return _resident_class(typeof(call_args))(call_args)

"""Exact float64 ties for NEXT5 OOF, with the reviewed weighted rank kernel."""
import numpy as np
from src.next.bootstrap import PreparedAUC as LegacyPreparedAUC,paired_summary


class PreparedAUC(LegacyPreparedAUC):
    @classmethod
    def create(cls,target,prediction,time_online,series_code,n_groups):
        # NEXT5 stores float64 compact-tree probabilities. Do not silently merge
        # distinct scores by downcasting to the legacy float32 OOF convention.
        y=np.asarray(target);p=np.asarray(prediction,dtype=np.float64)
        t=np.asarray(time_online);g=np.asarray(series_code)
        if any(a.ndim!=1 or a.shape!=y.shape for a in (p,t,g)):raise ValueError('Equal length row vectors required')
        if not len(y) or not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or not np.isfinite(t).all():raise ValueError('Finite scores/times and binary targets required')
        if np.any(t<0) or np.any(t!=np.floor(t)) or np.any(g!=np.floor(g)) or g.min()<0 or g.max()>=n_groups:
            raise ValueError('Valid integer time and group coordinates required')
        order=np.lexsort((p,t));sy=y[order].astype(np.uint8);sp=p[order];st=t[order];sg=g[order].astype(np.int32)
        change=st[1:]!=st[:-1]
        time_ends=np.r_[np.flatnonzero(change)+1,len(y)].astype(np.int32)
        tie_ends=np.r_[np.flatnonzero(change|(sp[1:]!=sp[:-1]))+1,len(y)].astype(np.int32)
        return cls(sy,sg,time_ends,tie_ends,n_groups)

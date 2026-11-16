"""Offline reference inputs for independent finite-prefix runtime checks."""
import numpy as np
from src.next9.states import describe_history


class ArrayBank:
    def __init__(self,seed,values,epsilon,context,H_length):
        self.full=np.vstack([np.asarray(seed,dtype=np.float32),np.asarray(values,dtype=np.float32)])
        self.times=np.arange(len(values),dtype=np.int64);self.series_index=np.zeros(len(values),dtype=np.int32)
        self.series=dict(epsilon=np.asarray(epsilon,dtype=np.float32)[None],context=np.asarray(context,dtype=np.float32)[None],
            H_length=np.asarray([H_length],dtype=np.int64))

    def history(self,rows,j):
        rows=np.asarray(rows,dtype=np.int64)
        return np.stack([self.full[r:r+16,j] for r in rows]) if len(rows) else np.empty((0,16),dtype=np.float32)

    def current(self,rows,columns):
        return self.full[np.asarray(rows)[:,None]+16,np.asarray(columns)[None,:]]

    def pre_state(self,rows,j):
        return describe_history(self.history(rows,j),self.series['epsilon'][0,0,j],self.series['H_length'][0]>=160)

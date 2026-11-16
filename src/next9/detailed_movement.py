"""Registered D2/D3/D5 extensions, preserving the original D1 target recipe."""
from dataclasses import dataclass
import hashlib
import numpy as np
from src.next9.movement_screen import feature_inputs,group_weights
from src.next8.features import mechanical_controls


@dataclass
class Forecast:
    width: int
    columns: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    training: dict

    def predict(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=self.width or not np.isfinite(x).all():
            raise ValueError('Frozen finite forecast schema required')
        return ((x[:,self.columns]-self.center)/self.scale)@self.coefficient+self.intercept

    def record(self):
        return dict(width=self.width,columns=self.columns.tolist(),center=self.center.tolist(),scale=self.scale.tolist(),coefficient=self.coefficient.tolist(),
            intercept=self.intercept.tolist(),training=self.training)


def design(bank,rows,j,states,method):
    if method not in ('D2','D3','D4','D5'):
        raise ValueError('Only precommitted detailed extensions')
    own=feature_inputs(bank,rows,j,states)
    context=bank.series['context'][bank.series_index[rows]]
    parts=[own,context]
    names=['own_'+str(k) for k in range(9)]+['H_'+str(k) for k in range(6)]
    if method!='D2':
        parts.append(np.log1p(bank.times[rows])[:,None]);names.append('observed_log_time')
    if method=='D5':
        for parent in mechanical_controls(j):
            history=bank.history(rows,parent)
            parts.extend([bank.current(rows,[parent]),history[:,[-1,-4,-16]]])
            names.extend([f'mechanical_F{parent}_current',*[f'mechanical_F{parent}_lag{k}' for k in [1,4,16]]])
    x=np.column_stack(parts).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite H/mechanical context')
    return x,names


def fit(x,y,ids,groups,scope,names,method):
    scope.admit_training(ids)
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.shape!=(len(ids),len(names)) or y.shape!=(len(ids),15) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Exact admitted multioutput forecast arrays')
    w=group_weights(groups)
    width=x.shape[1]
    seen=set();columns=[]
    for k in range(width):
        h=hashlib.sha256(x[:,k].tobytes()).hexdigest()
        if np.ptp(x[:,k])>1e-12 and h not in seen:
            seen.add(h);columns.append(k)
    columns=np.asarray(columns,dtype=int)
    x=x[:,columns]
    center=np.average(x,axis=0,weights=w)
    scale=np.maximum(np.sqrt(np.average((x-center)**2,axis=0,weights=w)),1e-6)
    z=(x-center)/scale
    intercept=np.average(y,axis=0,weights=w)
    gram=z.T@(z*w[:,None]);gram.flat[::len(gram)+1]+=10.
    coefficient=np.linalg.solve(gram,z.T@((y-intercept)*w[:,None]))
    return Forecast(width,columns,center,scale,coefficient,intercept,dict(**scope.provenance(),method=method,alpha=10.,names=names,
        exact_duplicate_and_constant_pruning='TRAIN-only, prevents duplicated own lags from changing ridge penalty',
        auxiliary_output_count=15,estimator_fit_calls=1,break_labels_read=False,
        D4_alias_D3='No NEXT8 certified stable context exists'))

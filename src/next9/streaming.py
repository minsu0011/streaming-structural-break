"""Finite-prefix runtime for the frozen NEXT9 numerical input schema."""
from dataclasses import dataclass
import copy
import numpy as np
from scipy.special import expit,softmax
from src.next9.states import describe_history
from src.next9.joint_design import assemble
from src.next9.cross import path_features as cross_path


class _SeriesIndex:
    def __getitem__(self,rows):return np.zeros(np.asarray(rows).shape,dtype=np.int32)


class _ObservedTimes:
    def __getitem__(self,rows):return np.asarray(rows,dtype=np.int64)


class PrefixBank:
    """Seventeen past vectors cover current and previous-issued state laws."""
    def __init__(self,seed,epsilon,context,H_length):
        seed=np.asarray(seed,dtype=np.float32)
        if seed.shape!=(16,153):raise ValueError('Frozen sixteen by153 H seed schema required')
        self.past=np.vstack([seed[:1],seed]).copy()
        self.series=dict(epsilon=np.asarray(epsilon,dtype=np.float32)[None,:,:],
            context=np.asarray(context,dtype=np.float32)[None,:],H_length=np.asarray([H_length],dtype=np.int64))
        self.series_index=_SeriesIndex();self.times=_ObservedTimes();self.time=0;self.value=None;self._states={}

    def _admit(self,rows):
        rows=np.asarray(rows,dtype=np.int64)
        if rows.ndim!=1 or np.any((rows<max(0,self.time-1))|(rows>self.time)):
            raise ValueError('Only current or previous-issued observable state is retained')
        return rows

    def stage(self,value):
        if self.value is not None:raise ValueError('Commit the previous observation first')
        value=np.asarray(value,dtype=np.float32)
        if value.shape!=(153,) or not np.isfinite(value).all():raise ValueError('Finite153 current features required')
        self.value=value.copy()

    def current(self,rows,columns):
        rows=self._admit(rows)
        if self.value is None or np.any(rows!=self.time):raise ValueError('Current feature must be staged at this observed time')
        return np.tile(self.value[np.asarray(columns,dtype=int)],(len(rows),1))

    def history(self,rows,j):
        rows=self._admit(rows)
        if not len(rows):return np.empty((0,16),dtype=np.float32)
        return np.stack([self.past[1 if r==self.time else 0:17 if r==self.time else 16,j] for r in rows])

    def pre_state(self,rows,j):
        rows=self._admit(rows)
        if not len(rows):return describe_history(np.empty((0,16)),np.empty(0),np.empty(0,dtype=bool))
        result=[]
        for row in rows:
            row=int(row)
            if row not in self._states:
                start=1 if row==self.time else 0
                self._states[row]=describe_history(self.past[start:start+16].T,self.series['epsilon'][0,0],self.series['H_length'][0]>=160)
            result.append({k:v[j] for k,v in self._states[row].items()})
        return {k:np.asarray([r[k] for r in result]) for k in result[0]}

    def commit(self):
        if self.value is None:raise ValueError('No current observation to commit')
        self.past[:-1]=self.past[1:];self.past[-1]=self.value;self.value=None;self.time+=1;self._states={}


def classifier_probabilities(model,x):
    """Same frozen transform/model; avoid per-point threadpool discovery."""
    z=model.transform(x)
    if model.estimator is None:return np.tile(model.prior,(len(z),1))
    estimator=model.estimator
    if model.training['method']=='LOGISTIC':
        scores=z@estimator.coef_.T+estimator.intercept_
        if len(estimator.classes_)==2:
            p=expit(scores[:,0]);p=np.column_stack([1-p,p])
        else:p=softmax(scores,axis=1)
    else:p=estimator.predict_proba(z)
    out=np.full((len(z),model.classes),1e-8)
    out[:,estimator.classes_.astype(int)]=p
    return out/out.sum(axis=1,keepdims=True)


@dataclass
class FrozenHeads:
    spec: dict
    models: dict
    self_laws: dict
    cross_laws: dict
    m6: object
    tabular: object
    correction: object

    def predict(self,bank,baseline=None,include_tabular=True):
        s=self.spec;rows=np.asarray([bank.time],dtype=np.int64)
        rules={int(k):v for k,v in s['rule_specs'].items()}
        run_maps={int(k):np.asarray(v,dtype=int) for k,v in s['run_maps'].items()}
        x,names=assemble(bank,rows,s['candidates'],rules,run_maps,self.self_laws,'M5')
        if tuple(s['input_names'])!=names:raise ValueError('Frozen online M input schema changed')
        values={method:float(np.float32(classifier_probabilities(model,x[:,s['selected_columns'][method]])[0,1]))
                for method,model in self.models.items()}
        result={method:values[alias] for method,alias in s['aliases'].items()}
        parts=[x]
        for j,law in self.cross_laws.items():parts.append(cross_path(bank,rows,law))
        cx=np.column_stack(parts).astype(np.float32)
        result['M6']=result['M5'] if self.m6 is None else float(np.float32(classifier_probabilities(self.m6,cx)[0,1]))
        if include_tabular:result['TABULAR_SAME_INPUT']=float(np.float32(classifier_probabilities(self.tabular,cx)[0,1]))
        if baseline is not None:
            result['M0']=float(baseline)
            result['M7']=float(baseline) if self.correction is None else float(self.correction.predict(np.asarray([baseline]),np.asarray([result['M5']]))[0])
        return result

    def predict_M7(self,bank,baseline):
        if self.correction is None:return baseline
        s=self.spec;rows=np.asarray([bank.time],dtype=np.int64)
        rules={int(k):v for k,v in s['rule_specs'].items()}
        mapping={int(k):np.asarray(v,dtype=int) for k,v in s['run_maps'].items()}
        x,names=assemble(bank,rows,s['candidates'],rules,mapping,self.self_laws,'M5')
        if tuple(s['input_names'])!=names:raise ValueError('Frozen online M input schema changed')
        method=s['aliases']['M5'];model=self.models[method]
        p=np.float32(classifier_probabilities(model,x[:,s['selected_columns'][method]])[0,1])
        return float(self.correction.predict(np.asarray([baseline]),np.asarray([p]))[0])


class ResearchStream:
    def __init__(self,heads,historical):
        from src.next8.features import FeatureState
        self.heads=heads;self.features=FeatureState(historical)
        self.bank=PrefixBank(self.features.history,self.features.epsilon,self.features.context,len(historical))

    def predict_one(self,point,baseline=None,include_tabular=True):
        value=self.features.update(float(point))[0];self.bank.stage(value)
        result=self.heads.predict(self.bank,baseline,include_tabular)
        self.bank.commit();return result

    def snapshot(self):return copy.deepcopy(self)


class IntegratedStream:
    """Failed inner qualification bypasses every new feature and state update."""
    def __init__(self,heads,baseline,historical):
        self.baseline=baseline.new_state(historical)
        self.research=ResearchStream(heads,historical) if heads.correction is not None else None

    def predict_one(self,point):
        base=self.baseline.predict_one(point)
        if self.research is None:return base
        value=self.research.features.update(float(point))[0];self.research.bank.stage(value)
        result=self.research.heads.predict_M7(self.research.bank,base)
        self.research.bank.commit();return result

    def snapshot(self):
        return dict(kernel=self.baseline.kernel,args=copy.deepcopy(self.baseline.args),
            prepared=self.baseline.prepared,research=copy.deepcopy(self.research))

    @classmethod
    def restore(cls,snapshot):
        from src.next5.runtime import ScalarDetector
        result=cls.__new__(cls)
        result.baseline=ScalarDetector(snapshot['kernel'],copy.deepcopy(snapshot['args']),snapshot['prepared'])
        result.research=copy.deepcopy(snapshot['research'])
        return result

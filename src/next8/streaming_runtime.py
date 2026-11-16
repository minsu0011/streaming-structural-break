"""Scalar/prefix execution of a frozen research-fold model, without fitting."""
import copy
from pathlib import Path
import numpy as np
from numba import njit,typeof
import joblib
from src.next8.features import FeatureState,N_FEATURES,h_context
from src.next8.runtime_features import raw_arguments,normalized_step
from src.next8.runtime_head import coordinate_arguments,state_arguments,head_arguments,conditions,coordinates,score
from src.next8.common import read_json,digest
from src.next5.runtime import resident_class,SharedResearchBaseline,baseline_step,ScalarDetector
from src.next.fused_runtime import unique_array_bytes


@njit(cache=True)
def finish_step(normalized,args):
    state_args,coordinate_args,head,history,counter,context,last_output=args
    t=counter[0]
    # These conditions use only H and previously stored normalized features.
    conditions(history,t,state_args)
    x=coordinates(normalized,history,t,context,state_args[-2],state_args[-1],coordinate_args)
    p=score(x,head)
    history[t%32]=normalized;counter[0]+=1;last_output[0]=p
    return p


@njit(cache=True)
def kernel(point,args):
    engine,normalization,model=args
    normalized=normalized_step(point,engine,normalization)
    return finish_step(normalized,model)


@njit(cache=True)
def zero_kernel(point,args):
    normalized,model=args
    return finish_step(normalized,model)


@njit(cache=True)
def replay_kernel(points,args):
    out=np.empty(len(points),dtype=np.float32)
    for k,point in enumerate(points):out[k]=kernel(float(point),args)
    return out


@njit(cache=True)
def replay_zero_kernel(points,args):
    out=np.empty(len(points),dtype=np.float32)
    for k,point in enumerate(points):out[k]=zero_kernel(float(point),args)
    return out


def points(values):
    array=np.asarray(values)
    if array.ndim!=1 or not np.isfinite(array).all() or np.any(np.abs(array)>np.finfo(np.float32).max):
        raise ValueError('Finite one-dimensional float32-representable observations required')
    return array.astype(np.float32)


class PreparedJoint:
    def __init__(self,model,selection,method,graph_models=None,*,resident=True):
        if method not in ('M1','M2','M3','M4','M5'):raise ValueError('Frozen runtime method required')
        if model.training['training_ids_sha256']!=selection['scope']['training_ids_sha256']:
            raise ValueError('Model/feature selection training memberships differ')
        self.model=model;self.selection=copy.deepcopy(selection);self.method=method;self.resident=resident
        self.graph_models={} if graph_models is None else copy.deepcopy(graph_models)
        for j,m in self.graph_models.items():
            if m.training['training_ids_sha256']!=model.training['training_ids_sha256']:
                raise ValueError('Graph/model inference refit scopes differ')
        self.coordinate_args,self.direct_features=coordinate_arguments(model,selection,method)
        self.head=head_arguments(model)

    def new_state(self,historical):return JointState(self,historical)


class JointState:
    def __init__(self,prepared,historical):
        historical=points(historical);self.prepared=prepared
        selection=copy.deepcopy(prepared.selection)
        if prepared.method in ('M1','M2'):selection['variable_features']=[]
        # No-feature intercept/context models do not need to fit 153 unused
        # coordinates. The retained context is still computed from this H.
        if not prepared.direct_features:
            seed=FeatureState(np.empty(0,dtype=np.float32));seed.context=h_context(historical)
            self.normalization_skipped_no_feature_coordinate=True
        else:
            seed=FeatureState(historical);self.normalization_skipped_no_feature_coordinate=False
        state,required=state_arguments(seed,selection,prepared.graph_models,use_graph=prepared.method in ('M3','M5'))
        needed=prepared.direct_features|required
        history=np.zeros((32,N_FEATURES),dtype=np.float32)
        history[16:]=seed.history
        coordinate=copy.deepcopy(prepared.coordinate_args)
        head=copy.deepcopy(prepared.head)
        model=(state,coordinate,head,history,np.zeros(1,dtype=np.int64),seed.context.copy(),np.full(1,np.nan))
        self.zero_features=not seed.engines or not needed
        self.used_features=tuple(sorted(needed))
        self.short_H=len(historical)<160
        if self.zero_features:
            self.args=(np.zeros(N_FEATURES,dtype=np.float32),model);self.kernel=zero_kernel
        else:
            engine,norm=raw_arguments(seed,needed);self.args=(engine,norm,model);self.kernel=kernel
        self._resident=resident_class(self.kernel,typeof(self.args))(self.args) if prepared.resident else None

    @property
    def model_args(self):return self.args[-1]

    @property
    def counter(self):return int(self.model_args[4][0])

    @property
    def current_feature(self):
        return self.model_args[3][(self.counter-1)%32].copy() if self.counter else np.zeros(N_FEATURES,dtype=np.float32)

    @property
    def own_orientation_condition(self):return self.model_args[0][-2].copy()

    @property
    def forecast_graph_condition(self):return self.model_args[0][-1].copy()

    @property
    def last_design(self):return self.model_args[1][-1].copy()

    @property
    def unrounded_probability(self):return float(self.model_args[2][-1][1])

    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>np.finfo(np.float32).max:raise ValueError('Finite float32 scalar observation required')
        value=float(np.float32(value))
        p=float(self._resident.predict_one(value) if self._resident is not None else self.kernel(value,self.args))
        if not np.isfinite(p) or not 0<=p<=1:raise RuntimeError('Invalid causal scalar prediction')
        return p

    def replay(self,online):
        online=points(online)
        return replay_zero_kernel(online,self.args) if self.zero_features else replay_kernel(online,self.args)

    def snapshot(self):return copy.deepcopy(self)

    @classmethod
    def restore(cls,snapshot):
        if not isinstance(snapshot,cls):raise TypeError('Exact NEXT8 runtime snapshot required')
        return copy.deepcopy(snapshot)

    def __getstate__(self):
        values=self.__dict__.copy();values.pop('_resident',None);return values

    def __setstate__(self,values):
        self.__dict__.update(values)
        self._resident=resident_class(self.kernel,typeof(self.args))(self.args) if self.prepared.resident else None

    @property
    def total_instance_array_bytes(self):return unique_array_bytes(self.args)


def load_prepared(model_directory,selection_path,graph_directory,method,*,resident=True):
    model_directory=Path(model_directory);done=read_json(model_directory/'DONE.json')
    actual=done['model_aliases'][method]
    model_path=model_directory/(actual+'.joblib')
    if digest(model_path)!=done['files_sha256'][model_path.name]:raise ValueError('Model artifact bytes changed')
    selection=read_json(selection_path)
    if selection['scope']!=done['scope']:raise ValueError('Model/selection scope mismatch')
    graph_directory=Path(graph_directory);graph_done=read_json(graph_directory/'DONE.json');graph={}
    if graph_done['scope']!=done['scope']:raise ValueError('Graph/model scope mismatch')
    for r in graph_done['models']:
        if r['status']=='COMPLETE':
            path=graph_directory/f"F{r['feature']:03d}_T3.joblib"
            if digest(path)!=r['model_sha256']:raise ValueError('Graph artifact bytes changed')
            graph[int(r['feature'])]=joblib.load(path)
    return PreparedJoint(joblib.load(model_path),selection,actual,graph,resident=resident)


class IntegratedState:
    """Fixed95/5 M6 composition; original reference normalization stays intact."""
    def __init__(self,baseline_prepared,joint_prepared,historical,eligible):
        from src.next8.data import membership_numpy_hash
        wanted=joint_prepared.selection['scope']['training_ids_sha256']
        reference=baseline_prepared.j102.training
        actual=reference['training_ids_sha256']
        legacy=membership_numpy_hash(joint_prepared.selection['selection_ids'])
        if actual not in (wanted,legacy) and reference.get('training_group_identity')!=wanted:
            raise ValueError('BL80 and M5 must have the same excluded-group training membership')
        if bool(reference.get('early_only_reference_new_fit_not_canonical',False)) != (joint_prepared.model.training.get('early_training_cutoff')==199):
            raise ValueError('BL80 and M5 training time domains differ')
        self.baseline=baseline_prepared.new_state(points(historical))
        self.joint=joint_prepared.new_state(historical) if eligible else None
        self.eligible=bool(eligible)

    def predict_one(self,point):
        baseline=self.baseline.predict_one(point)
        return .95*baseline+.05*self.joint.predict_one(point) if self.eligible else baseline

    def replay(self,online):return np.asarray([self.predict_one(x) for x in points(online)])

    def snapshot(self):return copy.deepcopy(self)

    @classmethod
    def restore(cls,snapshot):
        if not isinstance(snapshot,cls):raise TypeError('Exact integrated NEXT8 snapshot required')
        return copy.deepcopy(snapshot)

    def __getstate__(self):
        return {'baseline_args':self.baseline.args,'baseline_prepared':self.baseline.prepared,
                'joint':self.joint,'eligible':self.eligible}

    def __setstate__(self,state):
        self.baseline=ScalarDetector(baseline_step,state['baseline_args'],state['baseline_prepared'])
        self.joint=state['joint'];self.eligible=state['eligible']

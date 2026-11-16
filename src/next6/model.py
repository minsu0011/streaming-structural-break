"""Small fixed LightGBM head, with cross-fitted graph inputs supplied by caller."""
from dataclasses import dataclass
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training
from src.next5.reference_features import ReferenceState
from src.next6.nodes import NodeState,SET_C
from src.next6.graph import GraphState
from src.next6.common import ROOT,BASE,read_json,digest,ids_hash,activity

EXTRA_COLUMNS={'BASE_ONLY':[],'M1':[],'M2':list(range(12)),'M3':list(range(12,24)),
    'M4':list(range(24)),'OWN':list(range(24)),'RANDOM':list(range(24))}

@dataclass
class Model:
    method:str
    columns:np.ndarray
    compact:object
    graph:object
    training:dict
    source_sha256:dict
    def predict(self,x):return self.compact.predict(np.asarray(x,dtype=np.float32)[:,self.columns])
    def validate(self):
        for path,sha in self.source_sha256.items():
            if digest(ROOT/path)!=sha:raise RuntimeError('NEXT6 model source/contract changed: '+path)
    def new_state(self,historical):self.validate();return LiveState(self,historical)
    def save(self,path):
        p=Path(path)
        if p.exists():raise RuntimeError('Model overwrite forbidden')
        joblib.dump(self,p)

class LiveState:
    def __init__(self,model,historical):
        self.model=model
        self.j=ReferenceState(historical,{'version':'next5_reference_features_v1','kind':'J102'})
        self.c=ReferenceState(historical,SET_C)
        self.nodes=NodeState(historical)
        self.graph=GraphState(model.graph,self.nodes.pseudo_s[-16:]) if model.graph is not None else None
    def predict_one(self,point):
        j=self.j.update(point).copy();c=self.c.update(point).copy();x,s,t=self.nodes.update(point)
        base=np.concatenate((j,c)) if self.model.method=='BASE_ONLY' else np.concatenate((j,c,t))
        if self.graph is not None:
            extra=self.graph.update(t,s);base=np.concatenate((base,extra[EXTRA_COLUMNS[self.model.method]]))
        return float(self.model.compact.predict_one(base[self.model.columns]))

def fit_head(x,y,time,ids,guard,method,graph,validation_ids):
    guard.partition(np.unique(ids),validation_ids)
    protocol=read_json(BASE/'NEXT6_PROTOCOL_LOCK.json')
    with activity('fit_'+method,'MODEL_RESEARCH'):
        columns=np.flatnonzero(np.ptp(x,axis=0)>1e-12).astype(np.int32)
        xx=np.asarray(x[:,columns],dtype=np.float32)
        if not np.isfinite(xx).all():raise ValueError('Nonfinite training input')
        weight=weights_from_training(y,time,ids,'W3')
        est=make_estimator('M4_lightgbm',protocol['learner_params'])
        with threadpool_limits(limits=4):est.fit(xx,y,sample_weight=weight)
        compact=export_lightgbm(est)
        with threadpool_limits(limits=1):native=est.booster_.predict(xx[:2048],num_threads=1)
        assert np.max(np.abs(native-compact.predict(xx[:2048])))<1e-12
    sources={n:digest(ROOT/n) for n in ('src/next6/nodes.py','src/next6/graph.py','src/next6/model.py','artifacts/next6/NEXT6_PROTOCOL_LOCK.json')}
    return Model(method,columns,compact,graph,{'training_ids_sha256':ids_hash(np.unique(ids)),
        'training_series':len(np.unique(ids)),'training_rows':len(y),'head_inputs_crossfit':method not in ('BASE_ONLY','M1'),
        'constant_columns_train_only':True,'weights_train_only':True,'validation_labels_used':False},sources)

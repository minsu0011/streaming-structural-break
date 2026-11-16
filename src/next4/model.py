"""Fixed small base learners over independently registered causal feature banks."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm,export_histgb
from src.next.fitting import weights_from_training


def source_hash():
    root=Path(__file__).resolve().parents[2]
    names=('src/next4/model.py','src/models/learners.py','src/models/compact_trees.py','src/next/fitting.py')
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in names)).hexdigest()


def feature_module(name):
    if not isinstance(name,str) or not name.startswith('src.next4.') or name.count('.')!=2:
        raise ValueError('A NEXT4 feature module is required')
    return importlib.import_module(name)


@dataclass
class FeatureModel:
    module_name:str
    settings:dict
    names:tuple
    columns:np.ndarray
    compact:object
    estimator:object
    feature_sha256:str
    model_source_sha256:str
    training:dict

    def validate(self):
        module=feature_module(self.module_name)
        if module.implementation_hash(self.settings)!=self.feature_sha256 or source_hash()!=self.model_source_sha256:
            raise RuntimeError('Feature/model source changed after fitting')
        if tuple(module.feature_names(self.settings))!=self.names:raise RuntimeError('Feature names changed')

    def predict(self,x):return self.compact.predict(np.asarray(x,dtype=np.float32)[:,self.columns])

    def predict_one(self,x):return float(self.compact.predict_one(np.asarray(x,dtype=np.float32)[self.columns]))

    def make_state(self,historical):
        self.validate();return feature_module(self.module_name).make_state(historical,self.settings)

    def save(self,path):
        path=Path(path)
        if path.exists():raise RuntimeError('Never overwrite a fitted model')
        path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(self,path)

    @classmethod
    def load(cls,path):
        model=joblib.load(path)
        if type(model) is not cls:raise TypeError('Registered FeatureModel required')
        model.validate();return model


def fit_base(x,metadata,train,valid,*,guard,registration):
    train=np.asarray(train,dtype=np.int64);valid=np.asarray(valid,dtype=np.int64)
    if np.intersect1d(train,valid).size or len(np.unique(train))!=len(train) or len(np.unique(valid))!=len(valid):
        raise ValueError('Unique disjoint train/validation rows required')
    guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
    learner=registration['parameters']['learner'];weighting=registration['parameters']['weighting']
    if learner not in ('M4_lightgbm','M3_histgb'):raise ValueError('Other objectives require separately registered adapters')
    module=feature_module(registration['feature_module']);settings=registration['feature_settings']
    if module.implementation_hash(settings)!=registration['feature_sha256']:raise RuntimeError('Frozen feature source mismatch')
    names=tuple(module.feature_names(settings));matrix=np.asarray(x[train],dtype=np.float32)
    if matrix.shape[1]!=len(names) or not np.isfinite(matrix).all():raise ValueError('Finite registered feature columns required')
    columns=np.flatnonzero(np.ptp(matrix,axis=0)>1e-12).astype(np.int32)
    if not len(columns):raise ValueError('All features are constant')
    matrix=matrix[:,columns];target=np.asarray(metadata['target'][train],dtype=np.uint8)
    if set(np.unique(target))!={0,1}:raise ValueError('Both train classes required')
    weight=weights_from_training(target,metadata['time_online'][train],metadata['dataset_id'][train],weighting)
    estimator=make_estimator(learner,registration['parameters']['learner_params'])
    with threadpool_limits(limits=4):estimator.fit(matrix,target,sample_weight=weight)
    compact=export_lightgbm(estimator) if learner=='M4_lightgbm' else export_histgb(estimator)
    probe=matrix[:min(2048,len(matrix))]
    with threadpool_limits(limits=1):
        native=estimator.booster_.predict(probe,num_threads=1) if learner=='M4_lightgbm' else estimator.predict_proba(probe)[:,1]
    error=float(np.max(np.abs(compact.predict(probe)-native)))
    if error>1e-12:raise RuntimeError('Native/compact export parity failed')
    training={'training_rows':len(train),'validation_rows':len(valid),
        'training_series':len(np.unique(metadata['dataset_id'][train])),
        'validation_series':len(np.unique(metadata['dataset_id'][valid])),
        'retained_columns':len(columns),'feature_count':len(names),
        'weighting':weighting,'weight_statistics_training_only':True,'new_fits':1,
        'native_compact_max_abs_difference':error,'validation_labels_used_in_fit':False,
        'base_prediction_inputs_used_in_fit':False,'seal_rows':0}
    model=FeatureModel(registration['feature_module'],settings,names,columns,compact,estimator,
        registration['feature_sha256'],source_hash(),training)
    return model,np.asarray(model.predict(x[valid]),dtype=np.float32),training

"""One fixed learner; all target/weight/column statistics stay in training groups."""
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training
from src.next5.artifacts import ROOT,BASE,read_json,digest


def source_hash():
    files=('src/next5/model.py','src/models/learners.py','src/models/compact_trees.py','src/next/fitting.py','src/next5/guard.py')
    return hashlib.sha256(b''.join((ROOT/n).read_bytes() for n in files)).hexdigest()


def feature_module(name):
    if not isinstance(name,str) or not name.startswith('src.next5.') or name.count('.')!=2:
        raise ValueError('Explicit NEXT5 feature module required')
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
            raise RuntimeError('Frozen scientific feature/model source changed')
        if tuple(module.feature_names(self.settings))!=self.names:raise RuntimeError('Feature coordinate names changed')

    def predict(self,x):return self.compact.predict(np.asarray(x,dtype=np.float32)[:,self.columns])
    def predict_one(self,x):return float(self.compact.predict_one(np.asarray(x,dtype=np.float32)[self.columns]))
    def make_state(self,historical):
        self.validate();return feature_module(self.module_name).make_state(historical,self.settings)
    def save(self,path):
        path=Path(path)
        if path.exists():raise RuntimeError('Never overwrite fitted artifact')
        path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(self,path)
    @classmethod
    def load(cls,path):
        model=joblib.load(path)
        if type(model) is not cls:raise TypeError('NEXT5 FeatureModel required')
        model.validate();return model


def fit_registered(x,metadata,train,valid,*,guard,registration,allow_frozen_reproduction=False):
    cid=registration['experiment_id']
    directory=BASE/'references/registrations' if cid.startswith('REF_') else BASE/'registrations'
    registration_path=directory/(cid+'.json')
    if not registration_path.is_file() or read_json(registration_path)!=registration:
        raise RuntimeError('Exact persisted preregistration required before every fit')
    session=read_json(BASE/'SESSION.json')
    if session.get('shadow_opened') or (BASE/'shadow/FINALIST_MANIFEST_NEXT5.json').exists():
        if not allow_frozen_reproduction:raise RuntimeError('No research fit after finalist freeze or Shadow access')
        manifest=read_json(BASE/'shadow/FINALIST_MANIFEST_NEXT5.json')
        allowed={r['experiment_id']:r['registration_sha256'] for r in manifest['finalists']}
        allowed.update(manifest.get('reference_registration_sha256',{}))
        if allowed.get(cid)!=digest(registration_path):raise RuntimeError('Only hash-bound frozen finalists/reference reproduction allowed')
    train=np.asarray(train,dtype=np.int64);valid=np.asarray(valid,dtype=np.int64)
    if not len(train) or len(np.unique(train))!=len(train) or len(np.unique(valid))!=len(valid) or np.intersect1d(train,valid).size:
        raise ValueError('Unique disjoint train/validation rows required')
    guard.admit(np.unique(metadata['dataset_id']),purpose='learner input design')
    if len(valid):guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
    else:guard.admit(np.unique(metadata['dataset_id'][train]),purpose='frozen whole-ResearchPool training')
    module=feature_module(registration['feature_module']);settings=registration['feature_settings']
    if module.implementation_hash(settings)!=registration['feature_sha256'] or source_hash()!=registration['model_source_sha256']:
        raise RuntimeError('Preregistered scientific source changed before fit')
    names=tuple(module.feature_names(settings));matrix=np.asarray(x[train],dtype=np.float32)
    if matrix.shape[1]!=len(names) or not np.isfinite(matrix).all():raise ValueError('Finite registered design required')
    columns=np.flatnonzero(np.ptp(matrix,axis=0)>1e-12).astype(np.int32)
    if not len(columns):raise ValueError('All features constant')
    matrix=matrix[:,columns];target=np.asarray(metadata['target'][train],dtype=np.uint8)
    if set(np.unique(target))!={0,1}:raise ValueError('Both training classes required')
    parameters=registration['parameters'];assert parameters['learner']=='M4_lightgbm' and parameters['weighting']=='W3'
    weights=weights_from_training(target,metadata['time_online'][train],metadata['dataset_id'][train],'W3')
    estimator=make_estimator('M4_lightgbm',parameters['learner_params'])
    with threadpool_limits(limits=4):estimator.fit(matrix,target,sample_weight=weights)
    compact=export_lightgbm(estimator)
    with threadpool_limits(limits=1):native=estimator.booster_.predict(matrix[:2048],num_threads=1)
    error=float(np.max(np.abs(compact.predict(matrix[:2048])-native)))
    if error>1e-12:raise RuntimeError('Compact export parity failed')
    training={'training_rows':len(train),'training_series':len(np.unique(metadata['dataset_id'][train])),
        'validation_rows':len(valid),'validation_series':len(np.unique(metadata['dataset_id'][valid])),
        'feature_count':len(names),'retained_columns':len(columns),'new_fits':1,'shadow_rows':0,'seal_rows':0,
        'training_only_columns_weights':True,'validation_labels_used_in_fit':False,
        'native_compact_max_abs_difference':error,'research_split_sha256':guard.split.digest,
        'shadow_lock_sha256':guard.shadow_lock_sha,'training_ids_sha256':hashlib.sha256(np.sort(np.unique(metadata['dataset_id'][train])).tobytes()).hexdigest()}
    model=FeatureModel(registration['feature_module'],settings,names,columns,compact,estimator,
                       registration['feature_sha256'],source_hash(),training)
    return model,training

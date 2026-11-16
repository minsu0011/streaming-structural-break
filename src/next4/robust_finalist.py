"""Unchanged registered robustness recipes, materialized only after nomination."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from src.next4.artifacts import ROOT,BASE,read_json,digest
from src.next4.model import FeatureModel,feature_module,source_hash
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training
from src.next2.blend import combine
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next2.dependence_runtime import PreparedDependencePredictor
from src.next.fused_runtime import unique_array_bytes

ALLOWED=('B1_SPLIT_ENERGY_BRANCHES','C3_SHAPE_RATIO_DISTRIBUTION','D2_DISCRETE_SCALE_MIXTURE','D3_HAZARD_MIXTURE')


def fit_registered_all_dev(x,metadata,*,guard,registration,expected_ids,fresh_raw):
    cid=registration['experiment_id']
    if cid not in ALLOWED:raise ValueError('Existing admitted robustness recipe required')
    nomination=read_json(BASE/'finalist_nominations'/(cid+'.json'))
    if nomination['registration_sha256']!=digest(BASE/'registrations'/(cid+'.json')):raise RuntimeError('Nomination changed')
    ids=np.unique(metadata['dataset_id']);admitted=np.asarray(guard.admit(ids,purpose='NEXT4 nominated robustness all-DEV fit'))
    np.testing.assert_array_equal(np.sort(admitted),np.sort(expected_ids))
    module=feature_module(registration['feature_module']);settings=registration['feature_settings']
    if module.implementation_hash(settings)!=registration['feature_sha256'] or source_hash()!=registration['model_source_sha256']:
        raise RuntimeError('Original registered scientific source changed')
    parameters=registration['parameters'];assert parameters['learner']=='M4_lightgbm' and parameters['weighting']=='W3'
    names=tuple(module.feature_names(settings));matrix=np.asarray(x,dtype=np.float32);target=np.asarray(metadata['target'],dtype=np.uint8)
    assert matrix.shape==(len(target),len(names)) and np.isfinite(matrix).all() and set(np.unique(target))=={0,1}
    columns=np.flatnonzero(np.ptp(matrix,axis=0)>1e-12).astype(np.int32);assert len(columns)
    matrix=matrix[:,columns];weights=weights_from_training(target,metadata['time_online'],metadata['dataset_id'],'W3')
    estimator=make_estimator('M4_lightgbm',parameters['learner_params'])
    with threadpool_limits(limits=4):estimator.fit(matrix,target,sample_weight=weights)
    compact=export_lightgbm(estimator)
    with threadpool_limits(limits=1):native=estimator.booster_.predict(matrix[:2048],num_threads=1)
    error=float(np.max(np.abs(compact.predict(matrix[:2048])-native)));assert error<=1e-12
    training={'training_scope':'ALL_DEV_8000_NOMINATED_ROBUSTNESS','training_series':len(ids),'training_rows':len(target),
        'validation_rows':0,'validation_series':0,'new_fits':1,'frozen_base_new_fits':0,'feature_count':len(names),
        'retained_columns':len(columns),'weighting':'W3','weight_statistics_training_only':True,
        'native_compact_max_abs_difference':error,'seal_rows':0,'fresh_raw_reproduction':bool(fresh_raw),
        'real_in_sample_quality_score_computed':False}
    return FeatureModel(registration['feature_module'],settings,names,columns,compact,estimator,
        registration['feature_sha256'],source_hash(),training),training


class PreparedRobustness:
    def __init__(self,model,frozen,candidate):
        if candidate not in ALLOWED:raise ValueError('Bounded unchanged robustness recipe required')
        model.validate();self.model=model;self.candidate=candidate;self.augmented=candidate.startswith('B1_')
        self.reference=PreparedDependencePredictor(frozen.complement,resident=False) if self.augmented else PreparedSharedBlendPredictor(frozen,resident=False)

    def new_state(self,historical):return RobustnessState(self,historical)


class RobustnessState:
    def __init__(self,prepared,historical):
        self.prepared=prepared;self.feature=prepared.model.make_state(historical);self.reference=prepared.reference.new_state(historical)

    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value):raise ValueError('Finite current scalar required')
        expert=self.prepared.model.predict_one(self.feature.update(value));reference=self.reference.predict_one(value)
        return float(combine(expert,reference,.8) if self.prepared.augmented else combine(reference,expert,.9))

    @property
    def state_array_bytes(self):return unique_array_bytes((self.feature.args,self.reference.call_args[:-1]))


def load_prepared(directory):
    from src.next2.shared_inference import load_artifact
    directory=Path(directory);receipt=read_json(directory/'RESULTS.json');cid=receipt['candidate']
    assert receipt['status']=='PASS' and cid in ALLOWED
    assert digest(directory/'model.joblib')==receipt['files_sha256']['model.joblib']
    model=FeatureModel.load(directory/'model.joblib');frozen,_=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    return PreparedRobustness(model,frozen,cid)


def infer(datasets,model_directory_path):
    yield None
    prepared=load_prepared(model_directory_path)
    for historical,online in datasets:
        state=prepared.new_state(historical)
        for point in online:yield state.predict_one(point)

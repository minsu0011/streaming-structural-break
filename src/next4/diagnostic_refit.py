"""One all-DEV fit for synthetic diagnostics; no validation or quality estimate."""
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training
from src.next4.model import FeatureModel,feature_module,source_hash


def fit_all_dev(x,metadata,*,guard,registration,expected_ids):
    ids=np.unique(metadata['dataset_id'])
    admitted=np.asarray(guard.admit(ids,purpose='NEXT4 all-DEV diagnostic refit'))
    if not np.array_equal(np.sort(admitted),np.sort(expected_ids)):
        raise ValueError('Exactly the admitted DEV training partition is required')
    if registration['experiment_id']!='B2_LOG_ENERGY_SEMITAIL':
        raise ValueError('This one-fit diagnostic nomination only authorizes B2')
    module=feature_module(registration['feature_module']);settings=registration['feature_settings']
    if module.implementation_hash(settings)!=registration['feature_sha256'] or source_hash()!=registration['model_source_sha256']:
        raise RuntimeError('Registered feature or learner source changed')
    parameters=registration['parameters']
    if parameters['learner']!='M4_lightgbm' or parameters['weighting']!='W3':
        raise ValueError('The previously evaluated B2 learner recipe is required')
    names=tuple(module.feature_names(settings));matrix=np.asarray(x,dtype=np.float32)
    target=np.asarray(metadata['target'],dtype=np.uint8)
    if matrix.shape!=(len(target),len(names)) or not np.isfinite(matrix).all() or set(np.unique(target))!={0,1}:
        raise ValueError('Finite aligned DEV features and both training classes required')
    columns=np.flatnonzero(np.ptp(matrix,axis=0)>1e-12).astype(np.int32)
    if not len(columns):raise ValueError('All training features are constant')
    matrix=matrix[:,columns]
    weights=weights_from_training(target,metadata['time_online'],metadata['dataset_id'],'W3')
    estimator=make_estimator('M4_lightgbm',parameters['learner_params'])
    with threadpool_limits(limits=4):estimator.fit(matrix,target,sample_weight=weights)
    compact=export_lightgbm(estimator);probe=matrix[:2048]
    with threadpool_limits(limits=1):native=estimator.booster_.predict(probe,num_threads=1)
    error=float(np.max(np.abs(compact.predict(probe)-native)))
    if error>1e-12:raise RuntimeError('All-DEV native/compact parity failed')
    detail={'training_scope':'ALL_DEV_8000_SYNTHETIC_DIAGNOSTIC_ONLY','training_rows':len(target),
        'training_series':len(ids),'validation_rows':0,'validation_series':0,'new_fits':1,
        'weighting':'W3','weight_statistics_training_only':True,'feature_count':len(names),
        'retained_columns':len(columns),'native_compact_max_abs_difference':error,
        'seal_rows':0,'real_in_sample_quality_score_computed':False,'promotion':False,
        'fresh_raw_reproduction':False,'frozen_base_new_fits':0}
    return FeatureModel(registration['feature_module'],settings,names,columns,compact,estimator,
        registration['feature_sha256'],source_hash(),detail),detail

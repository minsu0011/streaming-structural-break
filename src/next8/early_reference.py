"""Early-only BL80 comparator with original feature/learner recipes."""
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from src.next8.common import ids_hash
from src.next8.data import rows_for
from src.next5.model import FeatureModel,feature_module,source_hash
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training


def fit_early_reference(x,metadata,scope,registration,guard):
    rows=rows_for(metadata,scope.training_ids)
    rows=rows[metadata['time_online'][rows]<=199]
    ids=np.unique(metadata['dataset_id'][rows]);scope.admit_training(ids)
    guard.partition(scope.training_ids,scope.validation_ids|scope.outer_excluded_ids)
    module=feature_module(registration['feature_module']);settings=registration['feature_settings']
    assert module.implementation_hash(settings)==registration['feature_sha256']
    assert source_hash()==registration['model_source_sha256']
    names=tuple(module.feature_names(settings));matrix=np.asarray(x[rows],dtype=np.float32)
    assert matrix.shape[1]==len(names) and np.isfinite(matrix).all()
    columns=np.flatnonzero(np.ptp(matrix,axis=0)>1e-12).astype(np.int32);matrix=matrix[:,columns]
    y=np.asarray(metadata['target'][rows],dtype=np.uint8)
    weight=weights_from_training(y,metadata['time_online'][rows],metadata['dataset_id'][rows],'W3')
    recipe=registration['parameters'];assert recipe['learner']=='M4_lightgbm' and recipe['weighting']=='W3'
    params={**recipe['learner_params'],'n_jobs':1}
    estimator=make_estimator('M4_lightgbm',params)
    with threadpool_limits(limits=1):estimator.fit(matrix,y,sample_weight=weight)
    compact=export_lightgbm(estimator)
    with threadpool_limits(limits=1):native=estimator.booster_.predict(matrix[:2048],num_threads=1)
    error=float(np.max(np.abs(native-compact.predict(matrix[:2048]))));assert error<1e-12
    training={**scope.provenance(),'training_rows':len(rows),'training_max_online_time':int(metadata['time_online'][rows].max()),
        'training_ids_numpy_sha256':hashlib.sha256(np.sort(ids).tobytes()).hexdigest(),
        'feature_and_learner_recipe':'Original frozen reference features, 100 depth4 LightGBM trees and W3 train-only weighting',
        'execution_thread_override':{'registered_n_jobs':recipe['learner_params']['n_jobs'],'actual_n_jobs':1},
        'early_only_reference_new_fit_not_canonical':True,'native_compact_max_abs_difference':error,
        'validation_or_late_training_labels_used':False,'H_normalization':'Original canonical feature definition retained; separate from NEXT8 same-reference normalization',
        'research_split_sha256':guard.split.digest}
    return FeatureModel(registration['feature_module'],settings,names,columns,compact,estimator,
        registration['feature_sha256'],source_hash(),training)

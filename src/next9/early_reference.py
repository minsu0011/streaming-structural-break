"""Original BL80 component recipes, fitted only on this inner TRAIN t<=199."""
import numpy as np
from threadpoolctl import threadpool_limits
from src.next9.common import ROOT,read_json,digest
from src.next9.reference_access import matrix
from src.next5.model import FeatureModel,feature_module,source_hash
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training


def fit(kind,bank,labels,scope):
    registration=read_json(ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json')
    module=feature_module(registration['feature_module']);settings=registration['feature_settings']
    assert module.implementation_hash(settings)==registration['feature_sha256']
    assert source_hash()==registration['model_source_sha256']
    source=matrix(kind,bank)
    rows=bank.sampled_rows(scope.training_ids,cutoff=199,stride=1)
    scope.admit_training(bank.row_ids[rows])
    y=labels.get(rows,scope,'TRAIN',early=True)
    x=np.asarray(source[rows],dtype=np.float32)
    columns=np.flatnonzero(np.ptp(x,axis=0)>1e-12).astype(np.int32);x=x[:,columns]
    if not len(columns) or not np.isfinite(x).all():raise ValueError('Original finite nonconstant reference features required')
    w=weights_from_training(y,bank.times[rows],bank.row_ids[rows],'W3')
    recipe=registration['parameters']
    assert recipe['learner']=='M4_lightgbm' and recipe['weighting']=='W3'
    params={**recipe['learner_params'],'n_jobs':1}
    estimator=make_estimator('M4_lightgbm',params)
    with threadpool_limits(limits=1):estimator.fit(x,y,sample_weight=w)
    compact=export_lightgbm(estimator)
    with threadpool_limits(limits=1):native=estimator.booster_.predict(x[:2048],num_threads=1)
    error=float(np.max(np.abs(native-compact.predict(x[:2048]))));assert error<1e-12
    training=dict(**scope.provenance(),training_rows=len(rows),training_max_online_time=int(bank.times[rows].max()),
        early_only_reference_not_canonical=True,source_registration_sha256=digest(ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json'),
        original_feature_and_learner_recipe_preserved=True,W3_training_only=True,math_threads=1,native_compact_max_error=error,
        validation_or_late_training_labels_used=False,estimator_fit_calls=1)
    model=FeatureModel(registration['feature_module'],settings,tuple(module.feature_names(settings)),columns,
        compact,estimator,registration['feature_sha256'],source_hash(),training)
    return model,source

"""Small fixed binary learners and train-only time-contribution weighting."""
from pathlib import Path
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm, export_histgb
from src.next.predictor import NextBundle, predictor_hash, arch_contract
from src.next.engine import feature_hash


def fitting_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def weights_from_training(target, time_online, ids, policy='W3'):
    y, t, ids = np.asarray(target,dtype=np.uint8), np.asarray(time_online,dtype=np.int32), np.asarray(ids)
    if policy == 'W0':
        return None
    if policy == 'W1':
        _, code, counts = np.unique(ids, return_inverse=True, return_counts=True)
        weight = 1./counts[code]
    else:
        positive = np.bincount(t, weights=y)
        negative = np.bincount(t, weights=1-y, minlength=len(positive))
        if policy == 'W2':
            weight = (positive*negative/np.maximum(positive+negative,1.))[t]
        elif policy in ('W3','EXISTING'):
            weight = np.where(y == 1, negative[t], positive[t])
        else:
            raise ValueError('Weighting policy outside frozen family')
    if weight.sum() <= 0:
        raise ValueError('No positive-negative training pairs')
    return weight/weight.mean()


def fit_binary(design, metadata, training_rows, validation_rows, *, guard, config, names,
               learner='M4_lightgbm', weighting='W3', params=None):
    if learner not in ('M4_lightgbm','M3_histgb'):
        raise ValueError('Learner requires its separately audited objective adapter')
    train = np.asarray(training_rows, dtype=np.int64)
    valid = np.asarray(validation_rows, dtype=np.int64)
    guard.partition(np.unique(metadata['dataset_id'][train]), np.unique(metadata['dataset_id'][valid]))
    x = np.asarray(design[train], dtype=np.float32)
    y = metadata['target'][train]
    columns = np.flatnonzero(np.ptp(x, axis=0)>1e-12).astype(np.int32)
    x = x[:,columns]
    if not len(columns) or not np.isfinite(x).all() or set(np.unique(y)) != {0,1}:
        raise ValueError('Finite nonconstant features and both training classes required')
    weight = weights_from_training(y, metadata['time_online'][train], metadata['dataset_id'][train], weighting)
    estimator = make_estimator(learner, params)
    with threadpool_limits(limits=4):
        estimator.fit(x, y, sample_weight=weight)
    compact = export_lightgbm(estimator) if learner == 'M4_lightgbm' else export_histgb(estimator)
    probe = x[:min(len(x),2048)]
    with threadpool_limits(limits=1):
        native = estimator.booster_.predict(probe, num_threads=1) if learner == 'M4_lightgbm' else estimator.predict_proba(probe)[:,1]
    np.testing.assert_allclose(compact.predict(probe), native, rtol=0, atol=1e-12)
    model = NextBundle(config, tuple(names), columns, compact, estimator, learner,
                       feature_hash(config), predictor_hash(), arch_contract() if any(name.startswith('arch8__') for name in names) else '')
    prediction = np.asarray(model.predict(design[valid]), dtype=np.float32)
    return model, prediction, {'training_rows': len(train), 'validation_rows': len(valid),
                              'retained_columns': len(columns), 'constant_columns_removed': len(names)-len(columns),
                              'native_compact_max_abs_diff_bound': 1e-12, 'weighting': weighting}

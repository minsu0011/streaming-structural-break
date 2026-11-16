"""Small fixed learners; original-group scope and frozen inference schema."""
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from src.next9.movement_screen import group_weights


@dataclass
class Classifier:
    width: int
    columns: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    classes: int
    estimator: object
    prior: np.ndarray
    training: dict

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.width or not np.isfinite(x).all():
            raise ValueError('Changed or nonfinite frozen inference schema')
        return (x[:, self.columns]-self.center)/self.scale

    def predict_proba(self, x):
        z = self.transform(x)
        if self.estimator is None:
            return np.tile(self.prior, (len(z), 1))
        with threadpool_limits(limits=1):
            p = self.estimator.predict_proba(z)
        out = np.full((len(z), self.classes), 1e-8)
        out[:, self.estimator.classes_.astype(int)] = p
        return out/out.sum(axis=1, keepdims=True)

    def linear_parameters(self):
        if self.classes != 2 or self.training['method'] != 'LOGISTIC':
            raise ValueError('Binary logistic coefficients required')
        coef = np.zeros(self.width)
        if self.estimator is None:
            return coef, float(np.log(self.prior[1]/self.prior[0]))
        raw = self.estimator.coef_[0]/self.scale
        coef[self.columns] = raw
        return coef, float(self.estimator.intercept_[0]-self.center@raw)


def fit_classifier(x, y, ids, groups, scope, *, method='LOGISTIC', classes=2):
    x, y, ids, groups = map(np.asarray, (x, y, ids, groups))
    scope.admit_training(ids)
    if (x.ndim != 2 or y.shape != (len(x),) or len(ids) != len(x)
            or len(groups) != len(x) or not np.isfinite(x).all()
            or not np.isin(y, np.arange(classes)).all()):
        raise ValueError('Aligned finite TRAIN design and registered classes required')
    if method not in ('LOGISTIC', 'DEPTH3'):
        raise ValueError('Unregistered learner')
    # Exact duplicate/constant pruning is TRAIN-only and label-free.
    selected, seen = [], {}
    for k in range(x.shape[1]):
        column = np.asarray(x[:, k], dtype=np.float64)
        if np.ptp(column) <= 1e-12:
            continue
        key = hashlib.sha256(column.tobytes()).hexdigest()
        if key in seen:
            np.testing.assert_array_equal(column, x[:, seen[key]])
            continue
        seen[key] = k
        selected.append(k)
    columns = np.asarray(selected, dtype=int)
    w = group_weights(groups)
    raw = np.asarray(x[:, columns], dtype=float)
    center = np.average(raw, axis=0, weights=w)
    scale = np.maximum(np.sqrt(np.average((raw-center)**2, axis=0, weights=w)), 1e-6)
    z = (raw-center)/scale
    counts = np.bincount(y.astype(int), weights=w, minlength=classes)
    prior = (counts+.5)/(counts.sum()+.5*classes)
    record = dict(**scope.provenance(), method=method, classes=classes,
                  rows=len(x), group_weighting='Equal total weight per original dependency group',
                  original_columns=x.shape[1], retained_columns=selected,
                  duplicate_pruning_training_only=True, estimator_fit_calls=0)
    estimator = None
    if len(selected) and len(np.unique(y)) > 1:
        if method == 'LOGISTIC':
            estimator = LogisticRegression(C=.1, max_iter=250, tol=1e-6, solver='lbfgs')
        else:
            estimator = HistGradientBoostingClassifier(max_depth=3, max_iter=100,
                learning_rate=.05, min_samples_leaf=128, l2_regularization=10.,
                early_stopping=False, random_state=9101604)
        with threadpool_limits(limits=1):
            estimator.fit(z, y.astype(int), sample_weight=w)
        record['estimator_fit_calls'] = 1
        if method == 'LOGISTIC':
            record.update(iterations=estimator.n_iter_.tolist(), converged=bool(np.max(estimator.n_iter_) < 250))
        record['parameters'] = estimator.get_params()
    return Classifier(x.shape[1], columns, center, scale, classes, estimator, prior, record)

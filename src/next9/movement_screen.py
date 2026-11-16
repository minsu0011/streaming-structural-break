"""Target-free fixed ridge movement screening over all153 coordinates."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from src.next9.states import directions, DOWN, FLAT, UP
from src.next9.common import ids_hash

HORIZONS = (1, 4, 16)
DESIGN_NAMES = ('F_lag1','F_lag2','F_lag4','F_lag8','F_lag16',
                'past_slope8','deltaPrev','directionRunCapped15','reversalAgeCapped15')


def group_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[inverse]
    return w / w.mean()


def grouped_means(values, groups):
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        return unique, np.bincount(inverse, weights=a) / counts
    return unique, np.column_stack([np.bincount(inverse, weights=a[:, k]) / counts for k in range(a.shape[1])])


def paired_group_summary(reference, candidate, groups):
    ids, mean = grouped_means(np.asarray(reference) - np.asarray(candidate), groups)
    gain = float(mean.mean())
    se = float(mean.std(ddof=1) / np.sqrt(len(mean))) if len(mean) > 1 else 0.
    z = gain / se if se > 0 else (np.inf if gain > 0 else -np.inf if gain < 0 else 0.)
    return dict(groups=len(ids), mean_gain=gain, ci_low=gain-1.96*se, ci_high=gain+1.96*se,
                p_one_sided=float(ndtr(-z)), positive_group_fraction=float(np.mean(mean > 0)))


@dataclass
class RidgeBundle:
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    training: dict

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(DESIGN_NAMES) or not np.isfinite(x).all():
            raise ValueError('Only exact finite pre-current own-history schema')
        return ((x-self.center)/self.scale) @ self.coefficients + self.intercept

    def record(self):
        return dict(center=self.center.tolist(), scale=self.scale.tolist(),
                    coefficients=self.coefficients.tolist(), intercept=self.intercept.tolist(),
                    training=self.training, design_names=list(DESIGN_NAMES),
                    auxiliary_targets_are_outputs_not_predictor_inputs=True)


def fit_ridge_bundle(x, auxiliary_targets, ids, groups, scope):
    scope.admit_training(np.unique(ids))
    x = np.asarray(x, dtype=float)
    y = np.asarray(auxiliary_targets, dtype=float)
    if x.shape != (len(ids), len(DESIGN_NAMES)) or y.shape != (len(ids), 15):
        raise ValueError('Exact causal input and three-horizon auxiliary output schemas required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite training data required')
    w = group_weights(groups)
    center = np.average(x, axis=0, weights=w)
    scale = np.maximum(np.sqrt(np.average((x-center)**2, axis=0, weights=w)), 1e-6)
    z = (x-center)/scale
    intercept = np.average(y, axis=0, weights=w)
    gram = z.T @ (z*w[:, None])
    gram.flat[::len(gram)+1] += 10.
    coefficients = np.linalg.solve(gram, z.T @ ((y-intercept)*w[:, None]))
    return RidgeBundle(center, scale, coefficients, intercept,
                       dict(**scope.provenance(), alpha=10., rows=len(ids), original_groups=len(np.unique(groups)),
                            actual_training_ids_sha256=ids_hash(np.unique(ids)), estimator_fit_calls=1,
                            fixed_joint_auxiliary_outputs=15, break_label_or_tau_used=False))


def feature_inputs(bank, rows, j, states):
    history = bank.history(rows, j)
    return np.column_stack([history[:, -1], history[:, -2], history[:, -4], history[:, -8], history[:, -16],
                            states['slope8'], history[:, -1]-history[:, -2],
                            states['direction_run_capped15'], states['reversal_age_capped15']]).astype(np.float32)


def auxiliary_targets(bank, rows, j, pre_direction):
    rows = np.asarray(rows, dtype=np.int64)
    if not np.array_equal(rows, bank.forecast_rows(rows, 16)):
        raise ValueError('All three future horizons must be admitted within the same series')
    current = bank.values[rows, j]
    s = bank.series_index[rows]
    result = []
    for k, horizon in enumerate(HORIZONS):
        future = bank.values[rows+horizon, j]
        direction = directions(future-current, bank.series['epsilon'][s, k, j])
        terminal = directions(future-bank.values[rows+horizon-1, j], bank.series['epsilon'][s, 0, j])
        reversal = ((pre_direction == UP) & (terminal == DOWN)) | ((pre_direction == DOWN) & (terminal == UP))
        result.extend([future, *(direction == cls for cls in range(3)), reversal])
    return np.column_stack(result).astype(np.float32)


def evaluate(bundle, x, y, current, current_direction, groups, train_y, train_current_direction, train_groups):
    prediction = bundle.predict(x)
    tw = group_weights(train_groups)
    records = []
    group_losses = []
    for k, horizon in enumerate(HORIZONS):
        column = 5*k
        target = y[:, column]
        d0a = (target-current)**2
        d0b = target**2
        d1 = (target-np.clip(prediction[:, column], -12, 12))**2
        ids, losses = grouped_means(np.column_stack([d0a,d0b,d1]), groups)
        group_losses.append(losses)
        p = np.clip(prediction[:, column+1:column+4], 1e-6, 1.)
        p /= p.sum(axis=1, keepdims=True)
        true_class = np.argmax(y[:, column+1:column+4], axis=1)
        train_class = np.argmax(train_y[:, column+1:column+4], axis=1)
        stay = float(np.clip(np.average(train_class == train_current_direction, weights=tw), 1/3+.001, .99))
        p0 = np.full_like(p, (1-stay)/2)
        p0[np.arange(len(p0)), current_direction] = stay
        loss1 = -np.log(p[np.arange(len(p)), true_class])
        loss0 = -np.log(p0[np.arange(len(p0)), true_class])
        reverse = y[:, column+4]
        rp = np.clip(prediction[:, column+4], 1e-6, 1-1e-6)
        prior = float(np.clip(np.average(train_y[:, column+4], weights=tw), 1e-6, 1-1e-6))
        reverse_loss = -reverse*np.log(rp)-(1-reverse)*np.log1p(-rp)
        reverse_prior = -reverse*np.log(prior)-(1-reverse)*np.log1p(-prior)
        record = dict(horizon=horizon, rows=len(y), groups=len(ids),
                      D0A_MSE=float(losses[:, 0].mean()), H_center_MSE=float(losses[:, 1].mean()),
                      D1_MSE=float(losses[:, 2].mean()),
                      persistence_contrast=paired_group_summary(d0a,d1,groups),
                      H_center_contrast=paired_group_summary(d0b,d1,groups),
                      direction_logloss_contrast=paired_group_summary(loss0,loss1,groups),
                      reversal_logloss_contrast=paired_group_summary(reverse_prior,reverse_loss,groups),
                      direction_accuracy=float(grouped_means(p.argmax(axis=1)==true_class,groups)[1].mean()),
                      D0C_accuracy=float(grouped_means(current_direction==true_class,groups)[1].mean()),
                      direction_persistence_train_stay=stay, reversal_train_prior=prior,
                      class_support=np.bincount(true_class,minlength=3).tolist(),
                      target_kind='AUXILIARY_FUTURE_F_ONLY', break_label_or_tau_used=False)
        records.append(record)
    return records, ids, np.stack(group_losses, axis=1)

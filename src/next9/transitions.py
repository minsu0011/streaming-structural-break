"""Frozen self movement laws and eight prefix-observable path summaries."""
from dataclasses import dataclass
import numpy as np
from src.next9.learning import fit_classifier
from src.next9.movement_screen import group_weights
from src.next9.orientation_contract import one_hot

SUMMARY_NAMES = ('stateNow', 'stateDurationCapped15', 'transitionSurprise',
    'reversalProbability', 'continuationProbability', 'stateEntropy',
    'timeSinceReversalCapped15', 'signedMomentumStrength')


def design(state, run, slope, context):
    context = np.asarray(context, dtype=float)
    if context.shape != (len(state), 6):
        raise ValueError('Exactly six fixed H context columns')
    return np.column_stack([one_hot(state, 4), run, slope, context])


@dataclass
class SelfLaw:
    variant: str
    table: np.ndarray
    classifier: object
    training: dict

    def probabilities(self, state, run, slope, context):
        state = np.asarray(state, dtype=int)
        x = design(state, run, slope, context)
        if self.classifier is not None:
            return self.classifier.predict_proba(x)
        return self.table[state]


def fit_law(state, next_state, run, slope, context, ids, groups, scope, variant):
    scope.admit_training(ids)
    state, next_state = np.asarray(state, dtype=int), np.asarray(next_state, dtype=int)
    x = design(state, run, slope, context)
    if len(next_state) != len(state) or not np.isin(next_state, np.arange(4)).all():
        raise ValueError('Aligned auxiliary next-state vocabulary required')
    if variant not in ('T0_PRIOR', 'T1_SELF_MARKOV', 'T2_SELF_RUN_SLOPE_H', 'T3_SAME_INPUT_DEPTH3'):
        raise ValueError('Unregistered transition variant')
    w = group_weights(groups)
    counts = np.full((4, 4), .5)
    if variant == 'T0_PRIOR':
        prior = np.bincount(next_state, weights=w, minlength=4)+.5
        counts[:] = prior
    else:
        np.add.at(counts, (state, next_state), w)
    table = counts/counts.sum(axis=1, keepdims=True)
    model = None
    if variant in ('T2_SELF_RUN_SLOPE_H', 'T3_SAME_INPUT_DEPTH3'):
        model = fit_classifier(x, next_state, ids, groups, scope,
            method='LOGISTIC' if variant == 'T2_SELF_RUN_SLOPE_H' else 'DEPTH3', classes=4)
    return SelfLaw(variant, table, model, dict(**scope.provenance(),
        variant=variant, smoothing=.5, next_state_is_auxiliary_target_only=True,
        break_labels_used=False, online_parameter_update=False,
        estimator_fit_calls=0 if model is None else model.training['estimator_fit_calls']))


def summaries(state, run, age, slope, epsilon, previous_issued, next_probabilities):
    """Surprise uses the law issued one step earlier, never a current refit."""
    state = np.asarray(state, dtype=int)
    n = len(state)
    previous_issued = np.asarray(previous_issued, dtype=float)
    p = np.asarray(next_probabilities, dtype=float)
    if previous_issued.shape != (n, 4) or p.shape != (n, 4):
        raise ValueError('Two causal fixed-law probability vectors required')
    for a in (previous_issued, p):
        if not np.isfinite(a).all() or np.any(a < 0) or not np.allclose(a.sum(axis=1), 1):
            raise ValueError('Finite normalized probabilities required')
    one_hot(state, 4)
    sign = np.choose(state, [-1., 0., 1., 0.])
    opposite = np.choose(state, [2, 1, 0, 3])
    reversal = np.where(np.isin(state, [0, 2]), p[np.arange(n), opposite], 0.)
    strength = sign*np.minimum(np.abs(slope)/np.maximum(epsilon, 1e-8), 12.)
    out = np.column_stack([state, run,
        -np.log(np.maximum(previous_issued[np.arange(n), state], 1e-8)),
        reversal, p[np.arange(n), state],
        -np.sum(p*np.log(np.maximum(p, 1e-8)), axis=1), age, strength])
    # UNKNOWN contributes no invented directional/path evidence.
    out[state == 3, 1:] = 0.
    if out.shape != (n, 8) or not np.isfinite(out).all():
        raise ValueError('Exactly eight finite registered summaries required')
    return out.astype(np.float32)

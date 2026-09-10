"""Pair-count weighted online-time AUC, independent of sklearn's implementation."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def ts_auc(target, prediction, time_online, *, return_details=False):
    y = np.asarray(target)
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(time_online)
    if y.ndim != 1 or p.shape != y.shape or t.shape != y.shape:
        raise ValueError("target, prediction and time_online must be equal-length vectors")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite prediction/time")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be binary")
    if np.any(t < 0) or np.any(t != np.floor(t)):
        raise ValueError("Online time must be a nonnegative integer")
    order = np.argsort(t, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(t[order])) + 1, len(t)]
    weighted_sum = 0.0
    total_pairs = 0
    details = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        if start == end:
            continue
        ix = order[start:end]
        positive = y[ix] == 1
        npos = int(positive.sum())
        nneg = len(ix) - npos
        pairs = npos * nneg
        auc = None
        if pairs:
            ranks = rankdata(p[ix], method="average")
            wins = float(ranks[positive].sum() - npos * (npos + 1) / 2)
            auc = wins / pairs
            weighted_sum += wins
            total_pairs += pairs
        details.append({"time_online": int(t[ix[0]]), "positive": npos, "negative": nneg, "weight": pairs, "auc": auc})
    score = weighted_sum / total_pairs if total_pairs else 0.5
    return (score, details) if return_details else score

def score_frames(prediction: pd.DataFrame, target: pd.DataFrame):
    """Canonicalize chronological order before the official cumcount convention.

    Official score() itself trusts within-series input order. The project always
    sorts (id,time), including at the official scorer boundary, to avoid silently
    scoring a shuffled frame as a different timeline.
    """
    for frame in (prediction, target):
        if list(frame.index.names) != ["id", "time"] or not frame.index.is_unique:
            raise ValueError("Unique (id,time) MultiIndex required")
        for name in ("id", "time"):
            if not np.issubdtype(frame.index.get_level_values(name).dtype, np.integer):
                raise ValueError("Integer id/time required")
    if list(prediction.columns) != ["prediction"] or list(target.columns) != ["target"]:
        raise ValueError("Unexpected columns")
    if not np.issubdtype(prediction.prediction.dtype, np.floating):
        raise ValueError("Floating predictions required")
    if not prediction.index.sort_values().equals(target.index.sort_values()):
        raise ValueError("Prediction/target indices differ")
    merged = prediction.sort_index().join(target)
    if ((merged.prediction < 0) | (merged.prediction > 1)).any():
        raise ValueError("Prediction outside [0,1]")
    online_t = merged.groupby(level="id", sort=False).cumcount().to_numpy()
    return ts_auc(merged.target.to_numpy(), merged.prediction.to_numpy(), online_t)

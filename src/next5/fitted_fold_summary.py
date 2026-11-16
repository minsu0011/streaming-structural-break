"""Descriptive comparisons across fixed, overlapping fitted models.

No fold-resampling uncertainty is computed here. When a seed bootstrap is
available, matching rows are averaged across artifacts before summarizing.
"""
from collections import defaultdict
import numpy as np
from src.next5.bootstrap import paired_summary


def fixed_artifact_mean(reference, candidate, expected_folds):
    expected = tuple(expected_folds)
    if set(reference) != set(expected) or set(candidate) != set(expected):
        raise ValueError('Exactly the predeclared fitted folds are required')
    old = np.stack([reference[f] for f in expected], axis=1)
    new = np.stack([candidate[f] for f in expected], axis=1)
    if old.shape != (2001, len(expected)) or new.shape != old.shape:
        raise ValueError('Original point estimate plus 2000 paired seed draws required')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Finite fixed-artifact scores required')
    if np.any((old < 0) | (old > 1) | (new < 0) | (new > 1)):
        raise ValueError('Scores must lie in [0, 1]')
    delta = new[0] - old[0]
    return {
        'fitted_artifacts': len(expected),
        'candidate_mean_artifact_TS_AUC': float(new[0].mean()),
        'reference_mean_artifact_TS_AUC': float(old[0].mean()),
        **paired_summary(old, new),
        'minimum_artifact_delta': float(delta.min()),
        'maximum_artifact_delta': float(delta.max()),
        'positive_artifacts': int(np.sum(delta > 0)),
        'negative_artifacts': int(np.sum(delta < 0)),
        'quantity': 'Mean TS-AUC of the fixed fitted artifacts, not TS-AUC of an ensemble prediction',
        'uncertainty': 'Paired generating seeds; fixed overlapping training folds and calibration thresholds',
    }


def describe_fold_values(rows, group_columns, value_column, expected):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_columns)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        identity = dict(zip(group_columns, key))
        required = set(expected[(identity['candidate'], identity['scope'])])
        folds = [int(row['fold']) for row in values]
        if len(folds) != len(set(folds)) or set(folds) != required:
            raise ValueError('Missing, duplicate or unexpected fitted fold in summary')
        numbers = np.asarray([float(row[value_column]) for row in values])
        if not np.isfinite(numbers).all():
            raise ValueError('Finite descriptive points required')
        output.append({
            **identity, 'quantity': value_column, 'fitted_artifacts': len(numbers),
            'mean_across_fixed_artifacts': float(numbers.mean()),
            'minimum_across_fixed_artifacts': float(numbers.min()),
            'maximum_across_fixed_artifacts': float(numbers.max()),
            'positive_artifacts': int(np.sum(numbers > 0)),
            'negative_artifacts': int(np.sum(numbers < 0)),
            'independent_fold_uncertainty_claimed': False,
        })
    return output

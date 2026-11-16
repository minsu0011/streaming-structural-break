"""Offline partitions of fixed scale evidence; never called by inference."""
import numpy as np


def first_detection_partition(path, tau, horizon):
    """Exhaustive G2 first-event outcome, retaining prebreak contamination."""
    values = np.asarray(path)[:horizon]
    if values.ndim != 2 or values.shape != (horizon, 45) or not 0 <= tau < horizon:
        raise ValueError('Complete fixed state path and an observed offline break required')
    qualified = np.flatnonzero(values[:, 40] > .5)
    raw_post = (values[tau:, 18] > .5).any()
    if len(qualified):
        return 'prebreak_qualified_memory' if qualified[0] < tau else 'correct_first_qualification'
    return 'postbreak_raw_confirmation_rejected' if raw_post else 'no_postbreak_raw_confirmation'


def scale_window_partition(path, tau, horizon, scale_cut):
    """Only windows whose four raw16-point blocks start at/after offline tau."""
    values = np.asarray(path)[:horizon]
    if values.shape != (horizon, 45) or not 0 <= tau < horizon or scale_cut <= 0:
        raise ValueError('Valid complete state path, offline break and fixed cutoff required')
    complete = np.flatnonzero(values[:, 15] > .5)
    z = values[complete, 9]
    first_full_point = 16 * ((tau + 15) // 16 + 4) - 1
    result = dict.fromkeys(('eligible_windows', 'neither_direction_coherent',
        'negative_coherent_but_below_raw_cut', 'positive_coherent_but_below_raw_cut',
        'negative_raw_confirmation', 'positive_raw_confirmation',
        'negative_raw_hard_ceiling_rejection', 'negative_raw_subceiling_rejection',
        'negative_raw_exceeds_admission_cut'), 0)
    for i in range(3, len(complete)):
        point = int(complete[i])
        if point < first_full_point:
            continue
        window = z[i - 3:i + 1]
        negative = int((window < 0).sum()) >= 3
        positive = int((window > 0).sum()) >= 3
        # The original kernel sums the physical ring in slot order. Preserve
        # that order, including opposing signed evidence and its rounding.
        physical = np.empty(4)
        physical[np.arange(i - 3, i + 1) % 4] = window
        strengths = []
        for direction, coherent in ((-1, negative), (1, positive)):
            total = 0.
            for value in physical:
                total += min(4., max(-4., direction * float(value)))
            strengths.append(max(0., total / 4) if coherent else 0.)
        assert float(values[point, 11]) == max(strengths) / scale_cut
        result['eligible_windows'] += 1
        if negative:
            if strengths[0] > scale_cut:
                result['negative_raw_confirmation'] += 1
                admission = float(values[point, 42])
                if admission >= 4 / scale_cut:
                    result['negative_raw_hard_ceiling_rejection'] += 1
                elif strengths[0] / scale_cut <= admission:
                    result['negative_raw_subceiling_rejection'] += 1
                else:
                    result['negative_raw_exceeds_admission_cut'] += 1
            else:
                result['negative_coherent_but_below_raw_cut'] += 1
        elif positive:
            result['positive_raw_confirmation' if strengths[1] > scale_cut else 'positive_coherent_but_below_raw_cut'] += 1
        else:
            result['neither_direction_coherent'] += 1
    assert result['eligible_windows'] == sum(result[key] for key in (
        'neither_direction_coherent', 'negative_coherent_but_below_raw_cut',
        'positive_coherent_but_below_raw_cut', 'negative_raw_confirmation', 'positive_raw_confirmation'))
    assert result['negative_raw_confirmation'] == sum(result[key] for key in (
        'negative_raw_hard_ceiling_rejection', 'negative_raw_subceiling_rejection', 'negative_raw_exceeds_admission_cut'))
    return {'first_possible_fully_postbreak_window_point': first_full_point, **result}

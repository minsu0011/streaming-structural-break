"""Reconstruct fixed-pool summaries from saved aggregate contrasts only."""
import numpy as np


def candidate_names(values):
    if isinstance(values, (str,bytes)): raise ValueError('An explicit candidate sequence is required')
    names = tuple(values)
    if not names or any(not isinstance(name,str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError('Unique nonempty candidate names required')
    return names


def summarize_saved_contrasts(contrasts, candidates):
    names = candidate_names(candidates)
    matrix = np.asarray(contrasts)
    if matrix.dtype != np.dtype('float64') or matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] != len(names):
        raise ValueError('Float64 matrix with observed row and at least two draw rows required')
    if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > 1):
        raise ValueError('Finite aggregate AUC contrasts in [-1,1] required')
    observed = matrix[0]; draws = matrix[1:]
    standard_error = draws.std(axis=0,ddof=1)
    if not np.isfinite(standard_error).all() or np.any(standard_error <= 0):
        raise ValueError('The registered nondegenerate pool requires positive draw-only sample standard errors')
    standardized = (draws-observed)/standard_error
    maximum = np.max(np.abs(standardized),axis=1)
    if not np.isfinite(maximum).all(): raise ValueError('Finite standardized errors required')
    critical = float(np.quantile(maximum,.95,method='linear'))
    rows = []
    for index,name in enumerate(names):
        estimate = float(observed[index]); se = float(standard_error[index])
        low,high = np.quantile(draws[:,index],[.025,.975],method='linear')
        lower = estimate-critical*se; upper = estimate+critical*se
        rows.append({'candidate':name,'reference':'BL2_C205_DEPENDENCE_80','estimate_delta':estimate,
            'conditional_percentile_ci_2_5':float(low),'conditional_percentile_ci_97_5':float(high),
            'conditional_standard_error':se,'probability_delta_positive':float(np.mean(draws[:,index]>0)),
            'joint_max_abs_standardized_quantile95':critical,
            'joint_conditional_envelope_lower':lower,'joint_conditional_envelope_upper':upper,
            'positive_under_joint_conditional_envelope':lower>0,
            'performance_effect_threshold':.0015,'effect_size_gate_pass':estimate>=.0015,
            'selection_adjusted':False})
    return {'rows':rows,'max_abs_standardized_error':maximum,'critical_value':critical,
        'replicates':len(draws),'candidate_count':len(names),'observed_row_excluded_from_draw_summaries':True,
        'sample_standard_error_ddof':1,'quantile_method':'linear','new_draws':0}


def load_saved_aggregate(path, expected_candidates, *, replicates=2000):
    expected = candidate_names(expected_candidates)
    if type(replicates) is not int or replicates < 2: raise ValueError('An explicit positive draw count is required')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files) != {'contrasts','candidates','max_abs_standardized_error'}:
            raise ValueError('Exact compact aggregate archive schema required')
        matrix = archive['contrasts']; names = archive['candidates']; maximum = archive['max_abs_standardized_error']
    if names.ndim != 1 or names.dtype.kind != 'U' or tuple(names.tolist()) != expected:
        raise ValueError('Saved candidate order differs from the registered pool')
    if matrix.shape != (replicates+1,len(expected)):
        raise ValueError('Saved observed/draw dimensions differ from registration')
    if maximum.dtype != np.dtype('float64') or maximum.shape != (replicates,) or not np.isfinite(maximum).all():
        raise ValueError('Saved standardized-error vector differs from registered draws')
    result = summarize_saved_contrasts(matrix,expected)
    if not np.allclose(maximum,result['max_abs_standardized_error'],rtol=0,atol=1e-14):
        raise ValueError('Saved standardized-error vector disagrees with the original contrast matrix')
    return result

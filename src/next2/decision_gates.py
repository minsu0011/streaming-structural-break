"""Literal frozen NEXT2 gates, with missing evidence separate from failed evidence."""
import math


def assess(full, reference, paired, alternates, h_regime, *, complexity_decreased=None,
           engineering_pass=None, independent_mechanism_gain=None, policy):
    checks = {}
    def check(name, value):
        checks[name] = None if value is None else bool(value)
    def finite(value):
        return value is not None and math.isfinite(float(value))
    def compare(name, value, bound):
        check(name, float(value) >= bound if finite(value) else None)
    delta = {k: full[k]-reference[k] for k in ('mean','median','worst_fold')} if full else {}
    simp, perf = policy['simplification'], policy['performance']
    for family, rule in [('simplification',simp),('performance',perf)]:
        for metric, key in [('mean','mean_delta_min'),('median','median_delta_min'),('worst_fold','worst_delta_min')]:
            compare(family+'_'+metric, delta.get(metric), rule[key])
    compare('simplification_bootstrap', paired.get('probability_delta_above_minus_0005') if paired else None,
            simp['bootstrap_probability_delta_above_minus_0005_min'])
    compare('performance_bootstrap', paired.get('probability_delta_positive') if paired else None,
            perf['bootstrap_probability_positive_min'])
    alt_complete = len(alternates) == 3 and all(finite(r.get('mean_delta')) and finite(r.get('worst_delta')) for r in alternates)
    check('simplification_alternates', all(r['mean_delta'] >= simp['alternate_mean_delta_min_each'] and
          r['worst_delta'] >= simp['alternate_worst_fold_delta_min_each'] for r in alternates) if alt_complete else None)
    check('performance_alternates', sum(r['mean_delta'] >= 0 for r in alternates) >= perf['alternate_nonnegative_count_min'] if alt_complete else None)
    h_value = h_regime.get('minimum_matched_fold_delta') if h_regime else None
    compare('simplification_h_regime', h_value, simp['h_regime_score_delta_min_each'])
    compare('performance_h_regime', h_value, perf['h_regime_score_delta_min_each'])
    check('simplification_complexity', complexity_decreased)
    check('engineering', engineering_pass)
    generation = None
    if 'mean' in delta:
        # The disjunction is literal: the >= .0035 branch needs no additional
        # independent-mechanism requirement, but still needs performance + engineering.
        if delta['mean'] >= .0035:
            generation = True
        elif delta['mean'] < .0025:
            generation = False
        elif paired is not None and alt_complete and independent_mechanism_gain is not None:
            generation = paired['probability_delta_positive'] > .95 and all(r['mean_delta'] > 0 for r in alternates) and independent_mechanism_gain
    check('generation_increment', generation)
    groups = {'distilled': [k for k in checks if k.startswith('simplification_')],
              'performance_scientific': [k for k in checks if k.startswith('performance_')]}
    groups['performance'] = groups['performance_scientific'] + ['engineering']
    groups['generation'] = groups['performance'] + ['generation_increment']
    result = {'checks': checks, 'delta': delta}
    for role, names in groups.items():
        failed = [k for k in names if checks[k] is False]
        missing = [k for k in names if checks[k] is None]
        result[role] = {'status': 'FAIL' if failed else ('PENDING' if missing else 'PASS'),
                        'failed': failed, 'missing': missing}
    return result

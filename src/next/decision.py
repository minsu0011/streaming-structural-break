"""Apply frozen promotion thresholds; missing evidence never becomes a pass."""
import math


def _finite(value):
    return value is not None and math.isfinite(float(value))


def evaluate_candidate(policy,evidence):
    promotion = policy['promotion']
    generation = policy['new_generation']
    gates = {}
    def scalar(name,value,threshold,greater=True):
        passed = None if not _finite(value) else (float(value)>=threshold if greater else float(value)<=threshold)
        gates[name] = {'status':'PENDING' if passed is None else 'PASS' if passed else 'FAIL',
            'value':value,'threshold':threshold,'comparison':'>=' if greater else '<='}
    scalar('primary_mean_delta',evidence.get('mean_delta'),promotion['mean_delta_min'])
    scalar('primary_median_delta',evidence.get('median_delta'),promotion['median_delta_min'])
    scalar('primary_worst_fold_delta',evidence.get('worst_delta'),promotion['worst_delta_min'])
    reps = evidence.get('bootstrap_replicates')
    final_bootstrap = reps == policy['bootstrap']['final_replicates']
    gates['final_bootstrap_replicates'] = {'status':'PASS' if final_bootstrap else 'PENDING',
        'value':reps,'required':policy['bootstrap']['final_replicates']}
    scalar('paired_bootstrap_probability',evidence.get('bootstrap_probability') if final_bootstrap else None,
        promotion['bootstrap_probability_min'])
    alternates = evidence.get('alternate_mean_deltas')
    alternate_complete = isinstance(alternates,(list,tuple)) and len(alternates)==3 and all(_finite(x) for x in alternates)
    scalar('alternate_nonnegative_count',sum(x>=0 for x in alternates) if alternate_complete else None,
        promotion['alternate_nonnegative_count_min'])
    scalar('historical_worst_delta',evidence.get('historical_worst_delta'),promotion['historical_worst_delta_min'])
    mechanisms = evidence.get('mechanism_deltas',{})
    required = evidence.get('required_supported_mechanisms')
    complete = bool(required) and all(name in mechanisms and _finite(mechanisms[name]) for name in required)
    scalar('supported_mechanism_worst_delta',min(mechanisms[name] for name in required) if complete else None,
        promotion['mechanism_delta_min_on_supported_category'])
    scalar('point_latency_ratio',evidence.get('point_latency_ratio'),promotion['point_latency_ratio_max'],greater=False)
    scalar('amortized_latency_ratio',evidence.get('amortized_latency_ratio'),promotion['amortized_latency_ratio_max'],greater=False)
    engineering = evidence.get('engineering_pass')
    gates['engineering'] = {'status':'PENDING' if engineering is None else 'PASS' if engineering is True else 'FAIL'}
    failures = [name for name,gate in gates.items() if gate['status']=='FAIL']
    pending = [name for name,gate in gates.items() if gate['status']=='PENDING']
    promoted = not failures and not pending
    mean = evidence.get('mean_delta')
    high_mean = _finite(mean) and mean>=generation['unconditional_mean_delta_min']
    probability = evidence.get('bootstrap_probability')
    clear_robustness = (alternate_complete and all(x>=0 for x in alternates) and sum(alternates)/3>=.0015 and
        _finite(evidence.get('historical_worst_delta')) and evidence['historical_worst_delta']>=0)
    independent_gain = (_finite(evidence.get('matched_statistic_mean_delta')) and evidence['matched_statistic_mean_delta']>=.001 and
        _finite(evidence.get('matched_supported_mechanism_gain')) and evidence['matched_supported_mechanism_gain']>=.003)
    conditional_mean = (_finite(mean) and mean>=generation['conditional_mean_delta_min'] and final_bootstrap and
        _finite(probability) and probability>=generation['bootstrap_probability_min'] and clear_robustness and independent_gain)
    return {'status':'NEW_DEV_CANDIDATE' if promoted else 'REJECTED' if failures else 'PENDING_REQUIRED_EVIDENCE',
        'promoted':promoted,'gates':gates,'failed_gates':failures,'pending_gates':pending,
        'new_generation_seal_worthy':bool(promoted and (high_mean or conditional_mean)),
        'new_generation_formula':'All promotion gates AND (mean_delta >= .004 OR (mean_delta >= .003 AND final_bootstrap_P >= .95 AND all_three_alternates_nonnegative AND alternate_mean_delta >= .0015 AND historical_worst_nonworse AND matched_new_statistic_mean_delta >= .001 AND matched_supported_mechanism_gain >= .003)).',
        'actual_seal_opening_permitted':False}

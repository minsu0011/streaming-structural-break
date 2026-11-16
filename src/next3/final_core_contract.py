"""Final guard, literal policy and immutable old-history checks without data values."""
from pathlib import Path
import datetime
import hashlib
import json
from src.next3.artifacts import digest

REQUIRED_HYPOTHESIS_KEYS = ('HYPOTHESIS', 'BOTTLENECK', 'CHANGE', 'EXPECTED_EFFECT',
    'RISKS', 'PRIMARY_METRIC', 'SECONDARY_DIAGNOSTIC', 'STOP_RULE')


def registered_hypothesis(hypothesis):
    """Respect the actual text or structured CHANGE format of locked recipes."""
    if not isinstance(hypothesis, dict):
        raise RuntimeError('A registered hypothesis must be an object')
    for key in REQUIRED_HYPOTHESIS_KEYS:
        value = hypothesis.get(key)
        if key == 'CHANGE' and isinstance(value, dict):
            if not isinstance(value.get('kind'), str) or not value['kind'].strip():
                raise RuntimeError('Structured CHANGE must name its registered kind')
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise RuntimeError('Structured CHANGE must contain finite JSON values') from error
        elif not isinstance(value, str) or not value.strip():
            raise RuntimeError('A registered hypothesis lacks nonempty '+key)
    return {'required_keys_checked': len(REQUIRED_HYPOTHESIS_KEYS),
            'structured_change': isinstance(hypothesis['CHANGE'], dict)}


def protected_files(root, expected, old_ledger_bytes):
    root = Path(root).resolve()
    if not expected: raise ValueError('Protected original file set must be nonempty')
    if not old_ledger_bytes: raise ValueError('A nonempty original ledger prefix is required')
    if hashlib.sha256(old_ledger_bytes).hexdigest() != expected.get('EXPERIMENT_LEDGER.csv'):
        raise ValueError('Original ledger prefix does not match its frozen checksum')
    exceptions = []
    for name, sha in expected.items():
        path = root/name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise RuntimeError('Protected original file missing or outside workspace: '+name)
        current = path
        while current != root:
            if current.is_symlink() or current.is_junction():
                raise RuntimeError('Protected original file now traverses a link: '+name)
            current = current.parent
        actual = digest(path)
        if name == 'EXPERIMENT_LEDGER.csv':
            if not path.read_bytes().startswith(old_ledger_bytes):
                raise RuntimeError('Original experiment ledger byte prefix changed')
            if actual != sha: exceptions.append({'path': name, 'original_sha256': sha,
                'current_sha256': actual, 'permitted_change': 'Exact original byte prefix followed by appended NEXT3 rows'})
        elif actual != sha:
            raise RuntimeError('Protected original file changed: '+name)
    return {'protected_files_checked': len(expected), 'all_old_bytes_exact_except_allowed_ledger_append': True,
        'old_ledger_prefix_preserved': True, 'append_only_exceptions': exceptions}


def literal_policy(policy):
    required = {
        'performance': {'mean_delta_min': .0015, 'median_delta_min': 0., 'worst_fold_delta_min': -.001,
            'bootstrap_probability_positive_min': .9, 'ci_lower_positive_required': True,
            'alternate_positive_count_min': 2, 'each_H_matched_delta_min': -.01,
            'supported_mechanism_min_break_series': 100, 'supported_mechanism_improvement_min': .002,
            'supported_improvement_CI_lower_positive': True, 'supported_mechanism_max_mean_loss': .005},
        'generation': {'requires_performance_and_engineering': True, 'mean_delta_min': .0025,
            'bootstrap_probability_positive_strict_min': .95, 'alternate_positive_count_min': 3,
            'worst_fold_delta_min': 0., 'new_independent_information_or_clear_robustness_repair': True,
            'especially_strong_mean_delta': .0035},
        'robustness': {'mean_delta_min': -.0008, 'worst_fold_delta_min': 0.,
            'SCALE_delta_vs_frozen_blend_min': .002, 'SCALE_CI_lower_positive': True,
            'ARCH_scale_down_delta_min_each': .01, 'ARCH_variance_factors': [.7, .5],
            'outlier_excess_alarm_delta_max': 0., 'no_supported_mechanism_mean_loss_below': -.005},
        'engineering': {'constant_state_required': True, 'streaming_parity_required': True,
            'direct_10M_seconds_ratio_max': 1.5, 'private_state_ratio_max': 2.}}
    comparisons = 0
    for group, fields in required.items():
        for key, value in fields.items():
            actual = policy.get(group, {}).get(key)
            if actual != value or isinstance(actual, bool) != isinstance(value, bool):
                raise RuntimeError('Literal preregistered policy changed: '+group+'/'+key)
            comparisons += 1
    if policy.get('static_fine_weight_search') is not False:
        raise RuntimeError('Fine weight search must remain disabled')
    if policy.get('routing_weight_bounds') != [0., .3]: raise RuntimeError('Router weight bounds changed')
    if policy.get('bootstrap', {}).get('selection_adjusted') is not False:
        raise RuntimeError('Conditional bootstrap cannot be relabeled selection-adjusted')
    if policy.get('nested_cross_fit', {}).get('outer_validation_used_for_training') is not False:
        raise RuntimeError('Outer validation must be excluded from meta training')
    return {'literal_numeric_or_boolean_gate_fields_checked': comparisons, 'thresholds_changed': False,
        'routing_and_selection_boundaries_preserved': True}


def require_fresh(record, audit_started_utc, *, now=None):
    created = datetime.datetime.fromisoformat(record['created_utc'])
    started = datetime.datetime.fromisoformat(audit_started_utc)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if created.tzinfo is None or started.tzinfo is None or now.tzinfo is None or created < started or created > now:
        raise RuntimeError('Final evidence timestamp is outside the actual separate audit interval')

"""Audit persistence improvements against a predeclared H-centered level."""
from collections import defaultdict
import numpy as np
from src.next8.common import BASE, read_json, digest
from src.next8.features import DETAILED, HORIZONS, NAMES, UNKNOWN
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.forecast_evaluation import load_forecasts
from src.next8.diagnostics import original_groups, equal_fold_loss_summary, time_H_domains
from src.next8.statistics import series_mean, holm


def fixed_center_delta(current):
    """Normalized H40 median is zero; forecast F[t+h]=0 without fitting."""
    values = np.asarray(current, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Finite observed feature values required')
    return -values


def loss_comparisons(current, target_delta, d1_delta):
    target = np.asarray(target_delta, dtype=np.float64)
    center = fixed_center_delta(current)
    own = np.asarray(d1_delta, dtype=np.float64)
    return {'PERSISTENCE_TO_H_CENTER_MSE': target ** 2 - (target - center) ** 2,
            'H_CENTER_TO_D1_MSE': (target - center) ** 2 - (target - own) ** 2,
            'PERSISTENCE_TO_H_CENTER_MAE': np.abs(target) - np.abs(target - center),
            'H_CENTER_TO_D1_MAE': np.abs(target - center) - np.abs(target - own)}


def summarize_center_control(cache, early=False):
    release = read_json(BASE / 'PREDICTION_FREEZE_NEXT8.json')
    assert release['status'] == 'FROZEN'
    rows = []
    domains = []
    panels = defaultdict(list)
    support = defaultdict(list)
    source = {}
    for fold in range(5):
        scope = early_scope(outer_scope(fold, cache.data), early)
        for j in DETAILED:
            for h in HORIZONS:
                points, target, forecasts, _, _, _ = load_forecasts(cache, scope, j, h, release)
                groups = original_groups(cache, points)
                known = target['state'] < UNKNOWN
                losses = loss_comparisons(cache.current(points, [j])[:, 0], target['delta'], forecasts['D1_delta'])
                for family, domain, mask in time_H_domains(cache, points):
                    mask = mask & known
                    if not mask.any():
                        continue
                    keys, group_losses = series_mean(groups[mask], np.column_stack(list(losses.values()))[mask])
                    for k, name in enumerate(losses):
                        item = {'feature': j, 'name': NAMES[j], 'horizon': h, 'early_only': early,
                                'outer_fold': fold, 'comparison': name, 'domain_family': family, 'domain': domain,
                                'known_H_only': True, 'series': len(keys), 'rows': int(mask.sum()),
                                'mean_loss_gain': float(group_losses[:, k].mean()),
                                'support': 'SUPPORTED' if len(keys) >= 64 else 'INSUFFICIENT_SUPPORT'}
                        domains.append(item)
                        if family == 'ALL' and len(keys) >= 64:
                            panels[j, h, name].append(group_losses[:, k].copy())
                            support[j, h, name].append(fold)
                path = BASE / 'forecasts' / scope.name / f'F{j:03d}_h{h}' / 'PREDICTIONS.npz'
                source[path.relative_to(BASE).as_posix()] = digest(path)
        print('H-center control gathered', scope.name, flush=True)
    primary = []
    for j in DETAILED:
        for h in HORIZONS:
            for name in ('PERSISTENCE_TO_H_CENTER_MSE', 'H_CENTER_TO_D1_MSE',
                         'PERSISTENCE_TO_H_CENTER_MAE', 'H_CENTER_TO_D1_MAE'):
                result = equal_fold_loss_summary(panels[j, h, name])
                passed = bool(result['status'] == 'SUPPORTED' and result['folds'] == 5 and
                              result['positive_folds'] >= 4 and result['mean_gain'] > 0 and result['ci_low'] > 0)
                record = {'feature': j, 'name': NAMES[j], 'horizon': h, 'early_only': early,
                          'comparison': name, **result, 'included_folds': support[j, h, name],
                          'all_matching_supported_folds_included': True,
                          'passed_before_Holm': passed, 'self_history_increment_beyond_fixed_H_center': False,
                          'p_for_primary_family': result['p_normal_cluster'] if passed else 1.,
                          'primary_claim': name == 'H_CENTER_TO_D1_MSE'}
                rows.append(record)
                if record['primary_claim']:
                    primary.append(record)
    assert len(primary) == 36
    for record, adjusted in zip(primary, holm([r['p_for_primary_family'] for r in primary])):
        record['Holm_p'] = float(adjusted)
        record['self_history_increment_beyond_fixed_H_center'] = bool(record['passed_before_Holm'] and adjusted < .05)
    return {'status': 'COMPLETE', 'early_only': early, 'rows': rows, 'domain_rows': domains,
            'source_prediction_sha256': source,
            'supported_self_history_increment_feature_horizons': sum(r['self_history_increment_beyond_fixed_H_center'] for r in primary),
            'comparison': 'Same frozen D1 and available heldout delta targets; fixed F future=H40 normalized median0, delta=-current_F',
            'new_model_fits': 0, 'new_classifier_candidates': 0, 'existing_D0_D5_T_model_predictions_or_gates_changed': False,
            'known_H_only': True, 'multiplicity': 'Holm36 fixed feature-horizon D1-versus-H-center MSE contrasts separately normal/early',
            'other_comparisons_and_domain_means_are_descriptive': True,
            'uncertainty': '2000 original-group draws within each original fold; equal-fold means; all supported matching folds retained',
            'zero_is_H_median_not_the_MSE_optimal_conditional_mean': True,
            'passing_does_not_prove_memory_beyond_every_H_only_or_unconditional_mean_predictor': True,
            'no_B_direction_C_gain_or_causal_mechanism_from_A': True}


LIMIT_KO = '''# 자기 이력 예측의 추가 해석 경계

D1이 현재 값을 유지하는 D0보다 좋다는 사실만으로 자기 기억의 정보량을 확정할 수 없다. 독립 잡음도 다음 값을 H 중심으로 예측하면 현재 잡음을 그대로 유지하는 것보다 오차가 작아질 수 있다. 원래 MOVEMENT_SUMMARY의 self_dynamics_supported는 사전 고정한 D0 대비 이득 판정을 뜻한다.

별도 감사에서는 같은 동결 D1과 같은 held-out target에 대해 F[t+h]=0이라는 H 중심값 기준을 대조했다. 0은 새 H60/H40 정규화에서 H40 중앙값이며, 추가 학습이나 validation 적합을 하지 않는다. delta 형태에서는 −F[t]다. H 상태가 유효한 행만 비교하고, 정상/early 각 12개 피처×3개 horizon의 D1 MSE 대조에 Holm36을 적용한다. 원래 D0–D5·T0–T4 예측, 피처 선정, M1–M6, 승격 기준은 바뀌지 않는다.

H 중심값보다 나아야 한다는 추가 비교를 통과하지 못한 D0 대비 이득은 중심 회귀라는 단순 설명을 배제하지 못한다. 통과해도 0은 MSE에 최적인 조건부 평균이라는 보장이 없으므로 모든 H-only/무조건 평균 모형을 통제했다고 주장하지 않는다. B 조건부 방향, C 구조변화 성능, 실제 인과 원인은 각각 별도 증거가 필요하다.

SELF_HISTORY_CENTER_CONTROL_NEXT8.csv에는 두 학습 범위, 모든 horizon, 지원 및 paired CI를 보존한다. MAE와 시간/H별 평균은 기술적 보조 결과이며 유리한 horizon 선택에 사용하지 않았다.
'''


"""Evidence-only scientific tables and Korean cards; no fits or score selection."""
import csv
import io
import json
from src.next8.common import BASE, atomic, read_json, digest, utc_now
from src.next8.features import DETAILED, NAMES, NORMALIZATION


def verified(directory, name):
    parent = BASE / directory
    done = read_json(parent / 'DONE.json')
    assert done['status'] == 'COMPLETE'
    assert digest(parent / name) == done['files_sha256'][name]
    return read_json(parent / name)


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table(path, rows, required=()):
    """Empty registries retain a schema; nested evidence is actual JSON."""
    columns = list(dict.fromkeys([*required, *(k for row in rows for k in row)]))
    assert columns, 'An empty CSV still needs an explicit schema'
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    atomic(path, stream.getvalue())


def number(value):
    return '미산출/지원 부족' if value is None else format(float(value), '.17g')


def interval(record, metric='mean_gain'):
    if record.get('status') != 'SUPPORTED':
        return '지원 부족 또는 미실행(0으로 대체하지 않음)'
    return (f"{number(record.get(metric))}, 95% CI [{number(record.get('ci_low'))}, "
            f"{number(record.get('ci_high'))}], 양의 fold {record.get('positive_folds', 0)}")


def feature_cards(science, movement, lineage):
    selected = sorted(set(DETAILED) | {r['feature'] for r in science['features']
                                     if r['selected_variable_TRAIN_folds'] or r['conditional_certified']})
    lines = ['# NEXT8 피처 설명 카드', '',
             '인증·보류 후보와 사전 고정 상세 조사 12개를 구분한다. 상세 조사에 포함됐다는 사실은 가변 후보 진입 또는 인증이 아니다.', '',
             '숫자는 고정된 정상 5개 outer 절차의 결과다. early-only 재현은 별도 표에 남긴다. 모든 피처는 같은 원시 시계열을 공유한다. '
             '예측 요인은 관측된 선행 정보이며 경제적 인과 원인으로 해석하지 않는다.', '']
    for j in selected:
        feature = next(r for r in science['features'] if r['feature'] == j)
        definition = lineage[j]
        own = sorted((r for r in movement['own_dynamics'] if r['feature'] == j), key=lambda r: r['horizon'])
        cross = [r for r in movement['cross_edges'] if r['destination'] == j and r['cross_factor_supported']]
        units = [r for r in science['conditional_certificates'] if r['feature'] == j and r['conditional_orientation_supported']]
        role = '인증된 조건부 방향 피처(B)' if units else ('TRAIN에서 진입한 보류 후보' if feature['selected_variable_TRAIN_folds'] else '사전 고정 상세 조사 피처; 채택된 가변 후보 아님')
        lines += [f'## F{j:03d} — {NAMES[j]}', '', f"역할: {role}. 최종 분류: `{feature['classification']}`. "
                  f"TRAIN 가변 선택 {feature['selected_variable_TRAIN_folds']}/5 fold; JOINT 인증은 자동 부여하지 않았다.", '']
        direction = []
        for unit in units:
            direction.append(f"{unit['orientation_model']}, {unit['own_or_H_condition_family']}, "
                             f"state {unit['state_pair']} → 방향 {unit['directions']}; "
                             '완전한 공통 범위·fold별 모델·지원은 VARIABLE_CERTIFICATION_REGISTRY.json 참조')
        direction_text = '; '.join(direction) if direction else 'held-out에서 인증한 상태별 양·음 방향 규칙 없음. 현재 값이나 움직임 부호로 정답 방향을 대신 정하지 않는다.'
        own_text = '; '.join(f"h={r['horizon']}: D0 MSE−D1 MSE {interval(r)}, Holm 후 자기 예측 지지={r['self_dynamics_supported']}" for r in own)
        if not own_text:
            own_text = '사전 고정 상세 12개 움직임 메뉴 밖이므로 이 보조 실험은 미실행.'
        cross_text = '; '.join(f"F{r['source']:03d}, h={r['horizon']}: 계산 계보 대조 {interval(r['mechanical_control'])}; "
                              f"해당 부모 제거 대조 {interval(r['matched_parent_removal'])}" for r in cross)
        if not cross_text:
            cross_text = '자기 이력·H·계산 계보·부모 제거·시간/H 재현을 모두 통과한 추가 요인 없음. 모든 가능한 요인이 없다는 증명은 아니다.'
        fixed_info = (f"BL 문맥에 고정 F를 추가한 그룹 평균 log-loss 이득 "
                      f"{number(feature['BL_conditioned_fixed_F_mean_loss_gain'])}, "
                      f"95% CI [{number(feature['BL_conditioned_fixed_F_gain_ci_low'])}, "
                      f"{number(feature['BL_conditioned_fixed_F_gain_ci_high'])}], "
                      f"765개 고정 대조 Holm 후 지지={feature['BL_conditioned_fixed_F_information_supported']}. "
                      '이는 재적합 문맥 대조이며 원래 BL80 대비 새 최종 모델의 TS-AUC와 같지 않다.')
        items = [
            ('무엇을 측정하는가?', definition['meaning_KO'] + ' 계산식: ' + definition['formula']),
            ('값의 음양은 무엇인가?', 'H 앞 60%로 맞춘 동일 엔진을 H 뒤 40%와 Online까지 이어 간다. F>0/F<0은 H40 중심보다 높음/낮음이다. 원래 통계량이 비음수여도 중심화한 F는 음수일 수 있다.'),
            ('상승·하락은 무엇인가?', 'MOVEMENT_DIRECTION은 F[t+h]−F[t]와 H에서 고정한 epsilon의 비교다. 음수인 F도 상승할 수 있다. VALUE_SIGN과 TARGET_ORIENTATION을 섞지 않는다.'),
            ('어떤 상태에서 양성 증거인가?', direction_text),
            ('어떤 상태에서 음성 증거인가?', direction_text),
            ('그 상태를 실시간으로 어떻게 알아내는가?', 'OWN은 F[t−1]의 H20/H80 상태, H 조건은 TRAIN에서 고정한 H 지표 구간이다. M5 조건은 t−1에 발행한 T3 h1 예측이며 현재 F[t]를 받기 전에 결정한다. 최대 확률 0.60·margin 0.15와 지원 조건을 요구하고 부족하면 UNKNOWN이다. H 조건은 시계열 안에서 일정하다.'),
            ('자기 과거만으로 얼마나 예측되는가?', own_text),
            ('어떤 기존 피처가 추가 설명력을 주는가?', cross_text),
            ('다른 가변 피처 이력이 도움이 되는가?', 'D3→D4 및 D5→D5_CROSS, 정확한 부모 제거 결과를 MOVEMENT_FACTOR_ABLATIONS.csv에 모두 보존했다. ' + cross_text),
            ('계산식으로 예상되는 관계는 무엇인가?', f"계보 그룹 `{definition['mechanical_group']}`, 직접 변환 `{definition['direct_transform_of']}`. {definition['window_or_memory']}. 필터·누적 기억으로 생기는 자기 예측과 같은 입력의 공유를 새로운 원인으로 부르지 않는다. 명시한 계보 대조가 모든 잠재 필터 상태의 충분통계라는 보장은 없다."),
            ('다른 시간/H 배경에서도 재현됐는가?', f"B 인증 단위 {len(units)}개. cross 요인은 각 기준에 고정 시간·H 구간 반복을 포함한다. 원시 방향 지도와 실제 지원은 TARGET_DIRECTION_ATLAS.csv, 상세 탈락 사유는 discovery_diagnostics/DOMAIN_FAILURES.csv에서 확인한다. 넓은 구간의 비유의성은 불변 또는 반전의 증명이 아니다."),
            ('BL80보다 추가 이득이 있는가?', fixed_info + ' 최종 M1–M6의 같은 시점 성능·paired CI는 MODEL_COMPARISON_NEXT8.csv와 PAIRED_UNCERTAINTY_NEXT8.csv에서 별도로 판정한다.'),
            ('어느 상황에서 사용하면 안 되는가?', 'H<160, H 경계 퇴화, 사전 범위 밖 H 문맥, 희소 상태, 확인되지 않은 미래, 새로운 모집단으로의 무검증 이전에는 인증 범위를 적용하지 않는다. 과거부터 반복 사용한 RP 개발 결과를 독립 hidden-test 확인으로 표현하지 않는다.'),
            ('불확실할 때 어떻게 fallback하는가?', '새 피처 branch의 짧은 H는 F=0/UNKNOWN이다. 관측 가능한 문맥이 UNKNOWN이면 해당 상태 상호작용은 계약에 따라 기권한다. M6는 TRAIN 안의 적격성 관문을 통과한 경우만 고정 95% BL80+5% M5를 쓰며 나머지는 원래 BL80이다. canonical BL80의 원래 H 조건·정규화는 유지한다.'),
        ]
        lines += [f'{i}. **{title}** {answer}' for i, (title, answer) in enumerate(items, 1)] + ['']
    return '\n\n'.join(lines)


def research_read_first(science, diagnostics):
    counts = science['counts']
    comparisons = science['normal_model_comparisons']
    models = ['| 모델 | 정상 outer 평균 TS-AUC | 원래 BL80 대비 paired Δ와 95% CI |', '|---|---:|---|']
    for method in ('M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
        row = next((r for r in comparisons if r['reference'] == 'M0' and r['candidate'] == method), None)
        models.append(f"| {method} | {number(science['normal_mean_TS_AUC'][method])} | {interval(row, 'mean_delta') if row else '기준'} |")
    answers = [
        ('A', '방향 안정 피처', str(counts['invariant_supported'])),
        ('B', '고정 비선형 피처', str(counts['nonlinear_supported'])),
        ('C', '가변 후보', str(counts['variable_candidates_ever_selected']) + '개(정상 5개 TRAIN 선택의 합집합)'),
        ('D', 'held-out 조건부 방향 피처', str(counts['conditional_features_supported']) + '개; 별도 실행 경로까지 같은 JOINT 인증 ' + str(counts['joint_variable_feature_count']) + '개'),
        ('E', '자기 이력 외 움직임 요인', str(counts['retained_cross_edges']) + '개 요약 간선; 고정된 기계적·제거 대조와 일반화 기준을 통과한 범위'),
        ('F', '움직임 요인과 의미 조건의 관계', '서로 다른 target과 검증이다. 동일하다고 가정하거나 인증을 이전하지 않았다.'),
        ('G', '상태전이망의 일반 predictor 대비 가치', str(counts['factorized_transition_supported_feature_horizons']) + '개 feature×horizon이 고정 T0/T1/T2/비동일 T4 대조를 모두 통과. 최종 M5 대 M3 및 동일 BL 혼합 대조는 별도 paired 표.'),
        ('H', '의미 방향을 미리 선택했는가?', '관측 전 선형 계수 선택은 실행 계약으로 분리했다. 현재 값에 의존하는 고정 비선형 미분까지 미리 안다고 주장하지 않는다. B 또는 JOINT 지지 여부는 D와 인증 단위 원문을 따른다.'),
        ('I', '최종 TS-AUC 개선', '아래 exact mean/paired 표 및 승격 숫자 기준 참조. 별도 최종 실행·시간·재현 감사가 끝나기 전 승격하지 않는다.'),
        ('J', '그림과 실제 증분', '8개 그림은 실제 계산·고정 사례 기준에서 만들었다. 빈 후보·빈 간선·미실행 비교는 그대로 표시하며 그림으로 A/B/C 성공을 대신하지 않는다.'),
    ]
    text = ['# NEXT8 먼저 읽기', '',
            f"과학적 판정: **{science['scientific_status_subject_to_final_integrity_audit']}**. "
            '이 문서는 연구 결과 조립본이다. 최종 시간·보호 경계·전체 replay·archive 검증의 판정은 FINAL_AUDIT_NEXT8.json 및 REPRODUCTION_NEXT8.json을 함께 확인한다.', '',
            '원래 RP 6,400개 시계열의 grouped outer 5-fold와 그 안의 discovery/cross-fit 절차를 유지했다. '
            'TRAIN에서 피처·상태·부모·방향·적격성을 정한 뒤 전체 예측을 동결하고 outer 결과를 공개한다. '
            '정상 학습과 t≤199 학습→다른 시계열 t≥400 평가를 구분했다.', '',
            *[f'**{letter}. {title}:** {answer}' for letter, title, answer in answers], '',
            '\n'.join(models), '',
            '승격의 숫자 조건은 원래 BL80 대비 평균 Δ≥0.0012, paired 95% CI 하한>0, '
            '5개 중 4개 이상 양의 fold, 최악 fold Δ≥−0.001, 충분한 주요 영역의 붕괴 없음이다. '
            '반올림으로 근접 실패를 통과시키지 않는다. 인과적 실행·계보·runtime·재현·최종 감사도 통과해야 한다.', '',
            '피처가 모두 탈락해 M1–M5가 절편 모형이 됐다면, 그 결과는 이 고정 선택 절차의 실패다. '
            '일반 비선형 모형이나 그래프 모형 전체의 표현력을 기각하는 실험은 아니다. '
            '비유의성만으로 DYNAMICS_ONLY·불변·정보 부재를 확정하지 않았다.', '',
            '자기 이력의 움직임 예측은 12개 상세 피처×3개 horizon 전체를 검사했다. D0–D5, 같은 입력의 T2, '
            '차수 일치 T4, 기계적 공유 입력, 정확한 부모 제거를 함께 보존한다. 공동 원시 입력과 누적 계산으로 생긴 예측성을 '
            '새 시장 메커니즘으로 설명하지 않는다.', '',
            '불확실성은 원래 시계열/group 단위 paired bootstrap 2,000회가 주된 기준이다. '
            '153개 원시 방향 지도의 CI는 명시한 1차 cluster 근사이며, 합성 반례의 광범위한 표는 점별 기술적 CI다. '
            'positive bootstrap fraction은 posterior가 아니다. 오래 반복 사용한 RP 개발셋은 독립 hidden 확인이 아니다.', '',
            'TRAIN 탈락 원인의 전체 자료는 discovery_diagnostics/SUMMARY.json, GATES.csv, DOMAIN_FAILURES.csv에 있다. '
            '시간 초반 또는 짧은 H에서 넓은 CI가 실용적 방향 반전을 배제하지 못한 경우와 실제 반전 지지를 구분한다.', '',
            '파일 읽는 순서: 이 문서 → RESEARCH_REPORT_NEXT8.md → FEATURE_EXPLANATION_CARDS_KO.md → '
            'MODEL_COMPARISON_NEXT8.csv / VARIABLE_CERTIFICATION_REGISTRY.json → FIGURES/NEXT8_EIGHT_FIGURES.pdf → '
            'FINAL_AUDIT_NEXT8.json / REPRODUCTION_NEXT8.json.', '']
    assert diagnostics['status'] == 'COMPLETE'
    return '\n\n'.join(text)


def build_scientific_reports():
    release = read_json(BASE / 'PREDICTION_FREEZE_NEXT8.json')
    assert release['status'] == 'FROZEN'
    science = verified('evidence_synthesis', 'SCIENTIFIC_SYNTHESIS.json')
    normal = verified('feature_certification', 'NORMAL.json')
    early = verified('feature_certification', 'EARLY.json')
    movements = [verified('movement_summary', n + '.json') for n in ('NORMAL', 'EARLY')]
    models = [verified('model_summary', n + '.json') for n in ('NORMAL', 'EARLY')]
    diagnostics = verified('discovery_diagnostics', 'SUMMARY.json')
    directory = BASE / 'science_reports'
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise RuntimeError('Existing report evidence preserved; no silent revision')
    features = science['features']
    table(directory / 'FEATURE_CLASSIFICATION_NEXT8.csv', features)
    for filename, key in [('INVARIANT_FEATURE_REGISTRY.csv', 'invariant_supported'),
                          ('STABLE_NONLINEAR_FEATURE_REGISTRY.csv', 'nonlinear_supported'),
                          ('VARIABLE_CANDIDATE_REGISTRY.csv', 'selected_variable_TRAIN_folds')]:
        table(directory / filename, [r for r in features if r[key]], ('feature', 'name', 'classification', key))
    atomic(directory / 'VARIABLE_CERTIFICATION_REGISTRY.json', {
        'status': 'COMPLETE', 'normalization': NORMALIZATION, 'normal': normal, 'early': early,
        'same_execution_JOINT_certificates': science['conditional_certificates'],
        'B_only_not_A_or_C': True, 'no_models_refit_from_outer_certificates': True})
    atlas = []
    for early_only in (False, True):
        for fold in range(5):
            scope = f'outer_{fold}' + ('_EARLY_T199' if early_only else '')
            parent = BASE / 'feature_evaluation' / scope
            done = read_json(parent / 'DONE.json')
            assert done['features'] == 153
            for j in range(153):
                key = f'F{j:03d}/DONE.json'
                assert digest(parent / key) == done['files_sha256'][key]
                record = read_json(parent / key)
                atlas.extend({'early_only': early_only, 'outer_fold': fold, **r} for r in record['atlas'])
    table(directory / 'TARGET_DIRECTION_ATLAS.csv', atlas)
    table(directory / 'CONDITIONAL_DIRECTION_RESULTS.csv', normal['conditional_units'] + early['conditional_units'],
          ('feature', 'name', 'early_only', 'orientation_model', 'conditional_orientation_supported'))
    table(directory / 'MOVEMENT_FORECAST_RESULTS.csv', [r for m in movements for r in m['fold_metrics']])
    table(directory / 'MOVEMENT_FACTOR_ABLATIONS.csv', [r for m in movements for r in m['factor_ablations']])
    transitions = []
    for m in movements:
        for r in m['transition_tests']:
            transitions.extend({**{k: v for k, v in r.items() if k != 'controls'}, **c} for c in r['controls'])
    table(directory / 'STATE_TRANSITION_RESULTS.csv', transitions)
    observability = []
    for name in ('NORMAL', 'EARLY'):
        result = verified('movement_summary', name + '_OBSERVABILITY.json')
        for r in result['features']:
            base = {'feature': r['feature'], 'name': r['name'], 'early_only': r['early_only']}
            observability.extend({**base, 'row_kind': 'FOLD_DOMAIN_METRIC', **item} for item in r['fold_metrics'])
            observability.extend({**base, 'row_kind': 'EQUAL_FOLD_PAIRED_GAIN', 'reference': ref, **item}
                                 for ref, item in r['comparisons'].items())
    table(directory / 'STATE_OBSERVABILITY_RESULTS.csv', observability)
    graph = {'status': 'COMPLETE', 'normal_summary_edges': movements[0]['summary_graph_edges'],
             'early_summary_edges': movements[1]['summary_graph_edges'], 'frozen_TRAIN_graphs': [],
             'summary_not_a_pooled_deployable_graph': True, 'causal_strength_claimed': False}
    for early_only in (False, True):
        for fold in range(5):
            scope = f'outer_{fold}' + ('_EARLY_T199' if early_only else '')
            path = BASE / 'parent_selection' / scope / 'GRAPH.json'
            assert digest(path) == release['files_sha256'][path.relative_to(BASE).as_posix()]
            graph['frozen_TRAIN_graphs'].append({'scope_name': scope, 'sha256': digest(path), **read_json(path)})
    atomic(directory / 'PREDICTIVE_FACTOR_GRAPH.json', graph)
    comparison = [r for m in models for r in m['comparisons']]
    table(directory / 'MODEL_COMPARISON_NEXT8.csv', comparison)
    table(directory / 'PAIRED_UNCERTAINTY_NEXT8.csv', comparison)
    table(directory / 'NESTED_CV_NEXT8.csv', [r for m in models for r in m['folds']])
    for filename, family in [('TIME_GENERALIZATION_NEXT8.csv', 'TIME'), ('H_BACKGROUND_RESULTS_NEXT8.csv', 'H')]:
        rows = [r for m in models for r in m['domains']
                if (r['domain_family'] == 'TIME' if family == 'TIME' else r['domain_family'] != 'TIME')]
        if family == 'TIME':
            rows += [{**r, 'domain': 'TRAIN_LE199_TEST_GE400', 'domain_family': 'EARLY_TO_LATE'} for r in models[1]['comparisons']]
        table(directory / filename, rows)
    counterexamples = []
    sd = read_json(BASE / 'synthetic_summary/DONE.json')
    for name, sha in sd['files_sha256'].items():
        path = BASE / 'synthetic_summary' / name
        assert digest(path) == sha
        counterexamples.extend({'table_family': path.stem, **r} for r in csv_rows(path))
    table(directory / 'COUNTEREXAMPLE_RESULTS_NEXT8.csv', counterexamples)
    negative = read_json(BASE / 'negative_controls/DONE.json')
    table(directory / 'NEGATIVE_CONTROLS_NEXT8.csv', negative['records'])
    for name in ('FEATURE_LINEAGE_NEXT8.csv', 'DIRECTION_RULES.json', 'STATE_TRANSITION_CONTRACT.md'):
        done = read_json(BASE / 'structural_inventory/DONE.json')
        path = BASE / 'structural_inventory' / name
        assert digest(path) == done['files_sha256'][name]
        atomic(directory / name, path.read_text(encoding='utf-8-sig'))
    mechanical = BASE / 'MECHANICAL_RELATION_AUDIT.csv'
    atomic(directory / mechanical.name, mechanical.read_text(encoding='utf-8-sig'))
    lineage = {int(r['coordinate']): r for r in csv_rows(BASE / 'FEATURE_LINEAGE_NEXT8.csv')}
    atomic(directory / 'FEATURE_EXPLANATION_CARDS_KO.md', feature_cards(science, movements[0], lineage))
    atomic(directory / '00_READ_FIRST_NEXT8_KO.md', research_read_first(science, diagnostics))
    atomic(directory / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
           'files_sha256': {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
           'scientific_synthesis_sha256': digest(BASE / 'evidence_synthesis/SCIENTIFIC_SYNTHESIS.json'),
           'all_required_reports_are_actual_results': True, 'remaining_final_audit_and_delivery_reports': True,
           'no_threshold_selection_fit_or_certificate_changed': True})


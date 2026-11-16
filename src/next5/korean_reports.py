"""Korean handoff answers sourced from completed NEXT5 scalar evidence."""
import csv
from statistics import mean, median
from src.next5.artifacts import BASE, ROOT, read_json, digest


def number(value, digits=10):
    return f'{float(value):.{digits}f}'


def interval(row):
    return '[' + number(row['ci_2_5']) + ', ' + number(row['ci_97_5']) + ']'


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def documents(context):
    timing, fits, tests = (context[k] for k in ('timing', 'fits', 'tests'))
    full = [read_json(p) for p in sorted((BASE / 'experiments').glob('*/full/RESULTS.json'))]
    assert len(full) == 19
    by_id = {r['candidate']: r for r in full}
    best = max(full, key=lambda r: r['mean'])
    assert best['candidate'] == 'X2_SCALE_TAIL' and best['mean_delta'] < .0012
    refs = read_json(BASE / 'preflight/FROZEN_REFERENCES.json')
    primary = refs['roles']['FROZEN_PRIMARY']['original_primary']
    boot = read_json(BASE / 'bootstrap/X2_SCALE_TAIL/RESULTS.json')['paired_all_time']
    fractures = {r['scope']: r for r in table(BASE / 'forensic/X2_SCALE_TAIL/PAIRED_FRACTURES.csv')}
    stress = BASE / 'synthetic/fitted_summary/complete_v1'
    stress_receipt = read_json(stress / 'RESULTS.json')
    assert stress_receipt['status'] == 'PASS' and stress_receipt['fitted_artifacts'] == 97
    score_summary_path = stress / 'FIXED_ARTIFACT_MEAN_TS_AUC.csv'
    assert digest(score_summary_path) == stress_receipt['files_sha256'][score_summary_path.name]
    score_summaries = table(score_summary_path)
    def stress_comparison(cid, scenario):
        rows = [r for r in score_summaries if r['candidate'] == cid and r['scope'] == 'full'
            and r['reference'] == 'BL80' and r['scenario'] == scenario]
        assert len(rows) == 1 and int(rows[0]['fitted_artifacts']) == 5
        row = rows[0]
        return (cid.split('_')[0] + ' ' + scenario + ' 평균Δ ' + number(row['estimate_mean_delta'])
            + ' (고정 파일 범위 ' + number(row['minimum_artifact_delta']) + '∼'
            + number(row['maximum_artifact_delta']) + ', 양수 ' + str(row['positive_artifacts']) + '/5)')
    impulse = table(stress / 'FOLD_IMPULSE_HETEROGENEITY.csv')
    long_null = table(stress / 'LONG_NULL_RESULTS.csv')
    correlations = table(BASE / 'information/SCORE_CORRELATIONS.csv')
    def rank_correlation(cid, left, right):
        rows = [r for r in correlations if r['candidate'] == cid and r['left'] == left and r['right'] == right]
        assert len(rows) == 1
        return number(rows[0]['within_fold_time_Spearman'])

    def family(letter):
        rows = [r for r in full if r['family'] == letter]
        text = '; '.join(r['candidate'] + ' Δ ' + number(r['mean_delta']) for r in rows)
        if letter == 'M':
            screen = read_json(BASE / 'experiments/M2_SIGN_RUN_EVIDENCE/screen/RESULTS.json')
            text += '; M2 screen Δ ' + number(screen['mean_delta']) + ', standalone ' + number(screen['standalone_mean']) + '로 탈락, full 미실행'
        return text + '.'

    def stability(split):
        row = read_json(BASE / 'stability/candidates/X2_SCALE_TAIL' / split / 'BOOTSTRAP/RESULTS.json')
        values = row['paired_all_time']
        return split + ' Δ ' + number(values['estimate_mean_delta']) + ', CI ' + interval(values) + ', P=' + str(values['probability_delta_positive'])

    def slice_text(scope):
        row = fractures[scope]
        return scope + ' Δ ' + number(row['estimate_mean_delta']) + ', CI ' + interval(row)

    def shock(scenario):
        records = [r for r in impulse if r['scope'] == 'full' and r['scenario'] == scenario
            and r['reference'] == 'BL80' and r['metric'] == 'mean_excess_at_horizon1024'
            and r['candidate'] in ('P1_TWO_TIMESCALE_CONFIRMATION', 'R2_CONDITIONAL_SCALE_PIT', 'X2_SCALE_TAIL')]
        assert len(records) == 3 and all(int(r['fitted_artifacts']) == 5 for r in records)
        return '; '.join(r['candidate'] + ' 평균Δ ' + number(r['mean_across_fixed_artifacts'])
            + ' (fold 범위 ' + number(r['minimum_across_fixed_artifacts']) + '∼' + number(r['maximum_across_fixed_artifacts']) + ')' for r in records)

    decomposition_path = BASE / 'impulse_decomposition/complete_v1/IMPULSE_EXCESS_DECOMPOSITION.csv'
    assert digest(decomposition_path) == context['decomposition']['files_sha256'][decomposition_path.name]
    decomposed = table(decomposition_path)
    def affine_text(scenario):
        parts = []
        for cid in ('P1_TWO_TIMESCALE_CONFIRMATION', 'R2_CONDITIONAL_SCALE_PIT', 'X2_SCALE_TAIL'):
            rows = [r for r in decomposed if r['candidate'] == cid and r['scope'] == 'full'
                and r['scenario'] == scenario and r['metric_kind'] == 'linear_summary'
                and r['metric_or_lag'] == 'mean_excess_at_horizon1024']
            assert len(rows) == 5 and {int(r['fold']) for r in rows} == set(range(5))
            parts.append(cid.split('_')[0] + ' 전체Δ ' + number(mean(float(r['total_candidate_delta_vs_BL80']) for r in rows))
                + ' = BL80 비중 감소 ' + number(mean(float(r['amplitude_only_delta_vs_BL80']) for r in rows))
                + ' + expert 추가 응답 ' + number(mean(float(r['new_expert_excess_above_affine_control']) for r in rows)))
        return '; '.join(parts)

    # Report every original late-break condition, with no new draws or fold CI.
    late_rows, late_status = [], []
    for cid in ('P1_TWO_TIMESCALE_CONFIRMATION', 'P2_SOFT_PERSISTENCE_COUNT', 'P3_SHOCK_RECOVERY'):
        grouped = {}
        for fold in range(5):
            receipt = next(r for r in context['late'] if r['candidate'] == cid and r['fold'] == fold)
            directory = BASE / 'synthetic_late_break' / cid / ('fold_' + str(fold))
            selected_tables = {}
            for name in ('LATE_BREAK_TS_AUC.csv', 'PAIRED_LATE_ALARMS.csv'):
                path = directory / name
                assert digest(path) == receipt['files_sha256'][name]
                selected_tables[name] = table(path)
            scores = [r for r in selected_tables['LATE_BREAK_TS_AUC.csv'] if r['reference'] == 'BL80']
            alarms = {(r['scenario'], r['event']): r for r in selected_tables['PAIRED_LATE_ALARMS.csv']
                if r['reference'] == 'BL80' and float(r['alpha']) == .05}
            assert len(scores) == 15 and len(alarms) == 30
            for row in scores:
                assert int(row['fold']) == fold and int(row['onset']) == 512 and int(row['paired_generating_seeds']) == 85
                scenario = row['scenario']
                grouped.setdefault(scenario, []).append((fold, float(row['estimate_mean_delta']),
                    float(alarms[(scenario, 'prebreak')]['estimate_mean_delta']),
                    float(alarms[(scenario, 'postbreak_detection')]['estimate_mean_delta'])))
        assert len(grouped) == 15
        candidate_rows = []
        for scenario, values in sorted(grouped.items()):
            assert len(values) == 5 and {v[0] for v in values} == set(range(5))
            deltas = [v[1] for v in values]
            row = {'candidate': cid.split('_')[0], 'scenario': scenario, 'mean_delta': mean(deltas),
                'min_delta': min(deltas), 'max_delta': max(deltas), 'positive': sum(v > 0 for v in deltas),
                'prebreak': mean(v[2] for v in values), 'detection': mean(v[3] for v in values)}
            candidate_rows.append(row)
            late_rows.append(row)
        late_status.append(cid.split('_')[0] + ': late 15개 조건 중 고정 파일 평균 TS-AUC Δ가 음수인 조건 '
            + str(sum(r['mean_delta'] < 0 for r in candidate_rows)) + '개')
    assert len(late_rows) == 45
    late_text = '; '.join(late_status) + '. onset=512의 기존 85개 paired seed에 한정되며 새 확증 gate가 아니다.'

    long_text = []
    for scenario in ('N0', 'N1', 'N2', 'N3'):
        values = [r for r in long_null if r['candidate'] == 'X2_SCALE_TAIL' and r['scope'] == 'full'
            and r['scenario'] == scenario and int(r['horizon']) == 16384 and float(r['alpha']) == .05]
        pairs = []
        for fold in range(5):
            new = next(r for r in values if int(r['fold']) == fold and r['model'] == 'candidate')
            old = next(r for r in values if int(r['fold']) == fold and r['model'] == 'BL80')
            pairs.append(new['null_eventual_alarm_rate'] + '/' + old['null_eventual_alarm_rate'])
        long_text.append(scenario + ': ' + ', '.join(pairs))
    runtime = context['latency']
    direct_runtime = {}
    for role in ('FROZEN_PRIMARY', 'X2_SCALE_TAIL'):
        records = [read_json(BASE / 'latency/fixed_v1' / ('repeat_' + str(block) + '_' + role + '.json')) for block in (1, 2, 3)]
        direct_runtime[role] = records
    point_times = '; '.join(role + ': point median/p95/p99 ' + '/'.join(
        number(median(r[key] for r in records), 3) for key in ('single_point_median_us', 'single_point_p95_us', 'single_point_p99_us'))
        + ' μs, H 초기화 median ' + number(median(r['historical_initialization_median_seconds'] for r in records), 6)
        + '초, 실제 10M ' + number(median(r['ten_million_seconds'] for r in records), 3) + '초'
        for role, records in direct_runtime.items())
    memory_text = '; '.join(role + ': 10M mutable state ' + str(records[0]['state_array_bytes'])
        + ' bytes, immutable tree ' + str(records[0]['immutable_tree_array_bytes']) + ' bytes, 실제 H별 state '
        + str(min(r['real_H_state_array_bytes_min'] for r in records)) + '∼'
        + str(max(r['real_H_state_array_bytes_max'] for r in records)) + ' bytes'
        for role, records in direct_runtime.items())
    runtime_text = '; '.join(r['candidate'] + ' median ' + number(r['scalar_median_ratio'], 3)
        + '배, 10M ' + number(r['ten_million_ratio'], 3) + '배, 사전 고정 median/amortized/10M의 3배 기준 ' + str(r['all_ratios_within_3x'])
        for r in runtime['summaries'])
    answers = [
        ('productive research time', number(timing['productive_research_seconds'], 3) + '초. 실제 CPU/log 또는 실질 작업 신호로 관측한 시간이며 36,000초 이상이다. 별도 final audit ' + number(timing['audit_productive_seconds'], 3) + '초는 연구 시간에 합산하지 않았다.'),
        ('wall-clock', number(timing['wall_seconds'], 3) + '초. 관측 불가 시간 ' + number(timing['unobserved_seconds'], 3) + '초를 연구 시간으로 소급 인정하지 않았다. 최종 보고서 시점의 수치이며 압축 검증 종료 시각은 Desktop 영수증에 별도 기록한다.'),
        ('family 수', 'S/P/W/E/R/M/X의 7개. S3는 reference 정보 분해로 새 candidate나 learner fit이 아니다.'),
        ('screen 수', '20개. 고정 2560개 ResearchPool subset의 fold 0/4에서 평가했다. M2 1개는 탈락했다.'),
        ('full 수', '19개. 고정 ResearchPool 6400개, grouped 5-fold. 원래 한도 20개를 초과하지 않았다.'),
        ('learner fit 수', str(fits['primitive_research_learner_fits']) + '개: screen 40, full 95, PRIMARY reference 24, label permutation 1, ALT/H reference 30, ALT/H candidate 105. 집계 결과와 unit fixture 학습은 중복 계산하지 않았다.'),
        ('시작 primary', '기존 BL2_C205_DEPENDENCE_80(BL80), next2_shared_performance_main.py. 원래 frozen 파일을 유지한다.'),
        ('시작 score', repr(primary['mean']) + '. 이전 8000개 DEV의 역사적 점수다. 이번 동일 RP 학습 범위의 BL80은 ' + repr(best['baseline_mean']) + '이며 두 모집단 점수를 직접 증감 비교하지 않는다.'),
        ('ResearchPool best', 'X2_SCALE_TAIL: mean ' + repr(best['mean']) + ', Δ ' + repr(best['mean_delta']) + '. 사전 기준 +0.0012에 ' + repr(.0012 - best['mean_delta']) + '만큼 미달한다. 반올림으로 승격하지 않았다.'),
        ('Shadow finalist 목록', '없음, 0개. 모든 full 후보가 공통 평균 증가분 기준에 미달했고 M2는 screen에서 탈락했다. 빈 manifest를 해시로 고정했다.'),
        ('Shadow scores', '없음. 미평가를 0점이나 패배로 바꾸지 않았다. Shadow 1600개에 대한 후보 평가·훈련·특징 분석을 하지 않았고 분석 경로의 반환 행은 0이다. Parquet 내부 디코딩까지 0임을 독립 계측한 것은 아니다.'),
        ('최종 selected model', 'FROZEN_PRIMARY=기존 BL80 유지. 기존 D2는 NEXT4_ROBUSTNESS의 SCALE 보조 역할로 유지한다. 새 NEXT5_PERFORMANCE/NEXT5_ROBUSTNESS는 없다. B2는 research near-miss reference다.'),
        ('최종 estimated gain', '이번에 채택하거나 주장하는 새 primary 증가분은 0이다. X2의 관측 RP 증가분을 hidden-test 개선 추정치나 채택된 증가분으로 사용하지 않는다.'),
        ('B2/D2 fusion 결과', family('S') + ' X 결합 결과: ' + family('X') + ' S4B의 분리 scale/tail 은행과 X 결합은 일부 RP 개선을 보였으나 새 역할을 얻지 못했다. 동일 RP B2 composite Δ +0.0011102871043433328, D2 composite Δ +0.0007385961732173874. S3 pair-rescue와 damage를 함께 기록했다. RP 최상위 X2의 고정 합성 감소 조건에서는 ' + stress_comparison('X2_SCALE_TAIL', 'V3') + '; ' + stress_comparison('X2_SCALE_TAIL', 'V4') + '. 원래 RP SCALE 구간의 양수 이득과 다른 배경의 감소 손실을 함께 보존하며, 이 범위를 신뢰구간이나 hidden-test 효과로 해석하지 않는다.'),
        ('scale likelihood 결과', 'S1 Gaussian과 S2 unit-variance Student-t5의 lower/upper likelihood를 분리했다. 후보 분산 배율은 0.5/0.7/1.5/2.0이며 표준편차 배율과 구별한다. ' + family('S') + ' 합성 반례의 다섯 고정 파일에서는 ' + stress_comparison('S4B_UNIFIED_SCALE_SPECIALIST', 'V3') + '; ' + stress_comparison('S4B_UNIFIED_SCALE_SPECIALIST', 'V4') + '; ' + stress_comparison('S4A_AUGMENTED_J_SCALE', 'CB_N3_050') + '. S4A의 유리한 첫 파일을 전체 학습 파일의 안정성으로 일반화하지 않는다. 범위는 신뢰구간이 아니다.'),
        ('robust null 결과', family('R') + ' H-only tail/PIT/self-normalization을 사용했다. 고정 합성 조건에서 ' + stress_comparison('R2_CONDITIONAL_SCALE_PIT', 'CB_N3_050') + '; ' + stress_comparison('R3_QUANTILE_SELF_NORMALIZED', 'V3') + '. 별도 학습 후보의 조건별 결과이며 변환 하나의 인과적 ablation은 아니다. H에서 추정한 종속 working null이므로 정확한 distribution-free calibration이나 anytime 보장을 주장하지 않는다.'),
        ('persistence 결과', family('P') + ' 5개 학습 artifact별 단발/반복 충격, late genuine break, long-null을 함께 검사했다. ' + late_text + ' 조건별 평균·범위·알람 방향은 01_GPT_HANDOFF_NEXT5.md의 표에 모두 남겼다. P3는 peak와 ungated slow EWMA를 비교하며 confirmation-only 상태는 미구현이다. P3의 결과로 그 더 넓은 구조를 반증하지 않는다. 0.9 BL80 혼합의 점수 진폭 감소와 실제 expert 추가 응답을 분해했으며 낮은 raw excess만으로 회복 개선을 선언하지 않는다.'),
        ('windowed evidence 결과', family('W') + ' 고정 길이 recent evidence를 사용한다. trimmed W2의 장점과 감소 조건별 손실을 전체 fitted-fold 표에서 함께 확인해야 한다.'),
        ('e-process 결과', family('E') + ' E1/E2는 누적·sparse-start likelihood evidence를 ranking 특성으로 사용한다. E1은 사전등록된 unreset 누적형이며, 최근 age 1–256 제한은 별도 E2에만 적용했다. 둘은 각각 학습한 후보여서 age 제한만의 인과적 ablation은 아니다. 종속성과 H nuisance 추정 때문에 엄밀한 e-process 또는 anytime-valid alpha를 주장하지 않는다. 원래 4개 long-null 코호트에서 누적 Gaussian LR를 clipped 혁신 에너지 합으로 재현했다. 최대 산술 오차는 ' + repr(context['null_drift']['maximum_cumulative_LR_identity_absolute_error']) + '다. 양의 working likelihood 비율은 학습 모델의 알람 확률이 아니며 원인의 기여를 분리한 결과로 해석하지 않는다.'),
        ('mean specialist 결과', family('M') + ' M1은 H 제외 평균 Δ -0.0010261474, 조건부 CI [-0.0020377821,-0.0000275769]로 일반화 약점을 보였다.'),
        ('new independent expert 여부', '확증된 새 독립 expert 없음. 같은 fold·time 안의 상대 순위를 모은 상관에서 X2 expert–BL80은 ' + rank_correlation('X2_SCALE_TAIL', 'BL80', 'expert') + ', X2 composite–BL80은 ' + rank_correlation('X2_SCALE_TAIL', 'BL80', 'composite') + '다. M2는 screen 상관 ' + rank_correlation('M2_SIGN_RUN_EVIDENCE', 'BL80', 'expert') + '이어도 성능이 나빠 탈락했다. X 제품항과 parent 결합의 기여를 분리한 ablation은 없으며, pair ordering·unrealized rescue는 배포 가능한 oracle 성능이 아니다.'),
        ('5-fold ResearchPool 결과', 'X2 fold 0–4: ' + ', '.join(repr(r['TS_AUC']) for r in best['fold_results']) + '. 대응 Δ: ' + ', '.join(repr(r['delta']) for r in best['fold_results']) + '. median ' + repr(best['median']) + ', worst ' + repr(best['worst']) + '. 중앙값 점수끼리의 차이는 +0.002210418424208882, 대응 fold 차이 중앙값은 +0.0010587484998473284다. 최저 점수끼리의 차이와 최소 matched-fold Δ도 별도 열로 보고하며, 원래 중앙값 차이 CI를 대응 차이 중앙값에 옮기지 않는다.'),
        ('ALT', stability('ALT_A') + '; ' + stability('ALT_B') + '. 다른 seed나 split을 추가하지 않았고, ALT 양수가 PRIMARY 기준 실패를 취소하지 않는다.'),
        ('regime', stability('H_EXCLUSION') + '. H classifier는 NEXT5 전에 전체 DEV H로 만든 고정 변환이라 역사적 transductive 요소가 있다. RP H descriptor를 새로 계산했으며 post-hoc mechanism label을 partition input으로 쓰지 않았다. 평균은 크기가 다른 5개 regime 차이의 동일 가중 평균이다. 학습 크기 2620–6267의 변화와 배경 변화가 함께 포함되며 pooled RP 점수나 순수 domain-shift 효과가 아니다.'),
        ('mechanism', slice_text('mechanism_SCALE_DOMINANT') + '; ' + slice_text('mechanism_DEPENDENCE_DOMINANT') + '. SCALE positive support 106, MEAN 16, TAIL 64. MEAN/TAIL을 supported 확증으로 주장하지 않는다. Post-hoc signature는 실제 DGP 정답이나 inference feature가 아니다.'),
        ('time fracture', slice_text('time_50_99') + '; ' + slice_text('time_100_199') + '; ' + slice_text('time_400_plus') + '. 7개 고정 구간 전체와 pair-mass 가중 기여를 보고한다.'),
        ('break age', slice_text('age_0_4') + '; ' + slice_text('age_50_plus') + '. 각 age의 positive와 같은 시간의 negative를 비교하고, 이른 구간 개선만으로 전체 TS-AUC를 대체하지 않는다.'),
        ('repeated outlier', shock('OB16') + '. 선형 분해의 고정 파일 평균: ' + affine_text('OB16') + '. 새 전체 RP 학습 파일의 결과가 아니며 진폭 감소만으로 회복을 주장하지 않는다. 기존 실제 D2 파일은 NEXT4 OB16에서 악화했던 한계를 유지한다.'),
        ('single outlier', shock('O1') + '. 선형 분해의 고정 파일 평균: ' + affine_text('O1') + '. lag 0–512, 1024/2048 horizon과 복구 censoring을 별도 기록했다. 첫 16연속 tolerance 충족은 영구 복구 보장이 아니다.'),
        ('long-null', 'X2 alpha=0.05, horizon 16384의 fold 0–4 후보/BL80 eventual alarm: ' + '; '.join(long_text) + '. 임계값은 독립 1024 calibration에서 고정했다. 16384에 같은 alpha 보장을 붙이지 않는다.'),
        ('bootstrap', 'X2 Δ CI ' + interval(boot) + ', P(Δ>0)=' + str(boot['probability_delta_positive']) + '. 원래 고정한 2000 whole-series draw만 사용했다. ALT/H는 PRIMARY-stratified multiplicity를 ID로 제한했으며 새 draw나 calibration/learner uncertainty를 포함하지 않는다.'),
        ('winner gap', 'X2−X1 +0.00010260464973255168, CI [-0.00010943783344274093,+0.000317392683116807], P=0.837. 기준 0.0007 미만이므로 TIE/UNCERTAIN이다.'),
        ('family multiplicity', '모든 family와 best-of-N 결과를 남겼다. 공통 draw의 centered max statistic은 고정 OOF 후보군에 대한 조건부 진단이다. feature 설계, screen, 전체 RP를 본 X parent 선택과 과거 DEV 탐색까지 교정하는 FWER 보장이 아니다. X2 bootstrap winner 빈도 0.6515도 진짜 최선일 확률이 아니다.'),
        ('shadow confirmation', 'NOT_EVALUATED_NO_ELIGIBLE_CANDIDATE. Shadow에서 성공하거나 실패한 것으로 해석하지 않는다. 부적격 후보를 구제하기 위해 Shadow를 소비하지 않았다.'),
        ('latency', point_times + '. 각 수치는 3개 고정 순서 블록 통계의 중앙값이다. ' + runtime_text + '. 비율은 원래 BL80 대비다. 9개 artifact × 3블록 × 실제 10M=270M 연속 업데이트를 측정했다. 반복 buffer 점수로 quality를 평가하지 않았다. 원래 D2 latency 수치는 역사적 NEXT4 측정과 구별한다.'),
        ('state memory', memory_text + '. 10M 연속 업데이트에서 state 배열 크기가 증가하지 않는지 확인했다. LATENCY_NEXT5.json에 RSS도 별도 기록하며 Python 전체 객체 크기와 숫자 배열 bytes를 혼동하지 않는다.'),
        ('leakage', 'RP admission-before-read, grouped train/held-out 분리, train-only column/weight, raw/lock SHA와 negative control을 확인했다. tau/ID/final length/future suffix는 feature API에 전달하지 않았다. 애플리케이션 guard 검증이며 OS 수준 파일 sandbox의 증명은 아니다.'),
        ('prefix invariance', '각 새 family에서 100개 이상 random prefix 검증. 대표 7개 실제 infer API에서는 각각 300개 이상 suffix/length 변이 검증과 모든 RP OOF point parity를 수행했다. 이전 역할 BL80/D2의 실제 entry와 reverse-series reset도 별도 검증했다.'),
        ('determinism', '동일 artifact의 반복 실행 fingerprint, 저장/로드와 native/compact 예측, 실제 current-point API의 OOF parity를 검증했다. 정확 일치 여부와 최대 오차는 DETERMINISM_NEXT5.json의 원 영수증을 따른다. 의미적 예측 일치와 joblib 바이트 일치를 구별한다.'),
        ('reproduction', '새 NEXT5 role이 없어 §87의 새 raw→fit 재현은 NOT_REQUIRED다. 기존 BL80/D2의 RP offline/stream parity와 BL80/B2/D2의 새 프로세스·별도 복사 payload 합성 fingerprint를 확인했다. 최종 전체 test suite ' + str(tests['tests']) + '개 통과, failure/error/skip 0. 이것을 새 raw 재학습으로 부르지 않는다.'),
        ('DEV saturation risk', 'HIGH. 이번 유한 메뉴는 종료했다. 12개 이상 가설·best gain<0.0007인 형식적 Case H는 해당하지 않지만, 과거부터 반복한 DEV 선택과 작은 winner gap의 위험이 사라지는 것은 아니다. X parent 선택은 전체 RP 점수 기반이며 nested 선택이 아니다.'),
        ('shadow contamination risk', 'NEXT5의 반환 행·캐시·훈련·평가·특징 분석 기준 Shadow 사용 0이다. 그러나 이 1600개 시계열은 과거 전체 DEV 연구에 포함됐다. 완전히 새롭고 독립적인 holdout이라고 주장하지 않는다. Arrow 내부 page/row-group 디코딩 경로 전체를 계측한 증거는 없다.'),
        ('A/B/C/D seal opened?', '아니오. 500개씩 총 2000개를 모델 평가·훈련·분석에 사용하지 않았고 guard는 전부 미개봉이다. ID 검증과 raw byte hashing은 관측/label 분석과 구별했다. 내부 Parquet 디코딩까지 0이었다는 독립 증명은 아니다.'),
        ('reduced usage', 'NEXT5 새 사용 0. 과거 누적 사용 2/2와 원래 usage ledger는 변경하지 않았다.'),
        ('public usage', 'NEXT5 public submission 0, cloud submission 0. 외부 제출을 실행하지 않았다.'),
        ('seal open recommendation', 'NO. 새 PRIMARY 기준 통과나 Shadow 확인이 없고 반복 DEV 선택 위험이 크다. LOCAL SEAL을 열지 않았다.'),
        ('다음 병목 3개', '① H/ARCH/heavy-tail 조건이 달라져도 유지되는 분산 감소 신호. ② 반복 이상치의 장기 영향과 late genuine break 검출을 함께 다루는 상태. ③ 제한된 기존 DEV에서 독립 신호의 이득을 확인할 검증력. 다음 세션에도 과거 Shadow 이력을 지우거나 이번 실패의 seed/weight 미세 탐색을 재개하지 않는다.'),
    ]
    assert len(answers) == 47
    readfirst = '# NEXT5 GPT 전달 — 완료된 연구와 검증\n\n기존 BL80 primary를 유지한다. 새 NEXT5 역할은 없고 Shadow와 네 LOCAL SEAL은 미개봉이다.\n\n'
    readfirst += '\n\n'.join(f'{i}. **{question}**\n\n{answer}' for i, (question, answer) in enumerate(answers, 1)) + '\n'
    handoff = '# Structural Break Real-Time NEXT5 인계\n\n[47개 답변](00_GPT_READ_FIRST_NEXT5.md), [모델 결과](MODEL_RESULTS_NEXT5.csv), [최종 역할](DEV_CANDIDATES_NEXT5.json), [감사](FINAL_AUDIT_NEXT5.json)를 먼저 읽는다.\n\n'
    handoff += '7개 family, 20개 screen, 19개 full, 295개 실제 learner fit을 수행했다. X2의 RP mean은 ' + repr(best['mean']) + ', Δ는 ' + repr(best['mean_delta']) + '다. 고정 기준 +0.0012에 미달해 Shadow finalist 0개를 고정했다. 새 primary나 generation을 선언하지 않았다.\n\n'
    handoff += '기존 BL80와 D2 SCALE 보조 역할, B2 near-miss reference를 보존했다. 기존 D2 실제 파일의 반복 이상치 악화도 보존한다. NEXT5 synthetic 결과는 97개 고정 CV 학습 artifact의 검증이며 새 전체 학습 deployment artifact의 결과가 아니다.\n\n'
    handoff += '표에서 PRIMARY/screen/ALT/H 제외 학습, post-hoc mechanism, 원래 8000 DEV 점수, 고정 합성 seed 결과를 구분한다. 낮은 impulse score excess는 고정 혼합에 의한 진폭 감소일 수 있다. age/lag support, 조건부 detection delay와 censoring, 1024 calibration과 16384 평가 horizon을 함께 읽는다.\n\n'
    handoff += '미개봉·0행 표시는 반환 행·캐시·훈련·평가·특징 분석과 guard 상태 기준이다. ID 조건을 사용한 Arrow scanner의 내부 Parquet 디코딩까지 0임을 독립 계측하지는 않았다. LIMITATIONS_NEXT5.md와 감사 JSON의 access_counter_scope를 함께 읽는다.\n\n'
    handoff += '[선택 한계](docs/NEXT5_SELECTION_LIMITATIONS.md), [합성 결과 읽기](docs/NEXT5_SYNTHETIC_READING_GUIDE.md), [중간 scale 반례](docs/NEXT5_RESEARCH_NOTES_03.md), [tree 사용 통계](docs/NEXT5_TREE_USAGE_NOTES.md)를 참조한다. 중간 첫 폴드 기록보다 최종 전체 artifact 표가 우선한다.\n\n'
    handoff += '[전체 폴드 합성 그래프 PDF](artifacts/next5/analysis_figures/stress_v1/COMPLETE_FITTED_STRESS_FIGURES.pdf)는 19개 full 후보의 반례 점수, X2/BL80의 긴 무변화 알람, P1/R2/X2의 충격 진폭 분해를 보여 준다. 그래프의 fold 평균·범위와 조건부 seed 구간은 서로 다른 양이다.\n\n'
    handoff += '[학습 폴드 차이 기록](docs/NEXT5_FITTED_FOLD_HETEROGENEITY_NOTES.md)은 E1의 첫 폴드 관찰이 전체 다섯 학습 파일로 어떻게 달라졌는지 보존한다. 분산 감소 반례의 부호는 엇갈렸지만 단발·반복 이상치의 평균 잔류 excess는 다섯 파일 모두 악화했다. 중간 요약보다 최종 97개 artifact 표가 우선한다.\n\n'
    handoff += '[요구사항별 증거 지도](docs/NEXT5_REQUIREMENTS_TRACEABILITY.md)에서 사양 항목과 구현·영수증 위치를 확인할 수 있다. 경로 지도 자체를 통과 판정으로 읽지 말고 최종 감사와 원래 실행 영수증을 확인한다.\n\n'
    handoff += '[긴 무변화의 evidence 진단](docs/NEXT5_NULL_DRIFT_FINDINGS.md)은 누적 E1와 sparse-start E2의 terminal feature 반응을 비교한다. 모든 오경보를 누적 증거 폭주 하나로 설명할 수 없었으며, working likelihood의 부호와 learned model의 eventual alarm을 구별한다.\n\n'
    handoff += '[동일 시간 정보 분해](docs/NEXT5_INFORMATION_FINDINGS.md)는 증가 credit과 손상 credit이 상당 부분 상쇄되는 모습을 보여 준다. X2의 순증가는 두 양의 차이이며, 거의 무상관인 M2의 실패도 보존한다. 작은 상관이나 미실현 rescue를 유용한 독립 신호의 증명으로 바꾸지 않는다.\n\n'
    handoff += '[속도·상태 측정 읽기](docs/NEXT5_RUNTIME_READING_GUIDE.md)는 세 번의 고정 순서 측정, 3배 판정의 정확한 대상, 반복 버퍼와 실제 표본, mutable state와 프로세스 RSS의 차이를 설명한다.\n\n'
    handoff += '원래 코드·staged 변경·NEXT3 사용자 중단·NEXT4 완료 상태는 보존됐다. 마지막 ledger는 원래 276개 행과 모든 바이트를 prefix로 유지하고 NEXT5 315개 행을 추가했다. 실제 fit 295개는 한 번씩만 센다.\n\n'
    handoff += '초기 preflight 실행 옵션 오류 1건과 이에 딸린 미실행 작업 3건은 QUARANTINED 상태로 보존하고 복구 영수증을 추가했다. checkpoint의 failure_count=4를 학습 모델 4개가 충돌했다는 뜻으로 읽지 않는다. M2 screen 실패와 19개 full 승격 기준 실패는 정상 완료된 과학적 평가다. 개발 중 summary schema와 package 필터 수정 등은 maintenance 영수증에 구별해 기록했다.\n\n'
    handoff += '이후 실제 API 두 건에서 import-time Numba 캐시 생성을 쓰기 차단 장치가 막았다. 미실행 후속 다섯 건도 보존했고, src.next5.readonly_streaming 진입점으로 7개 전체 RP 검사를 새 경로에서 완료했다. 원래 core/모델/수치 검증은 그대로다. 최종 quarantine 11개는 실제 실행 오류 3개와 미실행 대체 8개를 뜻하며 모두 성공 영수증으로 해결됐다. 보존된 초기 checkpoint의 4개와 시간 범위가 다르다.\n\n'
    handoff += '[읽기 전용 import 복구 기록](docs/NEXT5_READONLY_IMPORT_REPAIR.md)은 최초 발견과 중간 검증 범위를 설명한다. 최종 완료 근거는 maintenance/READONLY_IMPORT_REPAIR/RESOLVED.json과 후보별 API_IMPORT_REPAIR.json이다.\n\n'
    handoff += '[최종 실패 분류표](FAILURE_CLASSIFICATION_NEXT5.csv)의 32개 행은 실제 실행 오류 4건(Python queue 3건과 native 기록 오류 1건), 시작 전 대체 8건, screen 탈락 1건, full 승격 기준 탈락 19건을 구분한다. 초기 24개 행에 API 관련 7개와 병목 검토 게시 오류 1개를 추가했다.\n\n'
    handoff += '[Native 검토 기록 복구](docs/NEXT5_NATIVE_REVIEW_REPAIR.md)는 PowerShell 파일 교체 오류와 복구를 설명한다. 012 history는 작성됐지만 BOARD 게시가 실패한 부분 기록이다. 이를 보존하고 native fixture를 통과한 별도 v2로 013 검토를 게시했다. 점수·우선순위·측정 worker는 바꾸지 않았고 다른 Python 프로세스를 실행하지 않았다.\n\n'
    handoff += '합성 paired alarm의 원래 표에는 N0–N3 counterexample와 long_null의 코호트 종류 열이 빠져 있었다. 전수 집계를 실행하기 전에 동결 순서와 SHA 영수증으로 종류를 복원했다. 원래 97개 폴드 표의 모든 수치는 보존됐으며, 최종 표의 cohort_kind를 포함해 비교해야 한다. 이 수정은 새 fit·예측·bootstrap draw를 만들지 않았다.\n\n'
    handoff += '## 늦은 실제 변화: P 계열의 모든 고정 조건\n\n'
    handoff += '아래 45개 행은 후보 3개 × 원래 변화 조건 15개다. onset=512, horizon=1024, 조건별 동일한 85개 paired seed를 사용했다. 각 행의 TS-AUC Δ 평균·최소·최대와 양수 파일 수는 겹치는 다섯 고정 학습 파일의 기술적 통계이며 앙상블 점수나 독립 fold 신뢰구간이 아니다. 알람 Δ도 다섯 파일 평균으로, α=0.05의 원래 calibration 임계값을 그대로 사용했다. prebreak는 음수가, postbreak 첫 탐지는 양수가 유리하다. 조기 알람이 먼저 발생한 시계열은 postbreak 첫 탐지 성공에 포함하지 않는다.\n\n'
    handoff += '| 후보 | 조건 | TS-AUC Δ 평균 | 최소 | 최대 | 양수/5 | prebreak 알람 Δ | postbreak 첫 탐지 Δ |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for row in late_rows:
        handoff += '| ' + ' | '.join([row['candidate'], row['scenario'], number(row['mean_delta']),
            number(row['min_delta']), number(row['max_delta']), str(row['positive']),
            number(row['prebreak']), number(row['detection'])]) + ' |\n'
    handoff += '\n근거는 artifacts/next5/synthetic_late_break/의 원래 15개 완료 영수증과 CSV SHA다. 기존 2,000개 seed multiplicity만 제한해서 사용했으며, 이 표에는 새로운 합산 신뢰구간을 만들지 않았다. 85개 seed는 사전 진단 support 100개 미만이고 이 사후 층화에 승격 권한은 없다.\n\n'
    handoff += '전체 ZIP은 raw competition data, seal 값/label, credentials, real point OOF, feature cache와 이전 ZIP을 제외한다. SOURCE_MANIFEST_NEXT5.json과 외부 Desktop 검증 영수증에 파일·archive SHA가 있다. Raw 없이 이전 역할의 inference를 재생할 수 있으며 학습 재현에는 별도 hash-matched raw가 필요하다.\n'
    return {'00_GPT_READ_FIRST_NEXT5.md': readfirst, '01_GPT_HANDOFF_NEXT5.md': handoff}

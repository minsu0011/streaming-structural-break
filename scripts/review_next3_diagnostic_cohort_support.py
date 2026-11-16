"""Hypothetical future-cohort support arithmetic from existing DEV aggregates."""
from pathlib import Path
from fractions import Fraction
import csv
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scipy
from scipy.stats import binom
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def finite_log_tail(n, p, k):
    """Independent finite PMF sum: no scipy special/statistical implementation."""
    terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
             + i*math.log(p)+(n-i)*math.log1p(-p) for i in range(k, n+1)]
    largest = max(terms)
    return largest+math.log(math.fsum(math.exp(value-largest) for value in terms))


def main():
    protocol_path = BASE/'maintenance/DIAGNOSTIC_COHORT_SUPPORT_PROTOCOL.json'
    protocol = read_json(protocol_path)
    assert protocol['status'] == 'REGISTERED_EXPLANATORY_MAINTENANCE'
    for name, sha in protocol['source_sha256'].items(): assert digest(ROOT/name) == sha
    policy = read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    minimum = policy['performance']['supported_mechanism_min_break_series']
    assert type(minimum) is int and minimum == 100
    total = read_json(BASE/'preflight/NEXT3_PREFLIGHT.json')['DEV_series']
    assert total == 8000
    table_path = BASE/'final_tables/prepared_v2/MECHANISM_RESULTS_NEXT3.csv'
    with table_path.open(encoding='utf-8-sig', newline='') as stream:
        original = [row for row in csv.DictReader(stream)
                    if row['candidate'] == 'BL2_C205_DEPENDENCE_80'
                    and row['record_type'] == 'INHERITED_NEXT2_BLEND_VS_J102']
    counts = {}
    for row in original:
        value = float(row['break_series'])
        assert value.is_integer() and value > 0 and row['scope'] not in counts
        counts[row['scope']] = int(value)
    assert len(counts) == 6 and sum(counts.values()) == 3944
    assert {float(row['null_series']) for row in original} == {4056.}
    assert sum(counts.values())+4056 == total
    assert abs(math.exp(finite_log_tail(4, .5, 3))-5/16) < 1e-15
    assert float(binom.sf(2, 4, .5)) == 5/16
    out = BASE/'maintenance/diagnostic_cohort_support_v1'
    out.mkdir(parents=True, exist_ok=False)
    rows = []; scale_rows = []; max_error = 0.
    for mechanism in protocol['mechanisms']:
        count = counts['mechanism_'+mechanism]
        base_p = Fraction(count, total)
        for multiplier_text in protocol['composition_multipliers']:
            multiplier = Fraction(multiplier_text)
            rational_p = base_p*multiplier
            p = float(rational_p)
            assert 0 < p < 1
            expected_minimum_n = math.ceil(Fraction(minimum, 1)/rational_p)
            assert (expected_minimum_n-1)*rational_p < minimum <= expected_minimum_n*rational_p
            scale_rows.append({'mechanism':mechanism, 'existing_DEV_break_series':count,
                'existing_DEV_total_series':total, 'composition_multiplier':multiplier_text,
                'illustrative_p':p, 'minimum_N_for_expected_100':expected_minimum_n})
            previous = -1.
            for n in protocol['hypothetical_cohort_sizes']:
                probability = float(binom.sf(minimum-1, n, p))
                log_probability = float(binom.logsf(minimum-1, n, p))
                independent = finite_log_tail(n, p, minimum)
                error = abs(log_probability-independent); max_error = max(max_error,error)
                assert error <= protocol['absolute_log_probability_tolerance']
                assert 0 <= probability <= 1 and probability >= previous
                previous = probability
                rows.append({'mechanism':mechanism, 'existing_DEV_break_series':count,
                    'existing_DEV_total_series':total, 'composition_multiplier':multiplier_text,
                    'illustrative_p':p, 'hypothetical_new_independent_cohort_N':n,
                    'expected_break_series':float(n*rational_p), 'diagnostic_support_rule':minimum,
                    'binomial_probability_at_least_100':probability,
                    'log_probability':log_probability,'independent_log_tail':independent,
                    'absolute_log_probability_difference':error})
    assert len(rows) == 48 and len(scale_rows) == 12
    for name, data in [('HYPOTHETICAL_SUPPORT_GRID.csv',rows), ('EXPECTED_SUPPORT_SCALE.csv',scale_rows)]:
        with (out/name).open('x',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    report = {'status':'PASS_HYPOTHETICAL_INDEPENDENT_COHORT_SUPPORT_ARITHMETIC',
        'created_utc':utc_now(),'hypothetical_grid_rows':len(rows),'expected_scale_rows':len(scale_rows),
        'maximum_absolute_log_probability_difference':max_error,
        'deterministic_toy_control':'P(Binomial(4,0.5)>=3) = 5/16',
        'existing_DEV_series':total,'existing_break_counts':counts,'diagnostic_minimum':minimum,
        'scipy_version':scipy.__version__,'protocol_sha256':digest(protocol_path),
        'source_sha256':protocol['source_sha256'],
        'files_sha256':{name:digest(out/name) for name in ['HYPOTHETICAL_SUPPORT_GRID.csv','EXPECTED_SUPPORT_SCALE.csv']},
        'new_fits':0,'new_Monte_Carlo_draws':0,'new_quality_scores':0,'new_model_families':0,
        'real_observations_labels_or_predictions_read':False,'sealed_partition_composition_analyzed':False,
        'assumptions_verified_for_any_future_cohort':False,'TS_AUC_power_analysis':False,
        'new_promotion_or_seal_rule':False,'selection_adjusted':False,
        'scope':'Deterministic hypothetical IID Bernoulli category-count arithmetic using existing DEV summary proportions as illustrative inputs. The multiplier grid is not a confidence interval. No inference about any unopened sealed partition; 100 is an existing diagnostic support convention, not a power theorem.'}
    atomic_json(out/'RESULTS.json',report,immutable=True)
    base = [row for row in scale_rows if row['composition_multiplier'] == '1']
    selected = [row for row in rows if row['composition_multiplier'] == '1']
    document = '''# NEXT3 후속 독립 데이터의 진단 표본 규모

기존 DEV의 mechanism별 break-series 수를 이용해 **가상의 새 독립 데이터**에서 진단 표본이 얼마나 모일 수 있는지 계산했다. 현재 미개봉 seal의 구성·확률·결과를 분석한 것이 아니며, 그 데이터를 읽지 않았다. 새 TS-AUC 평가, 학습, bootstrap, Monte Carlo 추출은 모두 0회다.

## 계산의 전제와 목적

현재 DEV 8,000개 중 break-series 3,944개와 null-series 4,056개는 기존 집계표에서 확인했다. 아래 비율의 분모는 break만의 수가 아니라 전체 8,000개다. mechanism 분류는 기존 관측 기반 진단이므로 잠재 생성 과정의 정확한 종류로 간주하지 않는다.

가상의 새 독립 series마다 특정 진단 범주에 속할 확률이 고정 p이고 범주 지시자가 IID Bernoulli라는 전제를 **추가로 가정할 때만**, N개에서 그 범주의 개수 X는 Binomial(N,p)가 된다. 기존 DEV 비율을 p의 예시 입력으로 사용하고 0.5/1/2배의 사전 고정 민감도 표를 함께 만들었다. 이 세 배수는 추정 불확실성을 포함한 신뢰구간이 아니며, 미래 데이터에 이 전제가 성립하는지는 확인하지 않았다.

현재 정책의 100 break-series는 진단 결과를 supported로 표시하는 최소 기준이다. 이 숫자에 도달해도 효과 크기 +0.002의 검출, 좁은 CI, 낮은 선택 편향 또는 일반화가 보장되지 않는다. 별도 평가의 새 승격 기준이나 seal 개봉 기준을 만든 것도 아니다.

| 기존 진단 범주 | DEV break 수 / 전체 8,000 | 예시 p | 기대 개수가 100에 처음 도달하는 N |
|---|---:|---:|---:|
''' + ''.join(f"| {r['mechanism']} | {r['existing_DEV_break_series']} / 8000 | {r['illustrative_p']:.6f} | {r['minimum_N_for_expected_100']:,} |\n" for r in base) + '''
마지막 열은 `ceil(100/p)`이며 **100개 확보 확률 95%에 필요한 N이 아니다**. 기대값과 확보 확률은 서로 다르다.

## 사전 고정 규모별 조건부 계산

N은 1,000/2,000/5,000/10,000으로 고정했다. 범주별 계산은 주변 분포의 민감도 예이며, 각 범주를 동시에 충족할 확률이나 범주 간 독립성을 가정하지 않는다.

| 범주 | 가상 N | 기대 break 수 | IID 전제 아래 P(X ≥ 100) |
|---|---:|---:|---:|
''' + ''.join(f"| {r['mechanism']} | {r['hypothetical_new_independent_cohort_N']:,} | {r['expected_break_series']:.3f} | {r['binomial_probability_at_least_100']:.9g} |\n" for r in selected) + '''
[SciPy 공식 binomial 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)의 PMF와 survival function을 사용했다. `sf(99,N,p)`와 표준 라이브러리 `lgamma`를 이용해 100부터 N까지 PMF를 직접 합산한 로그 확률을 48개 조합에서 대조했다. 허용 로그 확률 절대 오차는 사전에 1e-8로 고정했으며, 손계산 가능한 4회·p=0.5·3개 이상인 5/16도 확인했다. 문서 표시는 SciPy 1.18.0이고 실제 사용한 고정 환경 버전은 결과 JSON에 기록했다.

## 후속 연구 설계에 남기는 제약

희소 진단 범주에서 100개를 모으는 문제와 전체 TS-AUC 차이를 평가하는 문제는 구분해야 한다. TS-AUC는 같은 online time에서 비교 가능한 양성·음성 pair, horizon, series 내부 의존, 두 모델의 paired score 차이까지 영향을 받으므로 위 범주 개수만으로 검정력을 계산할 수 없다. 기존 DEV 집계로 미래 효과 크기나 pair 분포가 같다고 가정하지 않는다.

후속 독립 평가를 설계할 때는 대상 모집단, series 단위 독립성, 전체 TS-AUC와 진단의 우선순위, 효과 기준, 분석 횟수와 고정 모델을 데이터 확인 전에 정해야 한다. 희소 범주를 더 많이 수집하는 표본 설계는 원래 분포와 달라질 수 있으므로 전체 점수와 가중·조건부 해석을 별도 검토해야 한다. 이 문서는 데이터 수집, seal 개봉, 정책 변경을 실행하거나 승인하지 않는다.

전체 48행과 12행 규모표, 입력 SHA 및 0회 평가 기록은 `artifacts/next3/maintenance/diagnostic_cohort_support_v1/`에 있다. 기존 조건부 bootstrap을 선택 보정된 증거로 바꾸지 않으며 B4의 일반화 불확실성은 계속 미해결이다.
'''
    path = ROOT/'docs/NEXT3_DIAGNOSTIC_COHORT_SUPPORT.md'
    with path.open('x',encoding='utf-8',newline='\n') as stream: stream.write(document)
    atomic_json(out/'REPORT.json',{'status':'PASS','created_utc':utc_now(),
        'document':path.relative_to(ROOT).as_posix(),'document_sha256':digest(path),
        'result_sha256':digest(out/'RESULTS.json')},immutable=True)
    progress('Hypothetical diagnostic cohort support verified',
        '48 deterministic binomial/finite-PMF joins; existing DEV summaries only; no future/seal composition claim, TS-AUC power analysis, new fits or draws')
    print('NEXT3 HYPOTHETICAL COHORT SUPPORT PASS',flush=True)


if __name__ == '__main__': main()

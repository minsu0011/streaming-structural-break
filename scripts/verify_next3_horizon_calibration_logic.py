"""Exact rank examples explain the fixed-horizon alarm threshold's scope."""
from pathlib import Path
import csv
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next2.synthetic_analysis import threshold


def exact_leave_one_out(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64); n = len(scores)-1
    rank = math.ceil((n+1)*(1-alpha)); alarms = 0; extended = 0
    for heldout in range(n+1):
        calibration = np.delete(scores, heldout)
        level = threshold(calibration, alpha)
        expected = None if rank > n else float(sorted(calibration)[rank-1])
        assert level == expected
        alarms += int(level is not None and scores[heldout] > level)
        # A bounded toy null score jumps to one only after calibration horizon.
        # No trained model, actual stream or project's alarm rate is computed.
        extended += int(level is not None and 1. > level)
    bound = max(0, n+1-rank)
    assert alarms <= bound
    return {'calibration_n': n, 'alpha': alpha, 'rank': rank, 'enumerated_test_ranks': n+1,
        'alarm_ranks': alarms, 'rank_bound_count': bound, 'marginal_alarm_fraction': alarms/(n+1),
        'extended_toy_alarm_ranks': extended, 'threshold_absent_means_no_alarm': rank > n}


def main():
    protocol_path = BASE/'maintenance/HORIZON_CALIBRATION_REVIEW_PROTOCOL.json'
    protocol = read_json(protocol_path)
    assert protocol['status'] == 'REGISTERED_EXPLANATORY_MAINTENANCE'
    for name, sha in protocol['source_sha256'].items(): assert digest(ROOT/name) == sha
    out = BASE/'maintenance/horizon_calibration_logic_v1'; out.mkdir(parents=True, exist_ok=False)
    suite = read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    assert suite['calibration_null_series_per_regime'] == 1000 and suite['online_length'] == 1024
    unique = np.arange(1001, dtype=np.float64)/1002.
    examples = [dict(name='unique_alpha05', **exact_leave_one_out(unique, .05)),
                dict(name='unique_alpha01', **exact_leave_one_out(unique, .01)),
                dict(name='tied_alpha05', **exact_leave_one_out(np.floor(unique*20)/20, .05)),
                dict(name='small_sample_infinite_threshold', **exact_leave_one_out(np.arange(10)/11., .05))]
    assert examples[0]['alarm_ranks'] == 50 and examples[1]['alarm_ranks'] == 10
    assert examples[0]['extended_toy_alarm_ranks'] == examples[1]['extended_toy_alarm_ranks'] == 1001
    assert examples[3]['alarm_ranks'] == examples[3]['extended_toy_alarm_ranks'] == 0
    table_path = BASE/'final_tables/prepared_v2/LONG_NULL_RESULTS_NEXT3.csv'
    with table_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['candidate'] == 'BL2_C205_DEPENDENCE_80'
                and row.get('regime') == 'N0' and row.get('alpha') == '0.05']
    assert len(rows) == 3
    rows.sort(key=lambda row: int(row['horizon']))
    assert [int(row['horizon']) for row in rows] == [1024, 4096, 16384]
    assert len({row['threshold'] for row in rows}) == 1
    for row in rows:
        assert row['record_type'] == 'INHERITED_NEXT2_FROZEN_SYNTHETIC_DIAGNOSTIC'
        assert int(row['series']) == 512 and int(row['eventual_alarm_count'])/512 == float(row['eventual_alarm_rate'])
        assert not row['ts_auc']
    rate = examples[0]['marginal_alarm_fraction']
    report = {'status': 'PASS_EXPLANATORY_RANK_ALGEBRA', 'created_utc': utc_now(),
        'exact_leave_one_out_examples': examples, 'new_Monte_Carlo_draws': 0,
        'hypothetical_16_blocks': {'per_block_marginal_rate_if_premise_holds': rate,
            'union_bound_if_every_block_marginal_is_bounded': min(1., 16*rate),
            'independent_events_formula_if_independence_also_holds': 1-(1-rate)**16,
            'premises_verified_for_current_carried_state_process': False,
            'not_a_forecast_of_actual_model_alarm_rate': True},
        'inherited_N0_counts': [{key: row[key] for key in ('horizon', 'series', 'threshold',
            'eventual_alarm_count', 'eventual_alarm_rate', 'record_type')} for row in rows],
        'new_fits': 0, 'real_observations_labels_or_predictions_read': False,
        'synthetic_prediction_arrays_read': False, 'thresholds_recalibrated': False,
        'source_sha256': protocol['source_sha256'], 'protocol_sha256': digest(protocol_path),
        'original_summary_table_sha256': digest(table_path),
        'scope': 'Deterministic rank enumeration and logical horizon counterexample only; inherited table counts are transcribed and divided, not newly evaluated. No new model, alarm metric, quality score, Monte Carlo draw or time-uniform validity claim.'}
    atomic_json(out/'RESULTS.json', report, immutable=True)
    text = f'''# NEXT3 고정 horizon 경보 보정의 범위

기존 0.05 임계값은 1,024점 구간의 최대 점수를 보정한다. 이 보정을 16,384점 또는 무한 관측 구간의 5% 경보 보장으로 해석하지 않는다. 아래는 기존 알고리즘의 수학적 범위를 점검한 문서이며 새 경보 정책이나 모델 결과가 아니다.

## 기존 코드와 rank 계산

`COUNTEREXAMPLE_SUITE_LOCK.json`은 배경별 calibration series 1,000개, calibration horizon 1,024, 평가 horizon 1,024/4,096/16,384를 고정했다. `src/next2/synthetic_analysis.py::threshold`는 각 calibration series의 최대 점수 M_i를 정렬해 `k = ceil((n+1)(1-alpha))`번째 값을 사용한다. 경보 비교는 strict `score > threshold`다. k가 n보다 크면 `None`으로 경보를 내지 않으며, 이는 임계값을 무한대로 두는 것과 같다.

고정 점수 함수에 대해 calibration 최대값들과 새 평가 최대값이 교환가능하면, 새 값의 순위 논리로 `P(M_new > M_(k)) <= (n+1-k)/(n+1) <= alpha`가 성립한다. 이 확률은 calibration 표본과 새 평가 사례의 무작위성을 함께 평균한 marginal 값이다. 특정 calibration 집합, 특정 H 또는 선택한 모든 모델에 각각 같은 보장을 주는 조건부 명제가 아니다. 동점은 strict 비교에서 보수적으로 처리된다. 일반적인 split conformal rank 논리와 marginal 해석은 [Angelopoulos·Bates, v6, §1 및 부록 D](https://arxiv.org/html/2107.07511v6)를 따른다.

기존 threshold 구현을 고정한 채 서로 다른 점수 1,001개의 test 순위를 모두 한 번씩 제외해 직접 대조했다. alpha=0.05에서는 k=951, 경보 순위 50/1,001 = **{50/1001:.12f}**이고 alpha=0.01에서는 k=991, 10/1,001 = **{10/1001:.12f}**다. 동점 예와 표본 수 9의 `None` 경계도 검증했다. 새 bootstrap이나 Monte Carlo draw는 0개이며 실제 모델의 예측 배열을 쓰지 않았다.

## 긴 관측 구간으로 자동 확장되지 않는 이유

M_T를 첫 T개 점수의 최대값이라 하면 T가 커질수록 M_T는 감소하지 않는다. calibration은 M_1024의 분포를 기준으로 하지만, M_16384의 분포가 같다는 가정은 없다. 프로젝트 자체의 논리 대조로 첫 1,024점은 위 rank 점수를 유지하고 그 이후 점수만 1이 되는 가상의 null 과정을 만들었다. 첫 horizon에서는 50/1,001의 경보 순위가 나오지만 긴 horizon에서는 1,001/1,001이 경보를 낸다. 이는 alpha에서 1로 증가할 수 있다는 가능성의 반례이며, 실제 frozen 모델의 예측이나 수치 전망이 아니다.

서로 다른 16개 block 각각에 동일한 marginal 경보 상한을 별도로 입증했다면 union bound를 적용할 수 있다. block 사건의 독립성까지 가정하면 `1-(1-p)^16`을 계산할 수 있다. 현재 detector는 recurrent state를 이어가고 score의 나이 효과도 있으므로, block별 동일 분포·상한·독립성의 전제를 확인하지 않았다. 이 두 공식을 현재 모델의 실제 장기 FPR 예측으로 사용하지 않는다.

## 원래 관측의 해석

기존 BL80의 N0 평가 512개에서 동일 임계값 **{rows[0]['threshold']}**를 사용한 원래 기록은 아래와 같다. 새 평가를 수행하지 않고 보존된 count/rate의 나눗셈만 확인했다.

| horizon | 원래 경보 수 / series | 원래 eventual alarm rate |
|---:|---:|---:|
''' + ''.join(f"| {row['horizon']} | {row['eventual_alarm_count']} / 512 | {row['eventual_alarm_rate']} |\n" for row in rows) + '''
긴 구간의 경보율 상승은 미해결 운영 약점이다. 동시에 1,024점 horizon의 marginal calibration 명제 자체를 이 긴 구간의 숫자만으로 반박했다고 쓰지 않는다. 고정 학습 모델, 일치하는 synthetic 배경 및 whole-series 교환가능성의 가정과 반복적인 진단·모델 선택의 영향을 따로 검토해야 한다. 경쟁 점수 TS-AUC의 개선 여부와도 별개다.

무한 또는 임의 중단 시각까지 유효한 경보를 주장하려면 추가 수학적 구조와 가정이 필요하다. [Howard 등, 비음수 supermartingale에 의한 time-uniform bound, v8](https://arxiv.org/html/1808.03204v8)는 그런 구조를 다루지만, 현재 학습된 확률 점수·Bayes 이름의 feature·plug-in H 보정이 그 조건을 만족한다고 입증한 것은 아니다. 이 문서에서 e-process나 새 threshold schedule을 구현·채택하지 않았다.

후속 B6 검증은 비교 horizon, calibration/evaluation 분리, late change의 clean detection, premature alarm, delay와 TS-AUC를 함께 사전 고정해야 한다. FPR만 낮추는 score 압축을 실질 개선으로 인정하지 않는다. 재현 근거는 `artifacts/next3/maintenance/horizon_calibration_logic_v1/RESULTS.json`이다.
'''
    document = ROOT/'docs/NEXT3_HORIZON_CALIBRATION_SCOPE.md'
    with document.open('x', encoding='utf-8', newline='\n') as stream: stream.write(text)
    atomic_json(out/'REPORT.json', {'status': 'PASS', 'created_utc': utc_now(),
        'document': document.relative_to(ROOT).as_posix(), 'document_sha256': digest(document),
        'algebra_result_sha256': digest(out/'RESULTS.json')}, immutable=True)
    progress('Fixed-horizon alarm interpretation verified',
        'Exact rank enumeration and horizon-extension counterexample; original counts only; no new alarm policy, model fit or quality score')
    print('NEXT3 HORIZON CALIBRATION LOGIC PASS', flush=True)


if __name__ == '__main__': main()

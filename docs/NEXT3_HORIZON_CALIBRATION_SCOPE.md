# NEXT3 고정 horizon 경보 보정의 범위

기존 0.05 임계값은 1,024점 구간의 최대 점수를 보정한다. 이 보정을 16,384점 또는 무한 관측 구간의 5% 경보 보장으로 해석하지 않는다. 아래는 기존 알고리즘의 수학적 범위를 점검한 문서이며 새 경보 정책이나 모델 결과가 아니다.

## 기존 코드와 rank 계산

`COUNTEREXAMPLE_SUITE_LOCK.json`은 배경별 calibration series 1,000개, calibration horizon 1,024, 평가 horizon 1,024/4,096/16,384를 고정했다. `src/next2/synthetic_analysis.py::threshold`는 각 calibration series의 최대 점수 M_i를 정렬해 `k = ceil((n+1)(1-alpha))`번째 값을 사용한다. 경보 비교는 strict `score > threshold`다. k가 n보다 크면 `None`으로 경보를 내지 않으며, 이는 임계값을 무한대로 두는 것과 같다.

고정 점수 함수에 대해 calibration 최대값들과 새 평가 최대값이 교환가능하면, 새 값의 순위 논리로 `P(M_new > M_(k)) <= (n+1-k)/(n+1) <= alpha`가 성립한다. 이 확률은 calibration 표본과 새 평가 사례의 무작위성을 함께 평균한 marginal 값이다. 특정 calibration 집합, 특정 H 또는 선택한 모든 모델에 각각 같은 보장을 주는 조건부 명제가 아니다. 동점은 strict 비교에서 보수적으로 처리된다. 일반적인 split conformal rank 논리와 marginal 해석은 [Angelopoulos·Bates, v6, §1 및 부록 D](https://arxiv.org/html/2107.07511v6)를 따른다.

기존 threshold 구현을 고정한 채 서로 다른 점수 1,001개의 test 순위를 모두 한 번씩 제외해 직접 대조했다. alpha=0.05에서는 k=951, 경보 순위 50/1,001 = **0.049950049950**이고 alpha=0.01에서는 k=991, 10/1,001 = **0.009990009990**다. 동점 예와 표본 수 9의 `None` 경계도 검증했다. 새 bootstrap이나 Monte Carlo draw는 0개이며 실제 모델의 예측 배열을 쓰지 않았다.

## 긴 관측 구간으로 자동 확장되지 않는 이유

M_T를 첫 T개 점수의 최대값이라 하면 T가 커질수록 M_T는 감소하지 않는다. calibration은 M_1024의 분포를 기준으로 하지만, M_16384의 분포가 같다는 가정은 없다. 프로젝트 자체의 논리 대조로 첫 1,024점은 위 rank 점수를 유지하고 그 이후 점수만 1이 되는 가상의 null 과정을 만들었다. 첫 horizon에서는 50/1,001의 경보 순위가 나오지만 긴 horizon에서는 1,001/1,001이 경보를 낸다. 이는 alpha에서 1로 증가할 수 있다는 가능성의 반례이며, 실제 frozen 모델의 예측이나 수치 전망이 아니다.

서로 다른 16개 block 각각에 동일한 marginal 경보 상한을 별도로 입증했다면 union bound를 적용할 수 있다. block 사건의 독립성까지 가정하면 `1-(1-p)^16`을 계산할 수 있다. 현재 detector는 recurrent state를 이어가고 score의 나이 효과도 있으므로, block별 동일 분포·상한·독립성의 전제를 확인하지 않았다. 이 두 공식을 현재 모델의 실제 장기 FPR 예측으로 사용하지 않는다.

## 원래 관측의 해석

기존 BL80의 N0 평가 512개에서 동일 임계값 **0.7902215719223022**를 사용한 원래 기록은 아래와 같다. 새 평가를 수행하지 않고 보존된 count/rate의 나눗셈만 확인했다.

| horizon | 원래 경보 수 / series | 원래 eventual alarm rate |
|---:|---:|---:|
| 1024 | 21 / 512 | 0.041015625 |
| 4096 | 69 / 512 | 0.134765625 |
| 16384 | 169 / 512 | 0.330078125 |

긴 구간의 경보율 상승은 미해결 운영 약점이다. 동시에 1,024점 horizon의 marginal calibration 명제 자체를 이 긴 구간의 숫자만으로 반박했다고 쓰지 않는다. 고정 학습 모델, 일치하는 synthetic 배경 및 whole-series 교환가능성의 가정과 반복적인 진단·모델 선택의 영향을 따로 검토해야 한다. 경쟁 점수 TS-AUC의 개선 여부와도 별개다.

무한 또는 임의 중단 시각까지 유효한 경보를 주장하려면 추가 수학적 구조와 가정이 필요하다. [Howard 등, 비음수 supermartingale에 의한 time-uniform bound, v8](https://arxiv.org/html/1808.03204v8)는 그런 구조를 다루지만, 현재 학습된 확률 점수·Bayes 이름의 feature·plug-in H 보정이 그 조건을 만족한다고 입증한 것은 아니다. 이 문서에서 e-process나 새 threshold schedule을 구현·채택하지 않았다.

후속 B6 검증은 비교 horizon, calibration/evaluation 분리, late change의 clean detection, premature alarm, delay와 TS-AUC를 함께 사전 고정해야 한다. FPR만 낮추는 score 압축을 실질 개선으로 인정하지 않는다. 재현 근거는 `artifacts/next3/maintenance/horizon_calibration_logic_v1/RESULTS.json`이다.

# NEXT3 W3 손실과 순위 목표 — 수식 표기 수정본

**이 문서를 현재 해석 문서로 사용한다.** 첫 생성본의 수식 escape가 표시 과정에서 손상되어 일반 텍스트 수식으로 고쳤다. 원본 문서·계산 receipt·생성 source는 보존했다. 두 점 계산 결과, 기존 D004 결과, 모델 선택은 바꾸지 않았다.

D004에서 W3 logloss가 낮아졌는데 TS-AUC는 악화한 실제 후보들이 있었다. 이는 이번 유한 후보·학습 절차에서 두 목적의 순서가 달랐다는 근거다. Logistic loss 자체가 일반적으로 AUC에 부적합하다는 증거는 아니다. 새 competition 데이터나 OOF를 읽지 않았고 모델을 학습하지 않았다.

## W3는 쌍의 두 endpoint 손실 평균으로 쓸 수 있다

시점 t의 양성·음성 집합을 P_t, N_t, 크기를 n+(t), n−(t)라 하자. W = Σ_t n+(t)n−(t) > 0일 때, raw W3 weight는 양성에 n−(t), 음성에 n+(t)를 준다.

```text
L_W3 = Σ_t [ n−(t) Σ_(i in P_t) −log(p_i)
           + n+(t) Σ_(j in N_t) −log(1−p_j) ] / (2W)

     = Σ_t Σ_(i in P_t, j in N_t)
           [ −log(p_i) − log(1−p_j) ] / (2W)
```

분모의 2는 쌍마다 endpoint가 두 개이기 때문에 필요하다. TS-AUC는 같은 eligible pair에 순위 credit을 주고 W로 나눈다. 따라서 비교 대상 쌍은 같지만 쌍에 매기는 값이 다르다. Single-class 시점의 raw W3 weight와 쌍 수는 모두 0이다. W=0이면 위 손실을 정의할 수 없으며 공식 scorer의 0.5 fallback과 별도로 처리해야 한다.

## 고정한 두 점 예시

이 표는 양성 한 점·음성 한 점의 사전 고정 예시다. 학습 모델이나 DEV 성능 결과가 아니다.

| 예시 | 양성 점수 | 음성 점수 | Pair AUC | W3 point logloss | Pairwise logistic |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.02 | 0.01 | 1.0 | 1.961036670641 | 0.402092423630 |
| 2 | 0.49 | 0.51 | 0.0 | 0.713349887877 | 0.733952515216 |

첫 예시는 순위가 맞지만 양성 확률이 매우 작다. 둘째는 순위를 뒤집지만 point logloss가 낮다. 제한된 후보 집합에서는 더 작은 point loss가 더 좋은 AUC를 보장하지 않는다. 예시의 극단적 차이를 실제 DEV 개선량으로 해석하지 않는다.

실수 logit f_i, f_j에 대해 다음 부등식이 성립한다.

```text
softplus(−f_i) + softplus(f_j) >= softplus(f_j−f_i)

(1+exp(−f_i))(1+exp(f_j))
  = 1 + exp(−f_i) + exp(f_j) + exp(f_j−f_i)
  >= 1 + exp(f_j−f_i)
```

Pairwise logistic은 같은 시점의 모든 logit에 동일한 상수를 더해도 변하지 않는다. Endpoint 손실은 변할 수 있다. 고정 예시에서 상수 5를 더한 항등식을 확인했다. 이 실수 연산 성질은 float32 rounding, clipping, saturation으로 새 동점이 생기는 경우의 완전한 rank 보존을 보장하지 않는다.

## 모집단 일관성과 유한 실험은 구별한다

Strongly proper loss로 bipartite ranking regret를 제한하는 결과가 있으며, 충분히 풍부한 함수 공간과 적절한 정규화 등의 조건에서 non-pairwise logistic 계열도 ranking consistency를 갖는다. 따라서 pairwise loss로 바꾸기만 하면 자동 개선된다고 말할 수 없다. [Agarwal, JMLR 2014](https://jmlr.org/papers/v15/agarwal14b.html)

다음은 이 프로젝트의 조건부 해석을 위한 별도 계산이다. 같은 시점 살아 있는 시계열의 양성 확률을 eta = P(Y=1 | X, t, alive), 양성·음성 가중치를 각각 a, b > 0이라 하면:

```text
weighted point loss optimum: p* = a·eta / [a·eta + b·(1−eta)]

d(p*) / d(eta) = a·b / [a·eta + b·(1−eta)]² > 0
```

최적값은 같은 시점의 eta에 대해 엄격히 증가한다. 하지만 실제 유한 gate 함수 공간, regularization, optimization, 시점 간 공유 파라미터와 재사용된 DEV가 이 최적값을 달성했다는 증거는 없다. D004는 이 유한 조건에서 읽어야 한다.

## 쌍 수가 독립 표본 수는 아니다

한 시계열은 여러 시점과 여러 쌍에 재등장한다. Ranking empirical risk의 U-statistic 구조를 다루는 연구는 독립적인 pair를 가정한 단순 표준오차와 다른 분석을 사용한다. [Clémençon, Lugosi, Vayatis](https://arxiv.org/abs/math/0603123)

여기서는 관측 길이와 label 구조에 따라 TS-AUC 분모도 달라진다. 위 결과를 그대로 적용해 competition 일반화 보장이나 유효 표본 수를 선언하지 않는다. D007의 whole-series multiplicity 이차형식은 기존 bootstrap 계산의 정확성 검증이며, 새로운 독립 데이터 검증이 아니다.

Incomplete U-statistic으로 pairwise 목적의 계산량을 줄이는 접근도 연구돼 있다. 다음 세션에서 검토할 계산 방법이지만 현재 모델의 개선 근거는 아니다. [Clémençon, Bellet, Colin, JMLR 2016](https://arxiv.org/abs/1501.02629)

## 다음 연구와 현재 결정

같은 데이터의 결과를 보고 후속 분석을 선택하는 적응성은 고정된 검정 목록의 다중 비교와 구별해야 한다. Reusable holdout 연구의 보호 절차를 이 프로젝트에 적용하지 않았으므로 그 보장을 현재 bootstrap에 붙이지 않는다. [공저자 Moritz Hardt의 연구 설명](https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/)

다음 세션의 pairwise 검토도 outer validation을 보지 않는 inner-OOF 학습, 고정된 같은-시점 pair 생성, 시계열 단위 검증과 기존 효과 크기·mechanism·robustness gate를 필요로 한다. 목적함수 변경은 새로운 관측 정보 도입과 다르므로 그것만으로 generation 교체를 주장할 수 없다. 이번 세션에서는 pairwise router를 학습하거나 scientific menu를 다시 열지 않는다.

원본 경험적 근거는 `docs/NEXT3_ROUTER_METRIC_ALIGNMENT_FINDINGS.md`와 `docs/NEXT3_PAIR_MATRIX_BOOTSTRAP_AUDIT_FINDINGS.md`다. 표의 정확한 수치와 두 점 계산은 `artifacts/next3/maintenance/methodology_review/ALGEBRA_RESULTS.json`에 보존했다.

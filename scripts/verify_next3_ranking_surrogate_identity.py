"""Fixed explanatory algebra only: no competition rows, model fits or new scores."""
from pathlib import Path
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def softplus(value): return max(value, 0.)+math.log1p(math.exp(-abs(value)))


def logit(probability): return math.log(probability)-math.log1p(-probability)


def pair_credit(a, b): return float(a > b)+.5*float(a == b)


def main():
    out = BASE/'maintenance/methodology_review'; policy = read_json(out/'POLICY.json')
    assert policy['status'] == 'REGISTERED_EXPLANATORY_MAINTENANCE'
    assert policy['fixed_toy_positive_negative_probabilities'] == [[.02, .01], [.49, .51]]
    assert policy['fixed_common_logit_shift'] == 5.
    rows = []
    for p_positive, p_negative in policy['fixed_toy_positive_negative_probabilities']:
        a, b = logit(p_positive), logit(p_negative)
        weighted_logloss = (-math.log(p_positive)-math.log1p(-p_negative))/2
        pair_endpoint_loss = (softplus(-a)+softplus(b))/2
        pairwise_logistic = softplus(b-a)
        shifted = softplus((b+5)-(a+5))
        assert abs(weighted_logloss-pair_endpoint_loss) <= 1e-14
        assert pairwise_logistic <= 2*weighted_logloss+1e-14
        assert abs(shifted-pairwise_logistic) <= 1e-14
        rows.append({'positive_probability': p_positive, 'negative_probability': p_negative,
            'literal_pair_AUC': pair_credit(p_positive, p_negative), 'balanced_point_logloss': weighted_logloss,
            'pair_endpoint_logloss': pair_endpoint_loss, 'pairwise_logistic_loss': pairwise_logistic,
            'common_logit_shift': 5., 'pairwise_logistic_after_shift': shifted,
            'balanced_point_logloss_after_shift': (softplus(-(a+5))+softplus(b+5))/2,
            'weighted_point_vs_pair_endpoint_abs_difference': abs(weighted_logloss-pair_endpoint_loss)})
    assert rows[0]['literal_pair_AUC'] == 1 and rows[1]['literal_pair_AUC'] == 0
    assert rows[1]['balanced_point_logloss'] < rows[0]['balanced_point_logloss']
    assert rows[1]['pairwise_logistic_loss'] > rows[0]['pairwise_logistic_loss']
    atomic_json(out/'ALGEBRA_RESULTS.json', {'status': 'PASS_EXPLANATORY_ALGEBRA_ONLY', 'created_utc': utc_now(),
        'policy_sha256': digest(out/'POLICY.json'), 'script_sha256': digest(Path(__file__)), 'fixed_examples': rows,
        'competition_data_accesses': 0, 'new_fits': 0, 'new_scientific_candidates': 0, 'new_DEV_quality_scores': 0,
        'scientific_promotion_authority': False,
        'scope': 'Two prospectively fixed handworked point pairs illustrate empirical loss ordering. The symbolic weighted-pair identity and monotonic conditional Bayes optimum are explained in the report; this fixture is not a statistical model comparison.'}, immutable=True)
    table = '\n'.join(f"| {i+1} | {row['positive_probability']} | {row['negative_probability']} | {row['literal_pair_AUC']} | {row['balanced_point_logloss']:.12f} | {row['pairwise_logistic_loss']:.12f} |"
        for i, row in enumerate(rows))
    report = r'''# NEXT3 W3 손실·순위 목표 해석 보완

D004에서 W3 logloss가 낮아졌는데 TS-AUC는 악화한 실제 후보들이 관측됐다. 이것은 이번 유한 후보·학습 절차의 대리 손실 순서가 평가 지표 순서와 달랐다는 근거다. **logistic loss 자체가 일반적으로 AUC와 불일치한다는 증거는 아니다.** 기존 후보의 수치와 탈락 결정은 바꾸지 않는다.

이 문서는 공개 1차 문헌과 별도로 고정한 두 점 예시를 사용한다. competition 관측값·label·OOF를 추가로 읽거나 모델을 학습하지 않았다. 새 DEV 결과나 과학 후보를 만든 작업이 아니다. 사전 등록·계산 결과는 `artifacts/next3/maintenance/methodology_review/`에 있다.

## W3의 pair-endpoint 항등식

시점 (t)의 양성 집합을 (P_t), 음성 집합을 (N_t), 크기를 (n_+(t),n_-(t))로 둔다. (W=sum_t n_+(t)n_-(t)>0)라 하자. 점 손실 (ell_i=-y_ilog p_i-(1-y_i)log(1-p_i))에 양성은 (n_-(t)), 음성은 (n_+(t))의 raw W3 weight를 준다.

[
L_{mathrm{W3}}=rac{sum_t[n_-(t)sum_{iin P_t}-log p_i+n_+(t)sum_{jin N_t}-log(1-p_j)]}{2W}
=rac{1}{2W}sum_tsum_{iin P_t,jin N_t}[-log p_i-log(1-p_j)].
]

분모의 **2**가 필요하다. 쌍 하나가 두 endpoint 손실을 제공하기 때문이다. 반면 TS-AUC의 분자는 같은 쌍의 순위 credit이고 분모는 (W)다. 두 목적은 같은 eligible pair를 참조하지만 쌍에 매기는 값이 다르다. single-class 시점은 raw W3 weight와 eligible pair가 모두 0이다. 모든 시점에 eligible pair가 없으면 공식 TS-AUC의 0.5 fallback과 W3의 정규화 문제를 별도로 다뤄야 한다. 이 식으로 0/0 손실을 정의하지 않는다.

## 두 점으로 보는 유한 후보 순서의 차이

아래는 학습한 후보가 아니라 사전에 고정한 양성 한 점·음성 한 점이다.

| 예시 | 양성 점수 | 음성 점수 | pair AUC | W3 point logloss | pairwise logistic |
| --- | ---: | ---: | ---: | ---: | ---: |
''' + table + r'''

첫 예시는 순위가 맞지만 양성 확률이 작다. 둘째 예시는 순위를 뒤집지만 point logloss가 낮다. 따라서 제한된 유한 후보 중 point logloss가 더 작은 후보가 반드시 AUC도 높은 것은 아니다. 이 극단적인 예시의 크기는 실제 DEV 개선량의 근거가 아니다.

실수 logit (f_i,f_j)에 대해 endpoint 합은
[
operatorname{softplus}(-f_i)+operatorname{softplus}(f_j)geoperatorname{softplus}(f_j-f_i).
]

지수 곱을 전개하면 왼쪽 로그의 인자가 (1+exp(-f_i)+exp(f_j)+exp(f_j-f_i))여서 부등식이 성립한다. pairwise logistic은 같은 시점 모든 logit에 같은 상수를 더하면 유지되지만 endpoint 손실은 변할 수 있다. 이 실수 연산 성질은 float32 rounding, clipping, saturation이 새로운 동점을 만들 때의 완전 동일성을 보장하지 않는다.

## 모집단 일관성과 이번 실험의 구분

strongly proper loss를 통한 bipartite ranking regret bound와, 충분히 풍부한 함수 공간·적절한 정규화 조건에서 non-pairwise logistic 계열의 ranking consistency가 알려져 있다. 따라서 손실을 pairwise로 바꾸는 것 자체가 자동 개선이라는 주장은 성립하지 않는다. [Agarwal, JMLR 2014](https://jmlr.org/papers/v15/agarwal14b.html)

이 프로젝트에 대한 다음 식은 위 논문에서 그대로 인용한 것이 아니라 별도의 단순한 조건부 계산이다. 같은 시점 살아 있는 시계열에서 (eta(x,t)=P(Y=1mid X=x,t,mathrm{alive}))이고 양성·음성 class weight가 각각 (a_t,b_t>0)라면 가중 point logloss의 조건부 최적값은
[
p^*(x,t)=rac{a_teta(x,t)}{a_teta(x,t)+b_t(1-eta(x,t))}.
]

(eta)에 대한 미분은 (a_tb_t/[a_teta+b_t(1-eta)]^2>0)다. 시점 안에서 Bayes 순위는 보존된다. 그러나 실제 제한된 gate, 유한 데이터, regularization, optimization, 시점 간 공유 파라미터, 재사용된 DEV는 이 최적값을 달성한다는 조건을 충족했다는 증거가 아니다. D004의 결과는 이 유한 조건에서 읽어야 한다.

## 쌍 수를 독립 표본 수로 세지 않는다

같은 시계열은 여러 시점과 여러 쌍에 등장한다. Ranking empirical risk의 U-statistic 구조를 다루는 문헌은 독립적인 쌍을 가정한 단순 표준오차와 다른 분석을 사용한다. [Clémençon, Lugosi, Vayatis](https://arxiv.org/abs/math/0603123)

이 프로젝트의 TS-AUC는 관측 길이와 label 구조에 따른 eligible-pair 분모도 포함한다. 위 고전 결과를 그대로 적용해 competition의 일반화 보장이나 정확한 유효 표본 수를 선언하지 않는다. D007이 재구성한 whole-series multiplicity의 이차형식은 기존 bootstrap 계산의 정확성을 확인했으며, 그것만으로 독립 검증이 되지는 않는다.

Incomplete U-statistic을 사용해 pairwise 목적의 계산량을 줄이는 방법도 연구돼 있다. 이는 다음 세션에서 검토할 계산 방법일 뿐, 이번 frozen 후보가 개선된다는 실증 근거는 아니다. [Clémençon, Bellet, Colin, JMLR 2016](https://arxiv.org/abs/1501.02629)

## 다음 연구의 한계와 현재 결정

같은 데이터의 이전 결과를 보고 다음 분석을 고르는 적응성은 고정된 검정 목록의 다중 비교와도 구별해야 한다. reusable holdout 연구가 다루는 보호 장치를 이 프로젝트에 적용한 것은 아니므로, 그 연구의 보장을 현재 bootstrap에 붙이지 않는다. [공저자 Moritz Hardt의 연구 설명](https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/)

다음 세션에서 pairwise 목적을 검토한다면 outer validation을 보지 않는 inner-OOF 학습, 고정된 같은-시점 pair 생성, 동일한 시계열 단위 검증, 기존 효과 크기·mechanism·robustness gate가 필요하다. 목적함수 변경은 새 관측 정보의 도입과도 다르므로 그것만으로 generation 교체를 주장할 수 없다. **이번 세션에서는 pairwise router를 학습하거나 scientific menu를 다시 열지 않는다.**

현재 선택은 그대로 두 역할이다. D004의 손실 불일치와 D002/D007의 조건부 불확실성은 해석 범위 안에서 유지한다. 원본 경험적 근거: `docs/NEXT3_ROUTER_METRIC_ALIGNMENT_FINDINGS.md`, `docs/NEXT3_PAIR_MATRIX_BOOTSTRAP_AUDIT_FINDINGS.md`.
'''
    destination = ROOT/'docs/NEXT3_RANKING_SURROGATE_METHODOLOGY.md'
    with destination.open('x', encoding='utf-8', newline='\n') as stream: stream.write(report)
    atomic_json(out/'REPORT_RECEIPT.json', {'status': 'PASS_EXPLANATORY_REPORT', 'created_utc': utc_now(),
        'report': destination.relative_to(ROOT).as_posix(), 'report_sha256': digest(destination),
        'algebra_receipt_sha256': digest(out/'ALGEBRA_RESULTS.json'), 'new_fits': 0, 'new_DEV_results': 0,
        'scientific_decisions_changed': False,
        'primary_sources': ['https://jmlr.org/papers/v15/agarwal14b.html', 'https://arxiv.org/abs/math/0603123',
            'https://arxiv.org/abs/1501.02629', 'https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/']}, immutable=True)
    progress('Ranking surrogate interpretation clarified', 'Fixed algebra illustrates finite candidate mismatch; no claim that logistic loss is generally ranking-inconsistent, and no new empirical candidate')
    print('NEXT3 RANKING SURROGATE EXPLANATION PASS TWO FIXED EXAMPLES; ZERO DEV RESULTS', flush=True)


if __name__ == '__main__': main()

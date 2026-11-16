# NEXT3 dependence specialist forensic

Scientific-menu review; engineering work and final audit remain in progress.

고정 BL80은 J102 80%와 C205 20%를 결합한다. C205는 lag 1/2/4/8의
중심화된 곱마다 EWMA8/32, 양/음 CUSUM, Bayes 0.25/1을 사용한다.
단독 평균 점수 약 0.5755는 J102보다 낮지만, 같은 시점의 보완 정보로
기존 결합 모델의 공식 TS-AUC를 높인다.

정확한 pair 질량 분해에서 DEPENDENCE_DOMINANT의 기여는 +0.001854,
SCALE_DOMINANT의 손실 기여는 약 -0.000175다. 따라서 SCALE 손실만
회복해 +0.0015 전체 성능 승격 기준을 넘는다는 가설은 근거가 약하다.
각 집단 점수 차이는 서로 다른 pair 질량을 가지므로 단순 합산하지 않는다.

| 고정 변경 | 전체 평균 차이: BL80 대비 | 해석 |
| --- | ---: | --- |
| P1 lag 1/2만 유지 | -0.000110798 | SCALE도 -0.001214; 단순 축소를 승격할 근거 부족 |
| P2 lag 4/8만 유지 | -0.001153921 | full 후보 진입 실패 |
| P3 local EWMA/CUSUM만 유지 | -0.000150128 | SCALE -0.002023, 의존성 이득도 약화 |
| P4 Bayes만 유지 | -0.000268681 | SCALE +0.000564, CI가 0 포함 |
| I2 H 고정 midrank 곱 | +0.000569907 | 가장 좋은 새 평균이나 +0.0015 효과 기준 미달 |
| O1 nested 선형 residual | -0.000130978 | 독립적인 양의 보완 정보 입증 실패 |

I1의 sign 곱은 동일한 기존 실패 실험을 source/config/fold hash로 확인해
재사용했다. 새 fit은 없다. I2는 부호 이진화와 달리 H의 순위 정보를 보존하지만
원래 H-only AR/ARCH 전처리까지 임의 단조변환 불변이라고 주장하지 않는다.
O1은 각 outer train 안에서 생성한 base OOF로만 residual 계수를 학습했다.
outer validation을 포함한 일반 OOF를 같은 행에 fit/score하지 않았다.

I2의 개별 조건부 CI는 [+0.000093, +0.001089]지만 SCALE 차이는
약 +0.000006으로 명확한 감소 문제 해결이 없다. 완료된 12개 full 후보의
공동 조건부 envelope에서는 I2 하한도 -0.000161이다. 이는 반복 선택 문제를
모두 보정한 신뢰구간이 아니며, 기존 CI를 독립 검증처럼 해석하면 안 된다.

현재 증거는 C205의 정확한 최소 필수 집합을 확정하지 못한다. 평가한 모든
단순 subset이 전체 평균을 낮췄다는 범위까지 결론을 제한한다. 고정 80/20
참조를 유지하며, 미세 weight 탐색이나 새로운 CV seed는 사용하지 않는다.

추가 D004 진단에서 R2와 R3는 기존 outer validation의 W3 확률 손실을
각각 0.000768, 0.000547 줄였지만 TS-AUC는 악화했다. R2의 관측된 모든
C205 가중치는 0.2보다 작다. R3는 fold 0–3에서 전부 0.1이며 fold 4에서만
0.1/0.2를 사용한다. 이는 확률 손실과 순위 목표의 불일치에 대한 단서이며,
새 목적함수의 성공을 증명하지 않는다. 세부 결과는
`docs/NEXT3_ROUTER_METRIC_ALIGNMENT_FINDINGS.md`에 남겼다.

원본 근거: `artifacts/next3/pair_contributions`, `experiments/P*`,
`forensic/I2_HISTORICAL_RANK_PRODUCTS`, `experiments/O1_LINEAR_RESIDUAL`,
`reuse/I1_SIGN_PRODUCTS.json`, `selection_uncertainty`.

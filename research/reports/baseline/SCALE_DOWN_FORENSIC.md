# NEXT3 scale-decrease forensic

Scientific-menu review; engineering work and final audit remain in progress.

현재의 문제는 감소 신호가 전혀 없다는 것이 아니다. 고정 합성 조건에서
현재 raw/forecast 값과 음의 방향 Bayes 요약에는 감소 구간을 구분하는 정보가
남지만, 최종 트리 점수의 구분력은 약하다. 이 결과는 학습된 트리의 단일
원인을 증명하거나 실제 DEV 성능을 보장하지 않는다.

1,312개 집계 행의 단계 진단은 기존 N0/N1/N2/N3와 분산 배율 0.7/0.5,
512개 고정 쌍, 네 관측 창을 그대로 사용했다. 4,096개 prefix 검증에서
인과적 단계 계산이 일치했다. 음/양 방향 Bayes를 log-add로 합친 값은
기존 양방향 값과 2e-6 이내였다. 이 창별 구분 통계는 공식 TS-AUC가 아니다.

실제 DEV의 SCALE_DOMINANT 132개 변화 시계열을 기존 관측 창의 log SD
차이로 나누면 감소 79개, 증가 53개다. 두 하위 집단은 각각 100개 지원 기준에
못 미친다. 구분에는 평가용 DEV 관측만 사용했으며 모델 입력이나 학습 타깃에
넣지 않았다. 이 구분은 잠재 혁신 분산 배율을 직접 관측한 것이 아니다.

| 변경 | 실제 DEV 결과 | 판단 |
| --- | --- | --- |
| R1 점수 차이 라우팅 | 전체 +0.000109; SCALE +0.003794, 양의 조건부 CI | 부분 개선. ARCH V3/V4 증가 폭이 각각 0.01에 못 미쳐 승격 불가 |
| R3 깊이 2 라우팅 | 전체 -0.000630; SCALE +0.002856; DEPENDENCE -0.001966 | 의존성 손실. 최종 gate의 모든 leaf가 C205 비중 0.1 |
| S3 음의 방향 Bayes 4개 추가 | 전체 +0.000023; SCALE +0.001449, CI가 0 포함 | 지원된 감소 문제 해결로 인정할 수 없음 |
| S1 H 하방/상방 RMS 보정 8개 추가 | screen +0.002610; full -0.000454, median -0.002663 | 고정 full 후보 진입 기준 실패 |
| S2 H 하위 사분위 점유율 8개 추가 | screen -0.003843 | screen에서 중단, full fit 없음 |

S1은 원래 J102 77개 입력을 보존하고 두 채널의 보정된 current, EWMA8,
EWMA32, Bayes1을 추가했다. H의 변환 후 평균/표준편차를 다시 보정했으며
관측 suffix나 변화 시점은 쓰지 않았다. S2와 S3도 사전 등록된 작은 메뉴만
평가했다. S4는 scale 가설 계열 예산 때문에 보류했으며 정확한 재사용 실험으로
허위 집계하지 않는다.

평가용 감소 하위 집단에서 R1의 전체 BL80 대비 차이는 +0.004847이고
조건부 95% CI는 [+0.002595, +0.007332]다. 그러나 감소 79개라는 지원 부족,
고정 ARCH gate 실패, 반복 DEV 선택의 한계가 남는다. 이 수치로 seal을
열거나 라우터를 배포할 근거는 충분하지 않다.

원본 근거: `artifacts/next3/scale_stages`, `directional_scale`,
`experiments/S1_LOWER_SEMISCALE`, `experiments/S2_LOWER_QUANTILE`,
`forensic/S3_NEGATIVE_BAYES`, `counterexamples/R3_DEPTH2_GATE/arch_gate`.
모든 CI는 고정된 학습 결과를 조건으로 한 시계열 단위 paired bootstrap이며
후보 선택 불확실성을 보정하지 않는다.

D003 유지보수 검증에서 S1과 S3의 기존 전체 DEV feature cache, S2의 기존
screen cache를 실제 한 점씩 갱신하는 상태와 전수 비교해 비트 단위 일치를
확인했다. 이미 저장된 held-out component prediction도 일치했다. 이 검증은
새 fit이나 점수를 생성하지 않았고, 기존 실패 결과를 승격 판단으로 바꾸지 않는다.

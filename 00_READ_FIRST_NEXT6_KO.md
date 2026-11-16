# NEXT6 연구 결과와 읽는 순서

새 모델은 사전 승격 기준을 통과하지 못했습니다. BL80와 D2의 기존 역할을 유지합니다.

이 문서는 최초 NEXT6 실험을 동결한 시점의 보고서입니다. 사용자의 추가 7시간 이상 연구 요청에 따른 병목 개선 실험은 별도 `artifacts/next6_extension`에서 진행하며, 이 보고서의 외부 평가 결과와 모델은 보존합니다.

실제 계산 결과로 만든 [동적 상태전이망](artifacts/next6/FIGURES/02_DYNAMIC_STATE_TRANSITION_NETWORK_fold0_outer_full.png), [예측적 선후관계망](artifacts/next6/FIGURES/01_PREDICTIVE_FEATURE_NETWORK_fold0_outer_full.png), [전체 그림 갤러리](artifacts/next6/FIGURES/index.html)를 먼저 볼 수 있습니다. 상태전이망의 숫자는 학습된 조건부 피처 상태 확률이며, 생성형 이미지나 경제적 인과관계 그림이 아닙니다.

## 같은 평가 범위의 결과

| 방법 | 외부 fold | 평균 TS-AUC | Δ BL80 | Δ 입력 완비 M1_STRICT |
|---|---:|---:|---:|---:|
| M0 | 5 | 0.620365 | +0.000000 | +0.001078 |
| AMPLITUDE_0.9_M0 | 5 | 0.620365 | +0.000000 | +0.001078 |
| BASE_ONLY | 5 | 0.620576 | +0.000211 | +0.001289 |
| M1 | 5 | 0.620007 | -0.000358 | +0.000720 |
| M2 | 5 | 0.619397 | -0.000968 | +0.000110 |
| M3 | 5 | 0.620834 | +0.000469 | +0.001548 |
| OWN | 5 | 0.620559 | +0.000193 | +0.001272 |
| RANDOM | 5 | 0.618606 | -0.001760 | -0.000681 |
| M4 | 1 | 0.610202 | -0.002807 | +0.001947 |
| M1_STRICT | 5 | 0.619287 | -0.001078 | +0.000000 |

M0는 기존 NEXT5 ResearchPool 6,400개 series·원래 grouped 5-fold의 BL80입니다. 과거 전체 DEV 8,000개에서 보고한 0.623268 수치와 직접 비교하지 않습니다. 원래 M1은 16개 연속형 노드·lag 1/2/4/8/16·delta·EWMA 8/32와 기존 J102/C205 피처를 쓰는 245열 대조였습니다. 감사에서 명시적 상태·경과 시점 입력이 빠진 것을 발견해, 외부 점수를 열기 전에 M1_STRICT를 고정했습니다. M1_STRICT는 상태 코드와 5개 lag, 상태 EWMA 8/32, 노드별 own-state 지속 길이와 현재 시점을 포함하는 390열 비그래프 대조입니다. 같은 고정 learner로 5회만 학습했으며 튜닝·그래프 재학습은 하지 않았습니다. 학습 recipe는 대조군 포함 총 8개입니다.

M2는 관계 요약, M3는 전이 요약을 추가합니다. 이득 판단에는 입력이 완비된 M1_STRICT를 사용하고, 원래 M1에 대한 gate도 유지해 기준을 더 엄격하게 했습니다. 원래 연속형 M1 결과는 MODEL_COMPARISON_CONTINUOUS_M1.csv에 보존했습니다. M4는 각 fold 내부에서 M2/M3 모두보다 +0.0002 이상 좋아야 외부 학습을 허용했습니다. 일부 fold만 있는 M4 평균은 전체 5-fold 후보 성적이 아닙니다.

- M2 대 BL80 평균 Δ의 95% paired CI: [-0.005993, +0.003804].
- M3 대 BL80 평균 Δ의 95% paired CI: [-0.004741, +0.005310].
- M2 대 입력 완비 M1_STRICT 평균 Δ의 95% CI: [-0.001562, +0.001805].
- M3 대 입력 완비 M1_STRICT 평균 Δ의 95% CI: [-0.000304, +0.003390].

1,000회 whole-series paired bootstrap은 고정된 외부 예측에 조건부입니다. 과거 ResearchPool 반복 사용, 이번 가설 설계 및 그래프 선택의 모든 불확실성을 제거하지 않습니다. 숨긴 데이터 인증 결과로 해석할 수 없습니다.

## 관계와 전이에서 확인한 것

공식 tau_index에 대해 target=1[t>=tau]를 적용했습니다. 변화가 없는 series는 전부 0입니다. pre=-32..-1, onset=0..7, earlyPost=8..31, establishedPost=32..127, laterPost=128+를 점수 확인 전에 고정했습니다. no-break, 같은 절대 시점의 not-yet-break, 학습 H 특성으로 매칭한 대조, 정답·최종 길이와 독립인 pseudo-event를 사용했고 없는 구간은 채우지 않았습니다.

각 series 내부 순위 상관을 먼저 계산해 평균했습니다. 부분 상관에서는 창 안의 선형 추세와 공통 크기를 제거했습니다. lag 검증은 자기 자신의 5개 과거값, 공통 에너지·조건부 분산 과거값, 현재까지의 경과 시점을 조건으로 둔 예측 오차 비교입니다. 상관 변화만으로 추가 예측 이득이 있다고 판단하지 않았습니다.

특히 창 길이 효과가 컸습니다. node 09 원 에너지 상방과 node 06 에너지 장기 지속의 부분 순위 상관은 정상 전체 길이에서 -0.009, 실제 onset 8시점 창에서 0.482였습니다. 그러나 변화 없는 series의 가상 onset 창에서도 0.498였고, 같은 시점·같은 길이의 H 매칭 not-yet 대조와의 onset 차이는 -0.002로 작아졌습니다. 이는 학습 구간에서 고른 설명용 사례이며, 전체 길이 정상 지도와 짧은 onset 지도의 큰 차이를 변화의 증거로 해석하면 안 됨을 보여줍니다. [창 길이 대조 그림](artifacts/next6/FIGURES/08_WINDOW_LENGTH_CONFOUND.png)을 함께 확인하세요. 기존 04_MAP_C 그림은 길이가 다른 창의 탐색적 차이이며, 같은 창의 관계 변화 판단에는 07_ALIGNMENT의 matched_not_yet_delta 지도를 사용합니다.

5개 외부 학습 그래프에서 선택된 서로 다른 edge는 18개, 5 fold 모두 선택된 edge는 2개입니다. 선택된 fold-edge 45개 중 선택 phase에서 기술적 held-out 재현 기준을 만족한 경우는 25개입니다. 같은 edge가 여러 fold에서 중복 집계될 수 있습니다.

- fold 4, node 11 → 7, lag 1: held-out 상대 MSE 개선 +0.1039, 95% CI [+0.0890, +0.1180], 1277 series. MECHANICAL_DERIVED.
- fold 0, node 11 → 7, lag 1: held-out 상대 MSE 개선 +0.0987, 95% CI [+0.0834, +0.1133], 1276 series. MECHANICAL_DERIVED.
- fold 4, node 9 → 7, lag 1: held-out 상대 MSE 개선 +0.0983, 95% CI [+0.0835, +0.1116], 1277 series. MECHANICAL_DERIVED.
- fold 0, node 9 → 7, lag 1: held-out 상대 MSE 개선 +0.0966, 95% CI [+0.0817, +0.1106], 1276 series. MECHANICAL_DERIVED.
- fold 3, node 11 → 7, lag 1: held-out 상대 MSE 개선 +0.0873, 95% CI [+0.0708, +0.1039], 1276 series. MECHANICAL_DERIVED.
- fold 3, node 9 → 7, lag 1: held-out 상대 MSE 개선 +0.0825, 95% CI [+0.0660, +0.0998], 1276 series. MECHANICAL_DERIVED.

Parent가 있는 node/phase/fold 전이 평가 50건 중 own-history 전이 대비 log probability gain의 CI 하한이 양수인 경우는 43건입니다. 이는 피처 상태 예측 결과이며, 최종 탐지 TS-AUC 이득과 구분합니다. 희소 context는 독립 series 8개 미만이면 own-history로 backoff하고, Dirichlet 0.5 평활화를 적용했습니다. 전체 3^16 상태 조합은 만들지 않았습니다.

edge/전이 재현 CI는 선택된 학습 그래프에 조건부인 기술통계이며 다중검정 전체를 보정한 확증이 아닙니다. 학습 자료가 겹치는 다섯 그래프의 선택 빈도도 독립 다섯 번의 증명으로 해석하지 않습니다.

정상 raw Gaussian/AR/ARCH/heavy-tail에 같은 계산기를 적용한 대조에서 11개 연결이 선택됐고 11개가 계산식 유래 연결로 표시됐습니다. 모든 노드는 동일한 단변량 raw를 공유합니다. 서로 다른 노드의 동시 확인도 독립된 정보원의 확인을 뜻하지 않습니다. 독립적인 경제적 원인이나 변화 이전의 강한 선행 신호는 입증하지 않았습니다. 급격한 합성 변화와 대응 control의 변화 이전 예측은 정확히 같았습니다.

## 이상치와 지속 변화

| 대조 사례 | 방법 | 지표 | 5 fold 평균 |
|---|---|---|---:|
| N0 | M0 | null_eventual_alarm_rate | 0.031 |
| N0 | M3 | null_eventual_alarm_rate | 0.056 |
| N0 | AMPLITUDE_0.9_M0 | null_eventual_alarm_rate | 0.031 |
| N1 | M0 | null_eventual_alarm_rate | 0.013 |
| N1 | M3 | null_eventual_alarm_rate | 0.031 |
| N1 | AMPLITUDE_0.9_M0 | null_eventual_alarm_rate | 0.013 |
| N2 | M0 | null_eventual_alarm_rate | 0.084 |
| N2 | M3 | null_eventual_alarm_rate | 0.028 |
| N2 | AMPLITUDE_0.9_M0 | null_eventual_alarm_rate | 0.084 |
| N3 | M0 | null_eventual_alarm_rate | 0.113 |
| N3 | M3 | null_eventual_alarm_rate | 0.081 |
| N3 | AMPLITUDE_0.9_M0 | null_eventual_alarm_rate | 0.113 |
| O1 | M0 | null_eventual_alarm_rate | 0.819 |
| O1 | M3 | null_eventual_alarm_rate | 0.628 |
| O1 | AMPLITUDE_0.9_M0 | null_eventual_alarm_rate | 0.819 |
| OB16 | M0 | null_eventual_alarm_rate | 0.875 |
| OB16 | M3 | null_eventual_alarm_rate | 0.850 |
| OB16 | AMPLITUDE_0.9_M0 | null_eventual_alarm_rate | 0.875 |
| V2 | M0 | postbreak_first_detection_rate | 0.531 |
| V2 | M3 | postbreak_first_detection_rate | 0.550 |
| V2 | AMPLITUDE_0.9_M0 | postbreak_first_detection_rate | 0.531 |
| V4 | M0 | postbreak_first_detection_rate | 0.084 |
| V4 | M3 | postbreak_first_detection_rate | 0.044 |
| V4 | AMPLITUDE_0.9_M0 | postbreak_first_detection_rate | 0.084 |
| D1 | M0 | postbreak_first_detection_rate | 0.594 |
| D1 | M3 | postbreak_first_detection_rate | 0.412 |
| D1 | AMPLITUDE_0.9_M0 | postbreak_first_detection_rate | 0.594 |
| D2 | M0 | postbreak_first_detection_rate | 0.994 |
| D2 | M3 | postbreak_first_detection_rate | 0.950 |
| D2 | AMPLITUDE_0.9_M0 | postbreak_first_detection_rate | 0.994 |
| LATE_V2 | M0 | postbreak_first_detection_rate | 0.469 |
| LATE_V2 | M3 | postbreak_first_detection_rate | 0.478 |
| LATE_V2 | AMPLITUDE_0.9_M0 | postbreak_first_detection_rate | 0.469 |

임계값은 외부 점수와 독립인 정상 합성 calibration 256개 series의 1,024시점 최대 점수에서 정했습니다. 4,096시점 정상 실험은 장기 오경보 진단이며 anytime 보장이 아닙니다. 표의 null alarm은 낮을수록, 진짜 변화 탐지는 높을수록 유리합니다. 탐지 지연은 탐지율·미탐과 함께, 이상치 회복 시간은 censoring과 함께 읽어야 합니다. 0.9×M0는 자체 재보정 임계값과 M0의 같은 수치 임계값을 모두 비교해 단순 점수 축소 효과를 드러냅니다. 상세 지표는 COUNTEREXAMPLE_RESULTS_NEXT6.csv와 METRIC_DIRECTIONS_NEXT6.csv에 있습니다.

실제 held-out 사례는 M3를 미리 지정하고 fold 0의 고정 ID 해시 순서에서 성공·지연·오경보·누락을 골랐습니다. 사례 부재는 CASE_STUDIES_NEXT6.json에 명시합니다. tau는 그림에 사후 표시했으며 예측기에 전달하지 않았습니다.

## 검증과 결정

- M2: 승격 보류 — delta_M0_at_least_.0012, delta_M1_at_least_.0002, paired_lower_M0_positive, paired_lower_M1_positive, at_least_4_positive_folds, worst_delta_at_least_minus_.001, delta_input_complete_M1_STRICT_at_least_.0002, paired_lower_vs_M1_STRICT_positive
- M3: 승격 보류 — delta_M0_at_least_.0012, paired_lower_M0_positive, paired_lower_M1_positive, at_least_4_positive_folds, worst_delta_at_least_minus_.001, paired_lower_vs_M1_STRICT_positive
- M4: 승격 보류 — complete_5_folds, delta_M0_at_least_.0012, paired_lower_M0_positive, paired_lower_M1_positive, at_least_4_positive_folds, worst_delta_at_least_minus_.001, scalar_latency_ratio_at_most_3, paired_lower_vs_M1_STRICT_positive

기존 및 새 테스트 1371개, 실패 0개, 오류 0개입니다. prefix/suffix·길이 변경, batch/scalar, 저장·로드, 상태 초기화·다른 series와의 교차 실행, 결정론, tau/label/phase/ID 입력 거절을 검사했습니다. 재현은 저장된 모델을 실제 H와 한 점씩 관측에 다시 적용한 결과이며 모든 모델을 처음부터 다시 학습했다는 뜻은 아닙니다. latency는 LATENCY_NEXT6.json의 실제 predict_one 측정치로 확인합니다.

보호 대상 기존 파일 701개가 그대로입니다. Shadow 1,600개와 Seal A/B/C/D는 열지 않았고 새 Shadow·split·public/cloud/reduced test를 만들거나 읽지 않았습니다. BL80 champion, D2 robustness, J102 fallback, B2/X reference 역할을 바꾸지 않았습니다. 모든 NEXT6 모델은 DEV 연구 산출물로만 보존합니다.

목표 연구 예산 10시간과 별도 audit 45~60분을 실제 사용 시간으로 주장하지 않습니다. 현재 세션 경과 1.63시간, 계측 작업의 중복 제거 합계 1.24시간입니다. 연구와 audit의 겹침·미계측 편집/검토 시간은 TIME_ACCOUNTING_NEXT6.json에 따로 기록했습니다. 시간을 채우는 대기나 추가 후보 탐색은 하지 않았습니다.

## 재현과 다음 작업

[인수인계](01_RESEARCH_HANDOFF_NEXT6.md)에 실행 명령·원본 의존성·ZIP 제외 범위를 정리했습니다. [다음 세션 계획](NEXT_SESSION_PLAN_NEXT6.md)은 이번 외부 점수로 미세 조합을 더 만드는 대신 미재현 원인과 계산식 유래 연결을 다루도록 작성했습니다.

# NEXT5 중간 연구 해석: 전체 선별 완료 후

이 문서는 중간 해석이다. 장기 합성 검증, 대체 분할, H-regime 제외 검증, 전체 스트리밍 API 검증과 최종 감사는 아직 완료되지 않았다. 최종 handoff 또는 배포 결정으로 사용하지 않는다.

## 비교의 기준

이번 평가의 기준 모델은 ResearchPool 6,400개만으로 다시 학습한 BL80이다. PRIMARY 5-fold 평균은 0.6203652100698587이다. 기존 8,000개 DEV에서 기록한 고정 배포 모델의 0.6232684128173095와는 표본과 분할이 다르므로 직접 차감하지 않는다.

7개 계열의 사전등록 후보 20개가 screen을 마쳤다. M2 sign-run은 고정 screen 기준에 실패해 full 학습하지 않았다. 나머지 19개는 5-fold full 평가와 같은 2,000회 whole-series 재표본화를 마쳤다.

## 현재 가장 높은 점수와 승격 기준

X2_SCALE_TAIL의 평균은 0.6215608600890692, 기준 대비 개선은 +0.0011956500192103591이다. 고정 OOF의 paired bootstrap 구간은 [0.0005291145497019923, 0.0018663422198899457]이고, 재표본에서 개선이 양수인 비율은 1.0이다.

그러나 사전 승격 조건은 평균 개선 **0.0012 이상**이다. 실제 값은 이 조건에 미달한다. 반올림하면 같은 숫자로 보인다는 이유로 승격하지 않는다. 현재까지 공통 PRIMARY 자격을 얻은 후보는 없다. 사전 계약에 따라 적격 후보가 끝까지 없으면 Shadow는 닫힌 채로 남는다.

같은 ResearchPool 학습 조건의 B2와 비교하면 X2의 차이는 +0.00008536291486704872이며, paired 구간은 [-0.0014116209851102236, 0.001492089907038894]이다. D2 대비 차이는 +0.00045705384599303844, 구간은 [-0.00008422800251901285, 0.000997675764414679]이다. 따라서 기존 scale 표현을 넘어선 확실한 새 성능 세대를 확보했다고 해석할 수 없다.

## 후보 순위의 불확실성

X2와 2위 X1의 차이는 +0.00010260464973255168이다. 동일한 재표본화에서 이 차이의 구간은 [-0.00010943783344274093, 0.000317392683116807]이다. 점수 차이는 고정된 0.0007 winner-gap 주의 기준보다 작다.

19개 full 후보를 함께 비교할 때 X2가 재표본 최고점인 비율은 0.6515이다. 이는 실제로 가장 좋은 모델일 확률이 아니다. 이미 고정된 OOF 예측의 표본 변동을 묘사한 값이다. 후보 구성, X 부모 선택, 이전 DEV 반복 사용 및 학습 자체의 불확실성은 보정하지 않았다. 각 PRIMARY fold를 하나씩 제외한 기술적 요약에서는 X2가 계속 1위지만, 이를 새로운 검증 분할이나 독립 확인으로 세지 않는다.

## 무엇이 개선됐는가

S4B 통합 scale specialist는 SCALE_DOMINANT 106개 양성 시계열에서 +0.010040280836845494, 구간 [0.005950024250004296, 0.014685157337884384]를 보였다. 전체 평균 개선은 +0.0009173128710316502에 그친다.

X2는 같은 SCALE 구간에서 +0.013274976264466831, 구간 [0.00776024471703832, 0.019673940904037717]이다. 온라인 시간 400 이후에도 +0.001070536959912105, 구간 [0.00019357865723401425, 0.0019508284916262876]를 보였다. 다만 변화 직후 age 0–4의 구간은 0을 포함한다. DEPENDENCE_DOMINANT 개선의 구간도 0을 포함한다.

이 분류는 실제 잠재 생성 원인의 정답이 아니다. 고정된 사후 관측 진단이며 모델 입력으로 사용하지 않는다. SCALE은 지원 기준 100개를 근소하게 넘는다. MEAN 및 TAIL은 지원 수가 부족하므로 안정적 성능 결론의 근거로 삼지 않는다.

## 다음 검증의 범위

각 계열의 최고 full 후보 하나씩 7개에 대해, 시작 시 고정한 ALT_A/ALT_B와 기존 H-only 5-regime의 제외 학습을 수행한다. 비교 기준과 후보는 각 검증에서 정확히 같은 ResearchPool ID로 학습한다. 총 135회 진단 목적 learner fit이며 새로운 후보나 파라미터 검색은 아니다. 기존 PRIMARY strata에서 고정한 재표본화 가중치를 ID로 제한해 재사용하고 새 real-data draw를 생성하지 않는다.

합성 검증은 기존 DGP의 모든 서로 다른 변화 전 배경을 포함하므로, 실행 기준으로 9개 calibration + 21개 counterexample + 4개 long-null, 총 34개 cohort이다. 초기 계획의 ‘29’ 표기는 배경 수를 적게 센 설명 오류이며, 실행 메뉴나 생성 규칙을 바꾼 것이 아니다. 완료된 full 후보의 5개 산출물 및 screen에서 닫힌 M2의 2개 산출물, 총 97개를 비교한다. 겹치는 학습 fold를 독립 반복 실험처럼 취급하지 않는다.

실시간 실행 어댑터의 24개 검증은 통과했다. 특징값은 원래 float32 특징과 정확히 일치했고, float64 결합 확률은 10⁻¹⁵ 허용 오차 이내였다. 모든 실제 ResearchPool OOF 출력, 한 번만 순회 가능한 입력, 미래 길이와 suffix 접근 방지, 300개 prefix, 순서 반전 초기화 검증을 추가 수행한다. 속도 측정은 원래 timing 표본과 ResearchPool의 교집합 102개만 사용한다.

## 근거 산출물

- `artifacts/next5/references/comparisons/RESULTS.json`
- `artifacts/next5/experiments/X2_SCALE_TAIL/full/RESULTS.json`
- `artifacts/next5/selection_uncertainty/RESULTS.json`
- `artifacts/next5/selection_uncertainty/MATCHED_REFERENCE_COMPARISONS.csv`
- `artifacts/next5/forensic/X2_SCALE_TAIL/PAIRED_FRACTURES.csv`
- `artifacts/next5/forensic/S4B_UNIFIED_SCALE_SPECIALIST/PAIRED_FRACTURES.csv`
- `artifacts/next5/stability/POLICY_LOCK.json`
- `artifacts/next5/synthetic/COHORT_COUNT_CLARIFICATION.json`
- `artifacts/next5/runtime_validation/CONTRACTS.xml`
- `artifacts/next5/runtime_validation/API_POLICY_LOCK.json`
- `artifacts/next5/runtime_validation/LATENCY_POLICY_LOCK.json`

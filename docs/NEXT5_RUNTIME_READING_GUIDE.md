# NEXT5 속도와 상태 메모리 측정 읽기

이 문서는 동결된 측정 계약을 설명한다. 측정의 완료 여부와 실제 수치는 `artifacts/next5/latency/fixed_v1/RESULTS.json` 및 최종 `LATENCY_NEXT5.json`을 따른다. 문서를 작성한 것 자체는 통과 영수증이 아니다.

기존 latency 표본 128개에서 NEXT5 ResearchPool에 속하는 102개만 원래 순서대로 사용한다. 새로운 표본 seed를 뽑거나 결과에 따라 시계열을 교체하지 않는다. 원래 frozen BL80, ResearchPool BL80 fold 0, 일곱 대표 후보 fold 0을 세 가지 고정 순서로 측정한다. 이 모델들은 품질 평가를 위한 새 학습 결과가 아니다.

| 수치 | 실제 측정 대상과 해석 |
| --- | --- |
| point median / p95 / p99 | 실제 표본의 현재 값에 대한 개별 `predict_one` 호출 시간 분포. 각 반복 안의 분위수이며 세 반복의 전체 호출을 합친 분위수가 아니다. |
| historical initialization | 각 시계열의 H로 상태를 만드는 시간의 중앙값. 모델 파일 로드와 최초 JIT 준비 시간은 별도 필드다. |
| amortized real sequence | 실제 표본 전체의 H 초기화 시간과 point loop 시간을 합해 실제 point 수로 나눈 값. H 길이와 online 길이 분포에 영향을 받는다. |
| 10M direct | 고정된 10,000개 입력 버퍼를 하나의 상태에서 반복해 1,000만 번 처리한 실제 wall time. 배열 유한성 검사, 해시 갱신과 100만 회 checkpoint 검사도 포함한다. |
| mutable state bytes | NumPy 상태 배열의 byte 수. 고정 H에서 online 길이가 늘 때 크기가 유지되는지 확인한다. 서로 다른 H 길이 사이에는 크기가 다를 수 있다. |
| immutable tree bytes | 수정하지 않는 tree 배열의 byte 수. mutable state와 별도다. |
| process RSS | 같은 프로세스에 올라온 모델, JIT 및 기타 메모리를 포함하는 프로세스 측정치. 후보 하나의 독립 메모리 사용량이 아니다. |

1,000만 회 스트림은 긴 실행에서의 유한 출력과 고정 상태 크기를 검사하는 공학적 스트레스다. 반복 버퍼로 새로운 독립 시계열이나 실제 대회 표본이 늘어나지 않는다. 이 출력으로 TS-AUC나 일반화 성능을 다시 선택하지 않는다. 세 반복 사이에는 출력 float64 해시가 같아야 한다.

동결된 3배 판정은 기존 **FROZEN_PRIMARY**를 분모로 한 point median, amortized real sequence, 10M direct의 세 비율에 적용한다. 각 반복에서 후보/동시 측정 reference 비율을 구한 뒤 그 비율 세 개의 중앙값을 사용한다. 이는 원시 시간의 중앙값끼리 나눈 값과 반드시 같지는 않다. `all_ratios_within_3x`가 p95, p99, H 초기화 또는 상태 byte의 비율까지 모두 판정한다는 뜻은 아니다. 그 수치는 원래 block 표에 그대로 보고한다. ResearchPool BL80 대비 비율도 보조 비교로 보존한다.

측정 중 다른 workspace Python 프로세스를 실행하지 않고, 매 100만 회 checkpoint에서 해당 조건을 검사한다. 운영체제 작업과 native PowerShell의 상태·영수증 확인은 생길 수 있다. 45–60분 병목 검토가 측정과 겹칠 때는 별도 동결된 native metadata reviewer를 사용하며, 실제 검토 이력은 최종 latency 보고서에 포함한다. 측정 block을 유리한 것만 골라 버리거나 순서를 바꾸지 않는다.

근거는 `artifacts/next5/runtime_validation/LATENCY_POLICY_LOCK.json`, `scripts/profile_next5_runtime.py`, `artifacts/next5/bottlenecks/TIMING_REVIEW_POLICY_LOCK.json`이다. 속도 기준 통과만으로 ResearchPool 품질 기준 실패를 상쇄하거나 새 역할을 승격하지 않는다.

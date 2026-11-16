# NEXT5 한계

새 primary 또는 NEXT5 역할은 없다. X2가 ResearchPool 최고였지만 원래 평균 증가분 기준 +0.0012에 미달했다. 미개봉 Shadow의 점수를 만들거나 일반화 성공/실패로 해석하지 않는다.

동결 구현의 worst_delta는 후보 최저 fold score에서 기준 모델 최저 fold score를 뺀 값이다. 사용자 요구를 각 대응 fold의 악화 한도로 읽으면 minimum_fold_delta가 더 엄격한 비교다. 두 정의와 각 통과 여부를 표에 분리했다. S4A는 전자의 하한을 통과하지만 최소 대응 Δ는 약 -0.0012384로 후자의 -0.001 한도에 미달한다. 원래 결과나 규칙을 사후 덮어쓰지 않았다. 모든 19개 full 후보가 이미 평균 개선 기준에 미달하므로 어느 정의에서도 finalist와 최종 역할은 바뀌지 않는다.

원래 median_delta 및 해당 bootstrap 구간도 각 모델 fold score 중앙값끼리의 차이다. 대응 fold 차이의 중앙값은 별도 열로 보고하며 원래 구간을 그 값의 CI로 부르지 않는다. P2의 중앙값 차이는 +0.0003344897이지만 대응 차이 중앙값은 -0.0001361490으로 하한 통과 여부가 다르다. X2는 각각 +0.0022104184와 +0.0010587485다. 이 차이 역시 평균 기준 미달과 미선정 결론을 바꾸지 않는다.

기존 8000 DEV를 장기간 반복 사용했다. NEXT5 Shadow 1600개는 이번 연구에서는 사용하지 않았지만 과거 DEV 탐색의 일부였다. X parent 선택은 전체 ResearchPool 결과를 본 뒤 수행했으므로 전체 후보 설계가 nested 검증된 것은 아니다. 2000 whole-series bootstrap과 centered max statistic은 고정 OOF에 조건부이며 과거 적응적 선택과 학습 불확실성을 제거하지 않는다.

H-regime 변환에는 과거 전체 DEV H로 fit한 역사적 transductive 요소가 있다. 이번 RP descriptor는 H만으로 새로 계산했다. Mechanism signature는 사후 해석이며 실제 생성 원인의 정답이 아니다. SCALE positive 106개, MEAN 16개, TAIL 64개이므로 작은 subset에서 확증을 주장하지 않는다.

H-exclusion은 크기가 1179/507/133/801/3780인 다섯 regime을 하나씩 제외하고 학습한다. 각 학습 크기는 5221/5893/6267/5599/2620이며, 후보와 기준 모델의 ID는 각 fold 안에서 일치한다. 보고 평균은 다섯 regime 차이에 동일 가중치를 둔 값이다. 배경 변화와 학습 표본 수 변화의 효과를 분리하거나 전체 RP pooled TS-AUC로 해석하지 않는다.

합성 결과는 동일한 고정 생성기와 seed에 조건부다. 97개 CV artifact는 훈련 fold가 겹치므로 독립 표본이 아니다. 평균 artifact TS-AUC는 ensemble prediction의 TS-AUC가 아니다. 별도 새 전체 RP/DEV 배포 모델은 만들지 않았다.

P3는 사전등록대로 빠른 peak와 ungated EWMA64를 비교했다. 사용자 예시의 confirmation이 있을 때만 증가하는 persistent 상태는 구현하지 않았으므로, 그 더 넓은 구조의 실패로 일반화하지 않는다. P1의 confirmation feature와 P2의 threshold occupancy는 별도 후보다. E1 역시 사전등록된 unreset 누적 mixture이며 최근 age 제한은 E2에만 적용했다. 고정 크기 배열과 제한된 evidence horizon은 다르다. E1과 E2는 별도 학습 모델이므로 window 제한만의 인과적 ablation으로 설명하지 않는다.

단발·반복 충격 뒤 probability excess 감소에는 0.9 BL80 혼합의 진폭 축소가 포함될 수 있다. 선형 분해는 recovery time을 분해하지 않는다. 첫 16연속 tolerance 충족은 영구 복구가 아니며, 2048 censoring과 recovered-only 통계를 구별한다. postbreak detection의 분모는 전체 변화 시계열이고 조건부 delay는 검출 성공한 사례만 포함한다.

임계값은 독립 null의 1024 horizon에서 정했다. 16384 eventual alarm에 같은 alpha 보장을 붙이지 않는다. H-estimated dependent likelihood는 ranking evidence이며 엄밀한 e-process/anytime validity를 입증하지 않았다.

N0–N3의 counterexample와 long_null은 같은 scenario/horizon이어도 생성 시드 partition이 다른 코호트다. 원래 paired alarm 표의 누락된 종류 열은 전수 집계 실행 전에 동결 반복 순서와 SHA 영수증으로 복원했다. 원래 폴드별 표와 모든 수치는 보존했다. 최종 paired alarm 비교에는 cohort_kind를 포함해야 하며, 수정 계약과 97개 표 보존 영수증을 함께 전달한다.

Late-break 진단은 기존 onset=512의 85개 seed만 사용한 사후 층화다. Scale 방향 진단과 impulse 진폭 분해도 결과를 본 뒤 추가한 해석 도구다. 새 후보나 gate를 만들지 않았으며 독립 확인으로 세지 않는다. Tree gain/split 사용량은 훈련 learner의 통계로, held-out 중요도나 인과적 제품항 기여가 아니다.

긴 무변화의 에너지·누적 likelihood 진단도 원래 N0–N3 각 256개 경로의 사후 해석이다. 누적 Gaussian LR와 clipped standardized innovation 에너지 합의 산술 관계를 확인한다. 양의 working likelihood 비율은 학습 모델의 알람 확률과 다르고, 그 자체로 false alarm의 인과적 기여를 분리하지 않는다.

원래 D2 실제 파일의 NEXT4 반복 이상치 악화는 남아 있다. SCALE 보조 역할 유지가 전체 성능·반복 충격에 대한 새 보장이 아니다. 원래 D2 latency는 역사적 측정으로 표시했다.

애플리케이션 admission guard와 파일·실행 증거를 검증했지만 OS sandbox 증명이나 외부 계정의 모든 활동 감사는 아니다. 같은 파일의 raw byte hashing과 seal 관측/label decoding은 다르다. 이번 작업은 reduced/public/cloud 제출을 하지 않았다.

Shadow/SEAL 0행과 미개봉 표시는 허용된 반환 행, 캐시, 훈련·평가·특징 분석 및 guard 상태 기준이다. 원래 raw reader는 ID 조건을 Arrow scanner에 전달하고 반환된 ID를 확인했다. Parquet 내부 page/row-group decoder를 별도로 계측하지 않았으므로 내부 디코딩까지 0이었다는 독립 검증을 주장하지 않는다. 이 더 강한 경계 요구와 분석에 사용된 행의 검사는 구별해야 한다.

새 NEXT5 role이 없어 새 raw→fit 재현은 해당하지 않는다. 기존 역할의 RP batch/stream parity와 raw-free 복사본의 새 프로세스 검증을 새 raw 재학습이라고 부르지 않는다. 최종 감사의 관측 시간과 실측 범위는 FINAL_AUDIT_NEXT5.json을 따른다.

실제 API 검사 두 건은 원래 core import의 Numba 캐시 디렉터리 생성 시도를 차단하고 추론 전에 실패했다. 뒤따를 미실행 다섯 건을 보존하고, 기존 read-only bootstrap을 먼저 적용하는 별도 진입점으로 7개 검사를 새 경로에서 수행했다. 원래 수학·모델·descriptor·prefix·쓰기 차단 검사는 보존했다. 원래 실패와 미실행 상태는 quarantine으로 남기고 성공 영수증을 연결했다. 작업 관리자 일시 정지 구간은 연구 시간으로 소급 인정하지 않았다.

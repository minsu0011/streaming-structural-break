# NEXT 최종 구현 검토

이 문서는 선택 완료 후 추가 감사에서 확인한 구현의 의미와 검증 경계를 정리한다. 모델, feature 구현, 선택 기준, 예측값을 수정하지 않았다.

## Serial variance의 수식

`src/next/serial_channel_evidence.py`는 H에서 추정한 lag-1 상관을 −0.95~0.95로 제한한다. 관측이 들어온 후 이 상관이나 age별 분산을 다시 추정하지 않는다. 표준화한 채널도 이후 −12~12로 잘리므로 Gaussian AR(1)은 근사 모형이다.

길이 n의 채널 합 S에 대해 분산은 `v = n + 2 Σ(n−k)ρ^k`이다. GLR 요약은 `S²/(2v)`다. 평균 변화 μ의 사전분산 q를 사용한 Bayes 요약은 다음 적분에 대응한다.

`BF = ∫ Normal(S; nμ, v) Normal(μ; 0, q) dμ / Normal(S; 0, v)`

q는 0.25와 1이고, age별 BF는 사전에 정한 구간 폭을 가중치로 평균한 뒤 로그를 취한다. `tests/test_next_serial_bayes_integral.py`는 양·음의 상관 조건에서 전체 공분산 행렬로 합의 분산을 계산하고, SciPy 수치 적분으로 위 적분을 직접 평가해 runtime 출력과 비교한다. 기존 테스트는 ρ=0일 때 원본 채널과 정확히 같은 출력인지, H 파라미터가 온라인 관측에 의해 바뀌지 않는지도 확인한다.

이 검증은 구현이 명시된 근사 식을 계산한다는 근거다. 해당 BF가 실제 에너지 채널의 정확한 likelihood ratio이거나, 전체 관측 기간의 오경보를 제한하는 martingale이라는 근거는 아니다. EWMA와 CUSUM 요약에는 이 serial 분산 보정을 적용하지 않는다.

## 실제 학습과 추론 경계

`serial_training.py`와 대응 variance trainer는 ID의 형식·중복·허용 집합을 먼저 검사한다. 허용 DEV ID가 아니면 H/O/tau를 변환하거나 길이를 읽기 전에 건너뛴다. 마지막에 DEV 8,000개와 known ID 전체 10,000개의 누락 여부를 검사한다. DEV 행은 고정 fold와 ID 순으로 모아 정해진 학습법을 적용한다.

실제 final training 검증은 원시 DEV iterator와 2,000개 opaque seal tuple을 제공한다. opaque 객체의 배열·길이·숫자·진릿값 접근은 예외를 내도록 했으며, 만들어진 모델의 바이트를 이전 독립 raw 재현 및 검증된 all-DEV 모델과 비교한다. 이 객체들은 실제 sealed 값이 아니다.

전달용 infer는 readiness `None`을 내보내기 전에 모델 로딩과 합성 warmup을 수행한다. 실제 H/O iterator는 그 시점까지 소비하지 않는다. 이후 각 O 관측마다 유한한 Python float 하나를 반환한다. 실제 미래 O 길이에 의존하거나 미래 원소를 미리 읽지 않는지는 별도 single-pass·suffix·cold API 계약으로 검증한다.

Numba cache 생성은 전달 entry가 project import 전에 억제한다. 원래 실패했던 `next_main.py`를 전달 모델 entry로 선택하지 않는다. 최종 역할과 entry는 `DEV_CANDIDATES_NEXT.json`이 지정한다.

## 성능표와 그림

원본 5-fold가 PRIMARY/ROBUST 품질 판정의 기준이다. screen, alternate split, H-regime 제외 재학습, mechanism별 OOF 절단을 구분한다. 모델 평균의 반올림 값으로 역할을 다시 선택하지 않는다. J102와 Y001의 2.8e-7 차이는 새 과학적 효과로 해석하지 않는다.

장기 null 그림 v1은 공유 y축의 autoscale이 첫 패널 범위에 고정되어 일부 큰 오경보율을 잘랐다. v1과 당시 driver를 보존하고, 전체 5개 regime의 Wilson 상단을 포함하는 공통 범위를 명시한 `long_null_v2`를 최종 문서에서 사용한다. v2의 5% 그림은 78.2%까지의 큰 값과 신뢰구간을 모두 표시한다. 원본 수치와 모델은 그대로다.

## 전달 파일의 독립 확인

`scripts/verify_next_handoff_manifest.py`는 원래 프로젝트나 raw data 없이 ZIP의 manifest, 정확한 파일 집합, 크기, CRC 및 SHA256을 검사한다. 중복 이름, Windows에서 충돌하는 대소문자 이름, 절대 경로·상위 경로, symlink를 거절한다. 압축을 풀거나 joblib을 역직렬화하지 않는다.

ZIP 내부 SHA256은 내부 일관성을 확인한다. 전체 ZIP SHA256을 별도 영수증과 대조해야 전달 도중 변조·교체도 확인할 수 있다. 최종 새 환경 검증의 실제 PASS 여부와 전체 테스트 수는 `artifacts/next/portable_delivery/RESULTS.json`, `artifacts/next/final_tests/RESULTS.json`, `artifacts/next/final_audit/RESULTS.json`에서 확인한다.

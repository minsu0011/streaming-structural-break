# NEXT3 공식 실행 계약 대조

2026-09-10 05:59 UTC에 공개 공식 소스와 기존 고정 스냅샷을 대조했다. 현재 `runner.py`와 `scoring.py`는 기존 파일과 byte 단위로 동일하다. 기존 scorer parity와 실행 계약의 근거를 계속 사용할 수 있다. 실제 검사는 `artifacts/next3/maintenance/official_contract_review_v1/RESULTS.json`에 원격 commit, URL, SHA256으로 남겼다.

[공식 문서](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time)는 초기 준비 완료 표시, 온라인 관측값마다 한 번의 예측, 미래 관측 접근 금지, 결정적 추론을 요구한다. 학습이 필요한 모델은 학습 코드도 제공해야 한다. TS-AUC는 같은 온라인 시점의 양성·음성 쌍을 비교한다. 이 요구를 raw 학습 재현과 실제 public API 검증의 기준으로 유지한다.

| 계약 | 로컬 근거 | 검증 범위 |
| --- | --- | --- |
| 첫 데이터 접근 전에 `None` 반환 | 두 역할의 public API receipt | 각 8,000 DEV / 4,014,405개 관측 |
| 현재 관측 한 개당 유한한 확률 한 개 | 동일 receipt | 길이·미래 suffix 접근을 거부하는 iterator |
| 시계열 상태 초기화 | 동일 receipt | 128개 역순 재생 |
| 추론 파일 쓰기 금지 | 동일 receipt | Python audit hook, OS 격리는 아님 |
| 단일 프로세스 요청 | `next3_lean_performance_main.py` | `INFER_PARALLELISM = 1` |
| 데이터와 예측의 float32 전송 | [공식 runner 고정 commit](https://github.com/crunchdao/competitions/blob/c8b40ad6cd1c8c6e1a9de3d4725645856db7d0ac/competitions/structural-break-real-time/scoring/runner.py) | 로컬 fingerprint도 float32 bytes; 직접 시간 측정에는 SDK IPC가 없음 |
| scorer 구현 | [공식 scorer 고정 commit](https://github.com/crunchdao/competitions/blob/61586d8248e16381ea3ccbf292156150e6028a06/competitions/structural-break-real-time/scoring/scoring.py) | 기존 스냅샷 SHA256과 동일 |

설치된 `crunch-cli 12.0.2`, `crunch-convert 0.11.0`, `crunch-encrypt 0.1.0`은 distribution metadata만 읽었다. SDK import, `.test`, reduced 평가, public 제출, cloud 실행은 수행하지 않았다. 공개 문서 조회와 SDK 실행은 별개다. Linux/cloud의 라이브 실행·실제 할당량·다른 CPU에서의 동일성을 입증한 것은 아니다.

원본 receipt의 `human_reviewed_utc`는 필드 이름 오류다. 이 검토는 같은 AI agent가 수행했으며 인간의 독립 검토가 아니다. 원본 파일은 그대로 보존하고 이후 source의 필드 이름을 `agent_reviewed_utc`로 바로잡았다. `source_label_correction/CORRECTION.json`이 원본 실행 source와 수정 source의 hash를 연결한다. 원본 receipt의 다른 값은 바꾸지 않았다.

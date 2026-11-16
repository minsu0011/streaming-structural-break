# NEXT5 읽기 전용 import 복구

초기 발견과 중간 검증을 기록한 문서다. 아래 12:35 UTC 복구 완료 갱신은 실제 영수증에 근거하며, NEXT5 전체 연구나 최종 감사의 완료를 뜻하지 않는다.

## 12:35 UTC 복구 완료 갱신

일곱 새 `API_RO_...` 작업이 모두 성공했다. E2, M1, P1, R2, S4B, W2, X2 각각에서 RP 6,400개 시계열의 3,203,157개 실제 current-point 출력이 원래 저장된 5개 held-out artifact 출력과 bitwise 동일했다. 각 최대 차이는 0, 파일 쓰기 시도는 0, prefix는 300건, reverse-order reset은 100개 시계열이다. 합계 2,100개 prefix와 22,422,099개 모델별 출력이며, 서로 다른 실제 시계열 44,800개를 새로 확보한 것으로 해석하지 않는다.

`RESOLVED.json`은 2026-09-11 12:35:12 UTC에 PASS로 생성됐다. 원래 API 실패 2건과 시작 전 대체 5건을 성공 영수증으로 연결했고 새 fit은 없다. 초기 preflight까지 포함한 보존 quarantine 11건도 모두 해결됐다. 원래 상태와 실패 로그를 삭제하거나 성공 상태로 덮어쓰지 않았다.

복구 후 실행 그래프 감사 `integrity_audits/readonly_import_repair_resolved_v1/execution_graph/`는 413개 등록 작업, 76개 frozen Python source, 모든 보존 실패의 해결 근거를 확인했다. 연구·스트레스·최종 감사의 미완료 작업은 여전히 미완료로 표시한다.

## 발견한 문제와 범위

원래 `src.next5.streaming`을 새 프로세스에서 import하면 기존 feature module의 Numba `cache=True` decorator가 먼저 실행됐다. 이후 호출하는 cache 비활성화 함수보다 이 단계가 앞섰다. 고정 설치된 Numba 0.67.0의 cache locator가 디렉터리 생성을 시도했고, 실제 API 검증의 파일 쓰기 감사가 `os.mkdir`를 차단했다.

원래 E2와 M1 API 검증은 이 import 단계에서 exit 1로 실패했다. 해당 두 실행의 실패 로그와 상태는 보존했다. 아직 시작하지 않았던 나머지 다섯 API 작업도 원래 queue를 보존한 채 대체 작업에 연결했다. 모델 성능 기준 미달과 이 실행 오류는 다른 분류다.

실패한 원래 API 2건과 시작 전 대체한 API 5건, 초기 preflight의 실제 실패 1건과 시작 전 대체 3건을 합하면 보존된 quarantine은 11건이다. 이는 실제 실행 오류 3건과 시작 전 대체 8건이다. 일곱 API 대체 작업의 성공이 모두 확인되기 전에는 복구 완료라고 선언하지 않는다.

## 변경 내용과 불변 조건

공개 읽기 전용 entry는 `src.next5.readonly_streaming`이다. 기존 `src.next.readonly_bootstrap.install()`을 feature import보다 먼저 실행한 뒤, 원래 NEXT5 core의 `infer`와 `load_prepared`를 그대로 내보낸다. 이 bootstrap은 기존 NEXT4에서도 사용한 방식이며 프로젝트 Numba 함수의 disk cache 활성화만 막는다. 새 모델 학습, feature 계산식, frozen weights, descriptor, blend weight, 점수 기준, 쓰기 금지 검사는 변경하지 않았다.

`verify_next5_streaming_readonly_core.py`는 원래 검증 코드에서 import entry와 새 출력 디렉터리 두 문자열만 바꾼 파생본이다. 바뀐 import도 원래 파일 쓰기 감사가 이미 활성화된 위치에서 실행된다. 파생본의 정확한 두 치환과 원본 SHA는 `DERIVATION.json` 및 diff로 확인한다.

대체 작업은 `API_RO_...`라는 새 ID와 `runtime_validation/api_readonly_v1/`이라는 새 디렉터리를 사용한다. 각 검증이 실제 통과한 뒤에만 요약 JSON과 fold JSON의 바이트를 기존 소비 경로로 복사하며, `API_IMPORT_REPAIR.json`이 원본·대체 영수증 SHA를 연결한다. 실패한 원래 START 및 실행 로그를 성공 결과로 덮어쓰지 않는다.

Latency는 동일한 측정 코드·정해진 모델·블록을 유지하며, 시작 전 의존성만 일곱 성공 대체 API와 복구 분류 작업으로 교체했다. 원래 queue와 변경 전후 SHA를 보존했다. 자체 소유 supervisor를 정지했던 공백 430.004345초는 연구 시간으로 인정하지 않았다.

## 초기 11:08 UTC에 확인했던 범위

새 프로세스 cold probe는 결정론적 fixture 192개 current-point 출력에서 원래 batch oracle과 bitwise 동일했고 최대 절대 차이는 0이었다. 파일 쓰기 시도는 0, 차단된 disk cache 활성화 요청은 32였다. 이는 fixture 검증이며 전체 RP 결과를 대신하지 않는다. 해당 두 계약 테스트도 실제 통과했다.

E2의 새 실제 API 검증은 11:01:48 UTC에 통과했다. RP 6,400개 series의 3,203,157개 출력이 저장된 5개 held-out artifact의 offline 출력과 bitwise 동일했다. 최대 차이 0, 파일 쓰기 시도 0, prefix 300건, reverse-order reset 100개 series를 확인했다. 다른 여섯 후보까지 이 결과를 일반화하지 않는다.

11:07:37 UTC 실행 그래프 감사는 413개 등록 작업과 76개 frozen Python source 경로를 통과했다. 진행 중인 대체 API는 완료로 계산하지 않았고, 아직 연결되지 않은 최종 실패 해결 영수증도 미해결로 명시했다.

## 증거 위치

- `artifacts/next5/maintenance/READONLY_IMPORT_REPAIR/`: 원래 실패·queue, 정지/재개, frozen policy, 파생본 diff, cold probe, 등록 영수증.
- `artifacts/next5/runtime_validation/api_readonly_v1/`: 후보별 새 전체 API 실행.
- `artifacts/next5/runtime_validation/api/`: 통과한 대체 결과와 바이트 SHA 연결 영수증.
- `artifacts/next5/failure_resolutions/`: 성공 검증이 끝난 후에만 생성하는 원래 quarantine별 해결 근거.
- `artifacts/next5/integrity_audits/readonly_import_repair_in_progress_v1/execution_graph/`: 진행 중 상태를 그대로 반영한 의존성·source 검사.

초기 기록 시점에는 일곱 API 전체 복구의 완료를 주장하지 않았다. 이후 실제 완료 상태는 문서 상단의 12:35 UTC 갱신과 연결된 원 영수증을 따른다.

## 기존 runner 주석의 출처

`next5_child_runner.py`의 이전 native-crash 관련 주석은 NEXT4 runner에서 namespace를 바꾸며 상속됐다. 실제 근거는 NEXT4의 `SYN_NATIVE_FAILURE_001/INSTRUMENTATION_CHANGE.json`이다. 이 주석을 NEXT5에서 새 native crash가 발생했다는 증거로 해석하지 않는다. `maintenance/CHILD_RUNNER_COMMENT_CLARIFICATION.json`에 비교 근거를 남겼고 실행 중인 runner source는 수정하지 않았다.

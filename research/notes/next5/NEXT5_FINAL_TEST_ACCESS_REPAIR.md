# 최종 전체 테스트의 접근 경계 복구

별도 최종 감사는 2026-09-11 18:43:20 UTC에 시작했다. 최초 `FA002_COMPLETE_TESTS`는 18:50:33 UTC에 1,357개 테스트 중 실패 8건·설정 오류 37건·skip 0으로 종료됐다. 원래 결과, JUnit, 접근 차단 기록과 QUARANTINED 상태를 보존했다.

45개 실패·오류의 직접 원인은 동일했다. 일괄 reader guard가 `artifacts/validation/GROUP_ASSIGNMENTS.parquet`까지 과거 시계열 배열로 분류해 차단했다. 해당 파일에는 dataset_id·group_id·role·fold 네 열만 있으며, SHA는 기존 split lock에 고정돼 있다. 21개 차단 시도는 모두 이 메타데이터 경로였다. 이 결과를 관측값·라벨의 허용된 접근이나 테스트 통과로 바꾸어 해석하지 않는다.

기존 NEXT4의 scalar-adapter 테스트 네 개에는 추가 문제가 있었다. 모델 참조를 불러오는 함수가 동일한 두 fold 모델을 읽기 전에 불필요하게 전체 DEV 배열을 준비하는 경로를 탔다. 이 경로를 열어 주지 않고, 테스트 전용 어댑터가 원래 J102/C205 fold-0 모델 가중치와 영수증만 읽도록 했다. 원래 global-row 파일의 SHA가 서로 같은지도 바이트 수준에서 확인한다. 모델·가중치·혼합 비율과 원래 batch/stream·prefix·state assertion은 그대로다. 과거 학습 membership의 원시 배열 재검증을 수행했다고 주장하지 않는다.

`src/next5/final_test_compatibility.py`는 정확한 메타데이터 경로·SHA·네 열의 schema만 허용하고 기존 raw/과거 배열 차단 장치를 유지한다. 관측값·라벨·예측 열이 추가되거나 SHA가 다르거나 비슷한 이름의 다른 경로이면 이 예외를 적용하지 않는다. 원래 테스트 파일은 수정하거나 건너뛰지 않았다.

복구의 집중 검증 53개는 40.53초에 모두 통과했다. 원래 실패한 파일의 테스트들과 새 경계 계약 6개를 포함하며, 접근 차단 장치 아래에서 금지된 읽기 시도 0을 확인했다. `artifacts/next5/final_audit/test_access_probe_v1/`에 원본 JUnit과 metadata/model proof가 있다. 전체 재검사는 별도 `FA002R_COMPLETE_TESTS`와 `final_tests_repair_v1/` 경로를 사용한다. 집중 검증 통과만으로 전체 suite 성공을 선언하지 않는다.

원래 `final_audit/PLAN.json`과 그 안의 모든 Python 소스는 그대로 보존했다. 변경은 별도 파생 소스와 정확한 diff로 남겼다. 아직 시작하지 않았던 FA003의 의존성을 성공한 재검사·복구 확인 작업으로 연결하고, FA008은 이 변경의 전후 SHA와 실행 순서를 검증하는 파생 진입점을 사용한다. 별도 `PLAN_AFTER_TEST_REPAIR.json`이 원래 계획과 복구 정책을 연결한다. 감사 시작 시각·시간을 소급하거나 새 모델 학습을 하지 않았다.

실제 전체 재검사와 복구 확인이 통과한 후의 완료 근거는 `artifacts/next5/maintenance/FINAL_TEST_ACCESS_REPAIR/RESOLVED.json`이다. 최종 분류표는 원래 연구 단계 32개 행에 최초 전체 suite 오류 1개를 추가한 33개 행을 사용하도록 준비했다. 복구 전후의 원래 테스트 목록을 비교하며, 새 최종 보고서 발행은 성공한 전체 suite·후속 감사·최소 별도 감사 시간까지 모두 요구한다.

## 19:10 UTC 완료 확인

전체 재검사는 19:09:53 UTC에 1,363개 모두 PASS, failure/error/skip 0으로 완료됐다. 접근 장치의 금지된 읽기 시도도 0이었다. 19:10:13 UTC의 복구 확인 작업은 원래 1,357개 testcase의 이름과 중복 개수가 모두 보존됐고 네 legacy 모델 테스트가 같은 가중치를 사용했음을 확인했다. 원래 FA002 실패 상태는 그대로이며 성공한 FA002R과 해결 영수증을 연결했다. 최종 실패 분류는 33개 행·실제 실행 오류 5건·해결된 queue quarantine 12개다. 수치 재계산과 나머지 감사는 이 시점에 진행 중이다.

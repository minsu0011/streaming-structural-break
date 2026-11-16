# 최종 실행 이력 감사의 정책 파일 수집 복구

2026-09-11 19:50:44 UTC에 시작한 FA008_EXECUTION_GRAPH는 19:51:04 UTC에 JSONDecodeError로 실패했다. 19개 후보의 2470개 범위 통계 재검산과 공식 scorer 비교 95건은 이미 PASS였으며, 이 오류는 점수 계산과 관계없는 정책 파일 수집 단계에서 발생했다.

원래 감사는 `artifacts/next5` 아래의 모든 `*LOCK*.json`을 수집했다. 이 범위에 두 완료된 전체 suite의 `full_tests/pytest_tmp`가 들어갔다. `tests/test_next3_final_source_contract.py`의 `test_existing_source_or_guard_change_is_rejected`는 임시 프로젝트의 정책 파일을 의도적으로 `changed\n` 8바이트로 바꾸고 변경 탐지를 검사한다. 두 suite에 남은 이 합성 fixture가 실제 정책으로 잘못 분류됐다. 실제 SEAL lock이 손상된 것이 아니다. 실제 원본과 봉인 상태 검사는 FA006에서 별도로 통과했다.

수정은 정확히 `final_audit/final_v1/full_tests/pytest_tmp`와 `final_audit/final_tests_repair_v1/full_tests/pytest_tmp` 두 경로 안의 정책 파일 44개를 실제 정책 수집 대상에서 제외한다. 이 파일들, 원래 실패 로그·상태, 원래 감사 소스와 두 기존 PLAN은 그대로 보존한다. 다른 경로의 잘못된 JSON을 무시하는 예외는 추가하지 않았다. 실제 정책 집합, 소스 해시, DAG, 의존 작업 성공 시각 및 완료 로그 검사는 그대로 실행한다.

먼저 다섯 개 범위 계약으로 제외된 집합과 원래 바이트, 실제 정책의 전수 파싱, 비슷한 외부 경로의 비제외, 실패 보존을 확인한다. 별도 작업 FA008R_GRAPH_CHECK는 미해결 실패를 명시한 상태로 전체 실행 이력과 소스를 재검사한다. 완료된 작업의 성공 영수증을 근거로 FA008R_GRAPH_RESOLUTION이 원래 실패 해결 영수증을 작성한 뒤 모든 실패가 해결된 최종 그래프를 다시 확인한다. 성공 전에 해결을 주장하지 않는다.

새 작업의 정의를 실시간 queue 밖에서 먼저 동결하고 복구 기록과 `PLAN_AFTER_GRAPH_REPAIR.json`을 작성한다. 의존성이 충족되지 않은 resolver를 먼저 게시하고 실행 가능한 probe를 마지막으로 게시해 등록 중 실행 경합을 피한다. 아직 시작하지 않은 FA009의 의존성만 새 resolver로 연결한다. 실행 중 scheduler의 소스나 감사 시작 시각·과거 시간은 변경하지 않는다.

최종 보고서는 별도 파생 publisher와 한국어 보충 모듈을 사용한다. 최종 실패 분류는 34행, 실제 실행 오류 6건(Python queue 5건·native 기록 1건), 보존된 queue quarantine 13개로 갱신한다. 원래 33행 문서·검토는 그 시점의 기록으로 남긴다. 기존 전체 suite 1363개는 통과했으며 이 수정에서 테스트 소스나 모델 소스는 변경하지 않는다. 추가 검증은 이 감사 범위의 다섯 계약과 그래프 재검사다. 새 fit·예측·bootstrap draw는 0이다.

이 문서를 작성한 시점에는 복구 작업을 준비 중이다. 완료 여부는 `artifacts/next5/maintenance/FINAL_GRAPH_POLICY_SCOPE_REPAIR/RESOLVED.json`과 최종 `FINAL_AUDIT_NEXT5.json`을 확인한다.

복구 완료 기록: 2026-09-11 20:01:10 UTC에 최종 그래프와 복구 확인이 PASS로 끝났다. 426개 작업과 94개 Python 소스 경로를 검증했고, 보존된 quarantine 13개의 해결 근거를 모두 확인했다. 최종 그래프 영수증 SHA256은 `abd0e3601912b3cab3196080afc38a7dd94f10d18e32f5c230eb52489d038532`다. 그 뒤 별도 복사본의 BL80·B2·D2 cold inference 검증도 각각 PASS로 완료됐다. 최종 문서 게시와 Desktop archive 검증은 별도 단계의 영수증을 따른다.

# NEXT5 최종 감사에서 확인한 과거 감독 프로세스 오류

2026-09-11 12:53:56 UTC에 감독 프로세스가 `runtime/ACTIVITY.json`을 읽다가 `PermissionError`를 한 번 기록했다. 최종 감사에서 원래 error JSON과 stdout 로그를 확인했다. Windows 파일 공유 충돌 가능성은 있지만, 어떤 프로세스가 파일을 잠갔는지 OS 추적으로 확인하지 않았으므로 원인을 확정하지 않는다.

이 오류는 실행 중인 latency worker의 종료 오류가 아니었다. 원래 감독 프로세스가 같은 측정 작업을 계속 관찰했고, 해당 작업은 종료 코드 0으로 완료됐다. 27개 원래 timing block, 실제 270M 연속 업데이트와 반복 fingerprint 일치 결과가 모두 보존됐다. 대체 latency run이나 모델 재학습을 하지 않았다. 뒤이은 고정 검증도 완료됐다.

원래 오류 파일을 바이트 그대로 `artifacts/next5/maintenance/SUPERVISOR_ACTIVITY_READ_REVIEW/ORIGINAL_ERROR.json`에 복사해 raw-free ZIP에 보존했다. 같은 디렉터리의 `RESULTS.json`에는 원본 SHA, 감독 소스·owner, latency 완료 증거와 413개 비최종 작업의 상태 SHA가 있다. 402개 COMPLETE와 해결 영수증이 있는 11개 QUARANTINED를 확인했다.

감독 프로세스가 당시 남긴 `modeling_blocked=true`와 사유는 역사적 상태로 보존했다. 현재는 finalist lock과 별도 최종 감사 때문에 새 학습이 금지되므로 이 플래그를 지울 이유가 없다. 감사 작업은 정상 진행된다. 감독 코드·세션 값 변경이나 시간 소급 인정은 하지 않았다.

최종 32행 실패 분류표는 queued scientific/validation 작업과 native 병목 게시 실패를 다룬다. 이 감독 heartbeat 오류와 보고서 준비 명령의 인코딩 오류는 별도 보존 기록이다. 32행 표 또는 그 안의 실제 실행 오류 4건을 모든 shell·개발·감독 오류의 총수라고 해석하지 않는다. 관련 준비 명령의 기록은 `docs/NEXT5_REPORT_ENCODING_REPAIR.md`에 있다.

19:10 UTC 갱신: 위 32행은 이 검토 당시 연구 단계까지의 범위다. 이후 최종 suite의 접근 장치 오류 한 건과 성공한 전체 재검사가 추가됐다. 최종 통합 분류는 33행·실제 실행 오류 5건이며, 이 감독 오류가 별도 기록이라는 범위는 같다. `docs/NEXT5_FINAL_TEST_ACCESS_REPAIR.md`를 함께 읽는다.

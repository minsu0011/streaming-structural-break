# NEXT3 연구·감사 시간 경계 검토

최종 감사가 아직 시작되지 않은 2026-09-10에 heartbeat 구간의 시간 배분을 검토했다. 이전 V4는 관측이 인정된 한 구간 전체를 구간 끝의 phase로 분류했다. 실제 감사 시작이 구간 중간에 있으면 시작 전의 연구 시간이 감사 시간에 포함될 수 있었다. 과거 감사 시간을 수정한 사례는 아니며, 수정 전 감사 누적값은 0이었다.

`src/next3/phase_accounting.py`는 기존 관측 인정 조건이 참일 때만 해당 구간에 credit을 부여한다. 실제 UTC 감사 시작과 구간의 겹침을 0부터 구간 길이 사이로 제한해 감사 몫을 계산하고, 나머지를 연구 몫으로 배분한다. 관측되지 않은 구간은 전부 미관측으로 남긴다. 실제 시간 경계와 누적값을 기록하며 관측 공백을 소급해서 채우지 않는다.

새 28개 사례와 기존 프로세스 생명주기 검사를 함께 실행한 M689에서 71개가 통과했다. 7개 경계값, 미관측 구간, 잘못된 입력, 실제 `Supervisor.tick`의 시간 배분과 기존 프로세스 소유권 동작을 확인했다. 당시 최신 전체 검사는 별도의 1,019개 결과이며, 추가 28개가 포함된 최종 전체 실행은 이후의 실제 XML과 source lock을 따른다.

V4 PID 152852의 생성 시각과 소스 SHA를 확인하고 활성 자식 작업이 없을 때만 교체했다. V5 PID 13040은 2026-09-10T11:12:04.274316+00:00에 재개했다. 계획된 교체 receipt는 이 시점까지 5개다. 이전 누적 credit을 이어받고 재시작 공백은 미관측으로 남겼다. 이후에도 활동이 확인되지 않은 구간은 정상적으로 미관측에 쌓인다. 약 12시간의 세션 wall, 관측된 연구/감사 시간, 미관측 시간은 서로 다른 값이다.

확인 가능한 증거:

- `artifacts/next3/maintenance/PHASE_ACCOUNTING_IMPLEMENTATION_LOCK.json`: 수정 소스·검사 4개의 SHA.
- `artifacts/next3/maintenance/phase_accounting_correction_001/`: 이전 소스와 사전 수정 계획.
- `artifacts/next3/maintenance/PHASE_ACCOUNTING_REGRESSIONS.xml`: 71개 실제 검사 결과.
- `artifacts/next3/runtime/maintenance_updates/SUPERVISOR_V5_REPLACEMENT_PLAN.json`: 교체 전 프로세스·누적값·검사 증거.
- `artifacts/next3/runtime/maintenance_updates/SUPERVISOR_V5_REPLACED.json`: 실제 교체와 새 프로세스·공백 처리 증거.

이 유지보수에는 새로운 모델 fit, 품질 점수, seal 접근 또는 제출이 없다. 모니터링은 프로젝트 프로세스 안의 관측이며 OS 전체 작업 시간을 증명하지 않는다.

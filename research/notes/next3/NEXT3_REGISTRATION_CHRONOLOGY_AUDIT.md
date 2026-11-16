# NEXT3 등록·학습·평가 시각 대조

2026-09-10 08:14 UTC에 기존 메타데이터를 대조했다. 새 데이터 평가나 학습 없이 172개 시각 연결과 470개 근거 파일의 SHA를 검사했고, 기록된 순서의 역전은 없었다.

| 검사 범위 | 수 | 실제 연결 근거 |
|---|---:|---|
| 새 과학 후보 | 13 | 원래 8-key 가설 → 작업 등록 → 작업 시작 → 단계 정책 → 단계 완료 |
| screen/full 단계 | 25 | 13 screen 및 12 full; full 작업 등록은 screen 완료 이후 |
| 후보별 폴드 산출물 | 86 | 실제 학습 72회와 학습 없는 heuristic 출력 14건 |
| 공통 nested base 학습 | 56 | 별도 N001 등록 → fit START → COMPLETE → outer/scope 완료 |
| 연구용 R3 전체 DEV gate refit | 1 | 합성 진단용 별도 등록 → START → 결과 완료 |
| 연구 학습 합계 | 129 | 72 + 56 + 1; 이번 감사의 새 학습은 0회 |

R2·R3·O1이 재사용한 nested 모델은 N001의 별도 등록에 따라 학습됐다. 이후 후보의 정책이 만들어지기 전에 해당 nested scope가 완료됐고, 재사용한 결과 SHA도 일치한다. 나중의 후보 등록이 공통 모델 학습보다 앞서야 한다는 잘못된 조건을 적용하지 않았다.

시각의 한계를 그대로 남겼다. 후보별 폴드 결과 86개에는 별도 완료 UTC가 없어, 단계 전체의 완료 시각만 해당 산출물의 상한으로 사용했다. nested 정책 2개에도 자체 생성 UTC가 없다. 이 정책들의 가설·outer 결과 hash 연결은 확인했지만 별도 정책 생성 시각을 추정하지 않았다. 실제 fit 시작 기록과 nested 56개의 시작·완료 기록은 그대로 검사했다.

이 결과는 애플리케이션이 기록한 UTC와 현재 보존된 SHA의 일관성 검증이다. 외부 시각 공증, 운영체제 시계 검증, 기록되지 않은 활동의 부재 증명은 아니다. 파일의 수정 시각을 등록 시각으로 대체하지 않았으며, 관측값·label·tau·feature·예측 배열을 읽지 않았다. 기존 O1 결과에 없는 역방향 policy hash도 새로 만들어 넣지 않았다.

근거는 `artifacts/next3/maintenance/registration_chronology/v1/RESULTS.json`, `EVENT_CHAINS.json`, `CANDIDATE_FOLD_CHRONOLOGY.csv`, `RECORDED_TIMESTAMP_LIMITATIONS.json`에 있다. 당시 live SESSION은 `SESSION_INPUT.json`에 별도 보존했다. 순서 역전·시각 누락·UTC 형식·숫자형 위장에 대한 helper 테스트 22개도 통과했다.

# Structural Break Real-Time — 첫 세션 결과

상태는 **BLOCKED_AUTH**입니다. 대회 전용 코드와 검증 기반은 구축했지만, 실제 대회 데이터는 인증이 없어 확보하지 못했습니다. 공식 API는 파일 목록을 반환했으나 서명된 다운로드 URL을 제공하지 않았고 파일 요청은 HTTP 403이었습니다. 요청한 10시간의 실제 데이터 연구런은 완료하지 못했습니다. 합성 검증을 대회 성능으로 표시하지 않았습니다.

작업 경로: `C:\Users\minsu\Documents\Structural Break`

## 바로 확인할 결과

* 전체 테스트 **68개 PASS**. 공식 평가기 합성 128사례·256회 비교 최대 오차 **0**.
* 미래 구간 변경·재순회·길이 조회·미리 읽기 차단 검사 PASS. 1,000만 관측값 스트레스 검사 PASS.
* 네 가지 기본 family와 시간척도/누적 증거를 포함한 **57개 streaming feature** 구현. 상태 배열 **988바이트/series**, 관측값 수에 따라 증가하지 않음.
* 공식 EWMA 재현, B0~B7 baseline, 선형/HistGB/LightGBM/XGBoost 후보의 합성 학습·저장·복원 검사 완료.
* 공식 runner 보호 로직을 실제 TCP 소켓으로 실행: 2,880개 예측, 두 번 실행 결과 동일, float32 직접 재생과 동일.
* 전체 공식 runner도 별도 프로세스로 실행: 학습·병렬성 확인·추론·10% 결정성 재검사 성공, 2,400개 예측 인덱스와 공식 형식 검사 PASS.
* 실제 대회 CV, champion 선정, 공식 데이터 기반 ablation은 **미실행**. LOCAL_FINAL_SEAL은 **열지 않음**.

## 요청하신 질문 22개에 대한 답

| 질문 | 현재 답 |
|---|---|
| 1. 대회 데이터 확보? | 아니요. 인증 및 다운로드 서명 부재, HTTP 403. |
| 2. 몇 series? | 실제 확인 불가. 공식 안내의 10,000개는 로컬 감사 결과가 아닙니다. |
| 3. break/no-break 비율? | 실제 측정하지 못했습니다. |
| 4. 데이터의 중요한 특징? | 원본 미확보로 분포·중복·결측·stationarity 감사 미실행. 공식 파일 목록과 용량만 확인. |
| 5. scorer parity? | PASS. 128개 합성 사례, 정렬 후 256회 비교, 최대 오차 0. |
| 6. leakage guard? | PASS. prefix 불변성, API 제한, one-pass, train/validation 분리 검사. |
| 7. split 봉인? | seed 20260908, ID SHA256 80/20 역할 배정, 별도 SHA256 5-fold. 정책 봉인 완료, 실제 ID 배정은 데이터 확보 후. |
| 8. official baseline 점수? | 미측정. 공식 코드 출력 재현만 PASS. |
| 9. 자체 baseline 중 최고? | 판단 불가. B0~B7 구현 완료, 실제 CV 없음. |
| 10. supervised 최고? | 판단 불가. 다섯 모델 엔지니어링 검사만 수행. |
| 11. DEV CV TS-AUC? | 없음. |
| 12. fold별 점수? | 없음. |
| 13. worst fold? | 없음. |
| 14. 가장 유효한 family? | 실제 개발 데이터 없으므로 판단 불가. |
| 15. 쓸모없는 family? | 판단 불가. 결과를 지어내어 제거하지 않음. |
| 16. 추론 지연? | 합성 입력 CPU 중앙값: 수동 2.1µs, logistic 2.5µs, HistGB 3.0µs, LightGBM 3.3µs/point. JIT·초기화·통신 제외. |
| 17. determinism? | PASS. 합성 공식 프로토콜 두 번 실행 최대 차이 0. |
| 18. LOCAL_FINAL_SEAL 개봉? | **NO**. 데이터 자체가 없으며 seal feature/예측을 만들지 않았습니다. |
| 19. reduced local test 사용 횟수? | **0회**. 라벨 파일도 내려받지 못했습니다. |
| 20. 가장 큰 병목? | Crunch 인증. 모델 계산보다 데이터 접근이 우선입니다. |
| 21. 다음 첫 3가지? | 안전한 인증·다운로드 → 원본 감사/라벨 검증/ID split 봉인 → baseline 5-fold CV와 사전 정의 모델·ablation. |
| 22. 제출 가능한 형태? | 함수 인터페이스와 합성 공식 프로토콜은 PASS. 인증된 crunch_tools.test·실제 클라우드·제출은 미실행. |

## 이어서 작업하기

`23_REPRODUCTION_COMMANDS.md`에 환경·테스트·재개 명령이 있습니다. 인증은 사용자가 나중에 `scripts/setup_private.py`를 실행해 숨김 입력으로 처리할 수 있습니다. 토큰을 대화, 소스, README 또는 명령행 인자로 기록하지 마세요.

실제 성능이 없는 CSV에는 `NOT_RUN_AUTH_REQUIRED`와 빈 점수만 들어 있습니다. 기본 `main.py`는 B7 엔지니어링 baseline이며 DEV champion이 아닙니다. `21_LIMITATIONS.md`, `22_NEXT_SESSION_PLAN.md`, `artifacts/validation/FINAL_AUDIT.json`도 확인하십시오.

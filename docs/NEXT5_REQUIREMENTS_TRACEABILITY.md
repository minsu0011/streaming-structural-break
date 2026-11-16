# NEXT5 요구사항과 증거 위치

이 문서는 사용자 사양의 항목을 구현·영수증 위치와 연결하는 지도다. 작업 중에 작성했으므로 파일 경로를 열거한 것 자체가 완료나 통과를 뜻하지 않는다. 최종 실행 상태는 `FINAL_AUDIT_NEXT5.json`, 각 `RESULTS.json`, Desktop 전달 영수증을 따른다. 원문은 `docs/USER_SPECIFICATION_NEXT5.md`이며 SHA는 preflight의 원래 context에 보존했다.

아래의 `BASE`는 `artifacts/next5/`다. `final_v1`, `complete_v1`과 최종 root 보고서는 해당 검증과 시간 기준을 실제 충족한 뒤 생성한다. 조건부 요구가 해당하지 않으면 미실행 이유를 기록하며, 생략한 평가의 수치를 만들지 않는다.

| 사용자 사양 | 구현·판단 | 확인할 증거 |
|---|---|---|
| §0–3, §94: 기존 작업 복원·frozen 역할 | 원래 프로젝트와 BL80/B2/D2/distilled reference, 기존 source/index 상태 보존 | `BASE/preflight/FROZEN_REFERENCES.json`, `PREEXISTING_SOURCE_AND_REPORT_SNAPSHOT.json`, `INITIAL_CONTEXT.json` |
| §4–5, §76: SEAL·reduced·외부 제출 금지 | RP admission을 먼저 검사; Shadow/SEAL의 학습·평가 사용 금지 | `src/next5/guard.py`, `data.py`, `NEXT5_SHADOW_GUARD.json`, 최종 `SEAL_GUARD_NEXT5.json` |
| §6, §78–80: 실제 연구 시간·별도 감사·checkpoint | CPU/log와 실질 작업 신호의 관측 시간; 공백 소급 인정 금지; 36,000초 이후 별도 감사 | `BASE/runtime/ACCOUNTING.json`, `BASE/checkpoints/`, `src/next5/finalization.py`, `BASE/phase_transitions/` |
| §7–11, §46: prospective RP/Shadow와 grouped split | 6,400/1,600 ID 잠금, PRIMARY/ALT_A/ALT_B, 동일 RP 학습 기준 reference | `NEXT5_SHADOW_LOCK.json`, `NEXT5_SPLIT_LOCK.json`, `BASE/references/`, `BASE/data/RESEARCH_POINTER.json` |
| §12–13, §68: 7개 family·후보 한도·사전등록 | 20 screen, 19 full; M2 screen에서 중지; 실제 fit 295회는 별도 count | `HYPOTHESIS_FAMILY_LOCK_NEXT5.json`, `CANDIDATE_BUDGET_NEXT5.json`, `BASE/registrations/`, `BASE/integrity_audits/research_v1/fit_inventory/` |
| §14–18: scale 통합 | S1/S2/S4A/S4B; S3 동일 시간 정보 분해는 후보 fit 아님 | `src/next5/scale.py`, `BASE/diagnostics/S3_SCALE_DECOMPOSITION/`, `SCALE_UNIFICATION_RESULTS.csv` |
| §19–23: persistent/transient | P1/P2/P3; P3는 peak와 ungated slow EWMA를 비교하는 제한된 설계이며 confirmation-only 상태는 미구현; 충격·긴 무변화·late-break를 함께 확인 | `src/next5/persistence.py`, `docs/NEXT5_METHOD_BOUNDARIES.md`, `BASE/synthetic_late_break/`, `PERSISTENCE_RESULTS.csv` |
| §24–27: recent window | W1 에너지 비, W2 H-threshold trimmed energy, W3 고정 window 불일치 | `src/next5/windowed.py`, `WINDOWED_EVIDENCE_RESULTS.csv` |
| §28–30: 누적·sparse-start evidence | E1은 사전등록된 unreset 누적 설계, 최근 age 제한은 별도 E2에만 적용; anytime validity를 선언하지 않음 | `src/next5/mixture.py`, `EPROCESS_RESULTS.csv`, `docs/NEXT5_METHOD_BOUNDARIES.md`, `BASE/null_drift/` |
| §31–34: robust null | R1 H-adaptive tail, R2 H-relative energy PIT, R3 robust self-normalization | `src/next5/robust_null.py`, 각 R registration, `ROBUST_NULL_RESULTS.csv` |
| §35–37: 제한된 mean/sign 연구 | M1 Student-t signed location; M2 sign-run standalone screen 기준 실패로 중지 | `src/next5/location.py`, `BASE/experiments/M2_SIGN_RUN_EVIDENCE/screen/`, `MEAN_SPECIALIST_RESULTS.csv` |
| §38–41: cross-family interaction | 단일 parent 선택 규칙 뒤 X1/X2/X3 각 8개 제품항; X2는 H-calibrated R2 PIT 은행 사용 | `BASE/locks/X_PARENT_SELECTION.json`, `X_SETTINGS_LOCK.json`, `src/next5/cross.py`, `CROSS_FAMILY_RESULTS.csv`, `BASE/feature_usage/` |
| §42–45, §73–74, §91: micro-search·meta·seed 통제 | 고정 learner/W3/seed/가중치; learned stacking meta model 없음 | `configs/next5/RESEARCH_POLICY_LOCK.json`, `NEXT5_RANDOMNESS_LOCK.json`, registration source hashes, 최종 fit inventory |
| §47–51, §79, §84–85: finalist·Shadow·역할 | 모든 full의 평균 Δ<0.0012; 부적격 후보를 Shadow로 구제하지 않음; 빈 finalist를 실제 종료 단계에서 잠금 | `scripts/freeze_next5_empty_finalists.py`, `FINALIST_MANIFEST_NEXT5.json`, `SHADOW_RESULTS_NEXT5.csv`, `DEV_CANDIDATES_NEXT5.json` |
| §48–49: PRIMARY 수치 기준 | 반올림 전 값; median/worst score끼리의 차이와 대응 fold 차이의 통계량을 분리; 원래 gate·CI 대상 보존 | `MODEL_RESULTS_NEXT5.csv`, `PAIRED_BOOTSTRAP_NEXT5.csv`, `docs/NEXT5_METHOD_BOUNDARIES.md`, `LIMITATIONS_NEXT5.md` |
| §52: 선택 후 8k reporting 허용 | 의무가 아닌 선택 사항; 미개봉 Shadow를 유지하므로 새 8k 점수는 산출하지 않음 | `DEV_CANDIDATES_NEXT5.json`, `SHADOW_RESULTS_NEXT5.csv`; 기존 8k 점수는 역사적 reference로만 유지 |
| §53–54: multiplicity·winner gap | 고정 OOF common-draw max-statistic와 winner gap; 선택 적응성을 모두 제거한 보장 아님 | `BASE/selection_uncertainty/`, `docs/NEXT5_SELECTION_LIMITATIONS.md` |
| §55: negative control | 1회 within-time label permutation, constant/time-only score, future traps | `BASE/negative_controls/label_permutation_v1/`, preflight tests |
| §56–57: causal API와 prefix | current point만 받는 state, 최초 None, 길이·suffix materialization traps, reset와 reverse series; cold import cache 경로를 기존 bootstrap으로 먼저 비활성화 | `src/next5/readonly_streaming.py`, 원래 `streaming.py`, `stream_contracts.py`, `BASE/runtime_validation/`, `PREFIX_INVARIANCE_NEXT5.json` |
| §58–62: same-time·fracture·H stress | 모든 full의 26 scopes, ALT/H 고정 검증; support 부족 구간 표시 | `BASE/forensic/`, `BASE/stability/`, `BASE/information/`, `TIME_FRACTURE_NEXT5.csv`, `BREAK_AGE_NEXT5.csv`, `REGIME_RESULTS_NEXT5.csv`, `MECHANISM_RESULTS_NEXT5.csv` |
| §63–65: 고정 synthetic·long-null·impulse | 34 cohorts × 97 후보 artifact; 7개 reference artifact; 1024/4096/16384와 11개 충격 lag | `BASE/synthetic/MENU_LOCK.json`, `FITTED_COVERAGE_LOCK.json`, `synthetic/fitted_summary/complete_v1/`, 최종 세 종류 stress CSV |
| §63–65의 집계 provenance | N0–N3 counterexample와 long_null을 별도 모집단으로 식별; 원래 수치 보존 | `BASE/maintenance/COHORT_PROVENANCE_REPAIR/`, 전수 집계 `COHORT_PROVENANCE_REPAIR.json`, `docs/NEXT5_SYNTHETIC_READING_GUIDE.md` |
| §66–67: latency·공유 계산 | 대표 실제 API 확인 뒤 9개 artifact × 3개 고정 순서 × 10M=270M 업데이트 | `src/next5/runtime.py`, `BASE/runtime_validation/LATENCY_POLICY_LOCK.json`, `BASE/latency/fixed_v1/`, `LATENCY_NEXT5.json` |
| §69–72: adaptive bottleneck loop | ACTIVE 최대 2개, 45–60분 간격 재평가; 유한 메뉴 종료 뒤 고정 검증·원인 분석 | `BASE/bottlenecks/history/`, `BOTTLENECK_HISTORY_NEXT5.csv`, research notes; Case H의 형식 조건과 별도의 반복 DEV 위험을 구분 |
| §69와 속도 측정의 병행 | 격리 속도 측정 중에는 다른 Python을 실행하지 않음; 고정된 PowerShell 도구로 기존 scalar SHA와 완료된 timing block만 검토 | 원래 `scripts/review_next5_bottlenecks_during_timing.ps1`과 별도 `_v2.ps1`, `BASE/bottlenecks/TIMING_REVIEW_POLICY_LOCK.json` 및 `_V2.json`; 실제 실행 내역은 history의 review_engine 필드 |
| §69·§81–83의 native 기록 게시 복구 | 012 history 작성 후 File.Replace 게시 실패를 보존; native fixture로 확인한 별도 v2로 013 게시; timing worker와 점수는 그대로 | `docs/NEXT5_NATIVE_REVIEW_REPAIR.md`, `BASE/maintenance/NATIVE_REVIEW_PUBLICATION_REPAIR/RESOLVED.json`, 최종 history의 publication_status; Python queue quarantine과 구별 |
| §75: 2000 whole-series bootstrap | 원래 2000 draw만 재사용; ALT/H는 ID로 제한; 합성은 원래 생성 seed의 paired draws | `BASE/bootstrap/COUNTS_LOCK.json`, `BASE/stability/`, `BASE/synthetic/EXECUTION_LOCK.json`, 최종 통계 재현 감사 |
| §77: SEAL 권고 | 개봉은 수행하지 않음; 새 PRIMARY/Shadow 확증이 없으므로 NO 권고 | `SEAL_OPEN_RECOMMENDATION_NEXT5.md` |
| §81–83: crash·resource·실패 분류 | 이전 failed/preflight quarantine 보존, 성공 영수증 연결; 과학적 탈락과 실행 오류 분리 | `BASE/failure_resolutions/`, `failure_classification/`, `maintenance/`, checkpoint resource 상태, execution graph audit |
| §81–83의 실제 API import 복구 | 원래 2건 실패·5건 시작 전 대체를 보존; 동일 수치 코드의 7개 새 전체 검증 뒤에만 해결 선언 | `docs/NEXT5_READONLY_IMPORT_REPAIR.md`, `BASE/maintenance/READONLY_IMPORT_REPAIR/`, 새 `API_RO_...` 작업과 최종 실패 해결 영수증 |
| §81–83의 과거 감독 heartbeat 오류 검토 | 12:53 ACTIVITY 읽기 PermissionError 원본을 보존; 기존 latency worker 정상 종료·전체 비최종 413개 상태를 확인; modeling block과 시간은 그대로 | `docs/NEXT5_MONITOR_EXCEPTION_REVIEW.md`, `BASE/maintenance/SUPERVISOR_ACTIVITY_READ_REVIEW/`; 32행 queue/native 분류표 및 개발 명령 오류와 범위를 구별 |
| §86: 별도 final audit | 전체 suite, raw byte/기존 자료 보존, 2470 fold-scopes·95 official score, 3536 synthetic file SHA, portable inference | `BASE/final_audit/PLAN.json`, `BASE/final_audit/final_v1/`, `REPRODUCTION_NEXT5.json`, `FINAL_AUDIT_NEXT5.json` |
| §87: selected new role fresh raw 재현 | 새 NEXT5 role이 없을 때 조건부 요구는 NOT_REQUIRED; 기존 역할 inference 검증과 구별 | `DEV_CANDIDATES_NEXT5.json`, `BASE/retained_artifact_validation/`, `BASE/portable_verification/` |
| §88–89: 최종 파일·47개 한국어 답변·ZIP | 고정 required 목록, 원래 ledger bytes + NEXT5 rows, raw-free source manifest, 정확히 20개 Desktop 요약 파일과 별도 전체 ZIP | `src/next5/handoff_payload.py`, `korean_reports.py`, `scripts/package_next5_handoff.py`, `SOURCE_MANIFEST_NEXT5.json`, `BASE/handoff/DESKTOP_DELIVERY_POINTER.json` |
| §88–89의 보고서 입력 연결 준비 | P late 조건 45행·충격 선형 성분·주요 scale 조건·최상위 X2 감소 취약성을 원래 SHA가 연결된 전수 표에서 읽음; 준비 AST 통과와 실제 완성 입력 검사·최종 감사를 구별 | `BASE/maintenance/KOREAN_LATE_AND_AFFINE_REPORT_PREPARATION_V1.json`, `KOREAN_SCALE_REPORT_PREPARATION_V2.json`, `KOREAN_FUSION_REPORT_PREPARATION_V3.json`; 실제 입력 렌더링 영수증과 최종 발행은 별도 |
| §90, §92–93: 허위 개선 금지·자율 판단 | hidden-test 개선을 RP 최고 점수와 동일시하지 않음; 규칙·봉인 상태를 성능 때문에 완화하지 않음 | `00_GPT_READ_FIRST_NEXT5.md`, `LIMITATIONS_NEXT5.md`, `NEXT_SESSION_PLAN_NEXT5.md` |

## 해석·적용상 차이를 숨기지 않는 항목

`worst_delta`와 최소 대응 fold 차이는 다른 양이다. 원래 계산은 보존하고 엄격한 대응 비교를 추가로 보고한다. X2의 +0.001195650019는 +0.0012 미만이며 두 정의 중 어느 것을 적용해도 평균 기준 미달은 변하지 않는다.

중앙값끼리의 차이와 대응 차이의 중앙값도 분리한다. P2의 median 하한과 S4A의 worst 하한은 정의에 따라 통과 여부가 다르다. 기존 19개 full의 scalar 영수증만으로 확인한 `BASE/integrity_audits/fold_summary_definitions_v1/`가 원래 값·추가 값·기준을 연결한다. 원래 CI의 통계량을 다른 대상으로 바꾸지 않았다.

X2는 직접적인 H-context-only predictor를 새로 만든 것이 아니라 H-relative PIT와 scale 특성을 결합했다. Tree 사용 비중은 훈련 통계이며 H contextual feature의 인과적 효과나 제품항만의 이득을 확증하지 않는다. X parent 선택은 전체 RP 결과에 적응적이어서 nested 후보 설계로 부르지 않는다.

완성된 finite candidate 메뉴가 종료됐다는 판단과, “12개 가설의 best Δ<0.0007”인 형식적 Case H는 별개다. 이번 best Δ는 그 기준보다 커 Case H는 false다. 같은 후보의 seed/weight 미세 변경을 시간 충족 수단으로 사용하지 않고, 미완료 전수 검증과 고정 경로의 원인 분석을 수행한다.

Shadow는 과거 DEV에 포함됐고 H-regime 변환에도 역사적 transductive 요소가 있다. 애플리케이션 loader의 admission-before-read와 guard 검증을 OS sandbox나 Parquet 내부 모든 decode 경로의 증명으로 바꾸어 주장하지 않는다. Raw byte SHA와 관측값·label의 연구 사용은 구별한다.

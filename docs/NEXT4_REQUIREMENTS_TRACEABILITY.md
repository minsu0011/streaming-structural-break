# NEXT4 명세 대조 — 최종 감사 검토

별도 감사의 실제 시작은 2026-09-10 18:18:24.789125 UTC다. 연구 관측 시간은 실제 phase 경계 분할 후 22,531.457043306495초로 확정됐다. 이 문서는 명세와 검증 근거의 대응을 기록하며, 감사·배포 완료 여부의 authority는 이후 작성하는 `FINAL_AUDIT_NEXT4.json`과 외부 delivery receipt다. 미완료 검사 결과를 이 표만으로 PASS로 간주하지 않는다.

| 사용자 명세 | 구현·판단 근거 | 검토 범위와 예외 |
|---|---|---|
| §0–3, §5–6 상태 복원·보호 | `artifacts/next4/preflight/`, `final_audit/final_v1/frozen_boundary/` | 기존 BL80/J102/DISTILLED 파일과 raw·분할·seal·reduced 해시를 확인한다. 기존 소스 908개 및 staged patch 보존, NEXT3 STOPPED_BY_USER를 구별한다. |
| §4, §61 실제 시간 | `runtime/ACCOUNTING.json`, `phase_transitions/RESEARCH_TO_FINAL_AUDIT/`, `final_audit/PLAN.json` | 연구 21,600초 이상 뒤 실제 시각으로 감사 시작. 연구·감사·비관측 시간을 분리하고 감사 45분 이상을 별도로 요구한다. |
| §7–11 유한 메뉴·등록·예산 | `configs/next4/`, `registrations/`, `registration_audit/final_v1/` | 선언 8 family, 점수가 있는 7 family, 16 screen/14 full. 14개 필수 등록 필드 및 실제 DEV 적합 이전 등록을 대조한다. 일부 unit/prototype은 후보 등록 이전이었다. |
| §12–15 dependence·scale | `DEPENDENCE_V2_RESULTS.csv`, `SCALE_ASYMMETRY_RESULTS.csv`, 각 family source와 tests | A 3개, B 2개 고정 버전. Portmanteau-like feature를 보정된 Ljung–Box 검정으로 표현하지 않는다. B2의 원래 CI 실패를 유지한다. |
| §16–18 distribution·likelihood | `DISTRIBUTION_SHIFT_RESULTS.csv`, `SEQUENTIAL_LIKELIHOOD_RESULTS.csv` | C 3개, D 3개 고정 버전. H 고정 bin, discrete working likelihood, bounded age/hazard. D3는 full BOCPD가 아니다. |
| §19–20, §26–28 순위 목적·nested | `RANKING_OBJECTIVE_NEXT4.csv`, `nested_reuse_audit/`, meta source/tests | 실제 outer train 안의 inner OOF만 meta fit에 사용한다. 기존 고유 nested base 56개를 재사용하고 frozen base 재적합은 0이다. E/G2의 16개 새 meta fit을 별도로 센다. |
| §21–25 multi-expert·H-context·state | `families/F_SELECTION.json`, `H_CONTEXT_RESULTS.csv`, `TEMPORAL_STATE_RESULTS_NEXT4.csv` | F1 중복/F2 조건부 종료를 0 fit/0 score로 기록한다. G1 screen 실패를 full로 올리지 않는다. H1은 고정 선형 recurrence와 작은 readout이다. |
| §29–31 prefix·현재값·음성 대조 | `PREFIX_INVARIANCE_NEXT4.json`, `LEAKAGE_AUDIT_NEXT4.json`, `negative_controls/` | Feature별 원래 prefix/oracle 검사와 finalist별 300개 검사를 구별한다. ID rename은 제한된 mock이며 실제 fitted 모델은 별도 20개 역순 reset을 검사한다. Label permutation 1 actual fit도 원장에 포함한다. |
| §32–36 성능·generation·불확실성 | 고정 `RESEARCH_POLICY_LOCK.json`, `bootstrap/`, `joint_uncertainty/`, `winner_gap/` | Mean +0.0013, CI lower>0 등을 완화하지 않는다. 14개 원래 전체 CI는 모두 0을 포함한다. Joint 진단과 bootstrap-best 빈도는 적응적 선택 보정이나 posterior가 아니다. |
| §37–38 ALT/H | `robustness/`, `robustness_forensic/`, `final_audit/final_v1/transfer_statistics/` | B1/B2/D2의 기존 4 partition×5 fold, 전체 및 SCALE 배열을 대조한다. 같은 DEV 재분할이며 독립 새 데이터가 아니다. 원래 OOF의 조건부 H slice와 H 전체 제외 재학습을 구별한다. |
| §39–40 기전·시간 파절 | `MECHANISM_RESULTS_NEXT4.csv`, `TIME_FRACTURE_NEXT4.csv`, `scored_support/` | SCALE 변화 132개는 지원되지만 raw SD 방향 79/53과 MEAN 21/TAIL 86은 각 100개 미만이다. Eventual-break membership과 해당 구간의 실제 scoring-positive support를 구별한다. |
| §41–42 이상치·긴 null | `COUNTEREXAMPLE_RESULTS_NEXT4.csv`, `LONG_NULL_RESULTS_NEXT4.csv`, `memory_diagnostics/` | 13 nomination의 21개 반례, 4개 null, 1024/4096/16384를 완료했다. 실제 D2의 burst 오염 악화와 horizon별 alarm 증가를 보존한다. Threshold는 1024 calibration 기준이며 anytime 보장이 아니다. |
| §43–44 diversity·blend | `COMPLEMENT_RESULTS_NEXT4.csv`, `STANDALONE_DIVERSITY_FOLDS_NEXT4.csv` | Raw correlation, time별 rank correlation, standalone 점수와 고정 blend gain을 구별한다. 사후 1% weight search를 하지 않는다. |
| §45–47 learner·미세조정·importance | 등록된 learner/source, `closed_menu_diagnostics/`, `outlier_attribution/` | 고정 LightGBM과 작은 pairwise/state readout을 평가했다. Feature importance/TreeSHAP는 설명 진단이며 feature 제거의 인과적 효과나 새 tuning authority가 아니다. |
| §48–52 adaptive loop·포화 | `BOTTLENECK_HISTORY_NEXT4.csv`, `BOTTLENECK_BOARD_FINAL_NEXT4.json`, `NEXT4_CLOSED_FAMILY_DECISIONS.md` | 동시 ACTIVE 최대 2개, 유한 메뉴 종료, 포화 HIGH. 45분을 조금 넘긴 두 board 검토 간격은 실제 timestamp와 이유를 그대로 남겼다. |
| §53–54 계산량·상태 | `LATENCY_NEXT4.json`, `latency/B2_DIRECT/`, `latency/D2_DISCRETE_SCALE_MIXTURE_DIRECT/` | 실제 scalar 10M, 동시점 BL80 비교. B2는 5쌍, D2는 연구 2쌍과 별도 감사 1쌍이다. Mutable array bytes는 전체 RSS와 다르다. |
| §55–60 scheduler·복구·cache·중복 | `jobs/`, `failure_resolutions/`, `execution_graph_audit/final_v1/`, `NEXT4_PROTOCOL_SCOPE_NOTES.md` | 실제 worker 실패 2개를 원본 QUARANTINED로 보존하고 별도 성공 evidence로 해결했다. Cold import 수정과 native crash 계측 수리를 모델 개선으로 세지 않는다. 원래 native 원인은 미확정이다. |
| §62–63 finalist·역할 | `DEV_CANDIDATES_NEXT4.json`, `NEXT4_FINAL_TRANSFER_REVIEW.md` | 주 성능 BL80 유지와 별도 D2 SCALE 역할을 구별한다. 역할은 2개이고, 새로운 performance champion/generation은 없다. 실제 release는 별도 감사 PASS를 요구한다. |
| §64–65 seal·public | `SEAL_GUARD_NEXT4.json`, 두 권고 문서 | Seal 4개 미개봉, 이번 reduced/public/cloud 사용 0. 과거 reduced 2/2 기록은 보존한다. 권고와 실제 실행을 구별한다. |
| §66 최종 감사 | `final_audit/PLAN.json`, `SUPPLEMENTAL_PLAN.json`, 각 실제 job receipt | 원래 PRIMARY와 기전, 사전 등록, 세 모델 API, 선택 후보 fresh raw/직접 runtime, 전체 tests, 복사 payload 및 기존 ALT/H 통계를 대조한다. 추가 ALT/H 검증 계획은 원본 계획을 보존한 별도 기록이다. |
| §67 fresh reproduction | `REPRODUCTION_NEXT4.json`, `reproduction/D2_DISCRETE_SCALE_MIXTURE/` | 실제 선택된 D2만 raw DEV→새 feature→동일 fit→inference를 수행한다. B2의 cached diagnostic fit 및 raw-free inference 복사는 fresh raw 학습과 다르다. |
| §68–69 산출물·42답변 | `00_GPT_READ_FIRST_NEXT4.md`, `01_GPT_HANDOFF_NEXT4.md`, `SOURCE_MANIFEST_NEXT4.json` | 명세의 모든 출력, 한국어 42개 답변, 전체 ZIP과 정확히 20개 요약 파일 폴더/ZIP을 대조한다. Raw·credentials·seal labels·실제 point OOF 전체는 제외한다. |
| §70–74 결과·종료·금지사항 | 최종 역할/감사, `LIMITATIONS_NEXT4.md`, `NEXT_SESSION_PLAN_NEXT4.md` | 성능 승격을 강제하지 않는다. D2 SCALE의 제한된 근거와 H·DEPENDENCE·반복 이상치의 손실을 함께 남긴다. 미완료 필수 검사를 미래 계획으로 넘겨 완료 처리하지 않는다. |

사용자 명세 원문은 `docs/USER_SPECIFICATION_NEXT4.md`이고 SHA-256은 `b4bbf541d72a3bea244dcfa6081450f3e1647da0576d8d1a197baa213f056fbe`다. 위의 짧은 경로 중 `runtime/`, `jobs/`, `bootstrap/`, `robustness/` 등 실행 evidence는 `artifacts/next4/` 아래를 뜻한다. 루트 보고서와 실제 완료 receipt를 함께 읽는다.

## 실제 감사 완료 기록 대조

전체 suite는 1,183개 PASS, 실패·오류·skip 0이며 검사 전후 실행 소스가 같았다. 실제 learner fit 166회 모두 완료됐고 원장은 기존 242행의 셀을 보존한 276행이다. 실행 그래프는 143개 작업 중 141개 정상 완료와 원본 실패를 보존한 2개 해결 기록을 확인했다. PRIMARY, 지원 기전, 기존 ALT/H 전체 및 SCALE의 원래 bootstrap 배열을 대조했으며 공식 scorer 차이는 0이다.

fresh raw 재현은 새 feature와 재구성한 target/좌표를 실제 학습 입력으로 전달했다. 기존 cache는 동일성 대조와 고정 좌표 확인에 사용했으며 fresh 학습 matrix를 대신하지 않았다. 추가로 2026-09-10 19:05:16.581573 UTC에 fresh 학습 파일을 직접 로드한 결합 모델 API가 DEV 8,000개, 4,014,405점 전체에서 기존 D2 결합 모델 fingerprint와 정확히 일치했다. 이 추가 검사에는 새 fit·quality score·seal 접근·추론 중 파일 쓰기가 없었다.

추가 결합 API 검사의 실제 계획과 결과는 `artifacts/next4/final_audit/final_v1/FRESH_COMPOSITE_API/`에 있다. 이 harness는 전체 suite 이후 작성한 독립 감사 도구이며, 1,183개 suite 검사에 포함됐다고 세지 않는다. 기존 등록된 모델 실행 소스는 바꾸지 않았고 검사 전후 해시를 다시 대조했다. 원래 감사 계획과 ALT/H 보충 계획도 그대로 보존했다.

raw 재현과 전체 suite 실행 뒤의 보호 파일 마감 검사는 `artifacts/next4/final_audit/final_close/frozen_boundary/RESULTS.json`에 있다. 소스 908개, staged patch, frozen 모델, raw/분할/seal/reduced 해시와 이전 Desktop 20개 파일의 보존을 확인했다. 포함 배열·CSV 범위 및 credential indicator 검토는 `final_audit/final_v1/PAYLOAD_*_REVIEW.json`에 남겼다. 이 검토는 실제 ZIP CRC 검증을 대신하지 않는다.

전달본 사용 범위와 모델 간 의존성은 [전달 파일 사용 안내](NEXT4_HANDOFF_USE.md)를 참고한다. 최종 시간 조건과 보고서 확정은 루트 `FINAL_AUDIT_NEXT4.json`, 실제 ZIP과 20개 폴더의 검증은 외부 delivery receipt를 authority로 삼는다.

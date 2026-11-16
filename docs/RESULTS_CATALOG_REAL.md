# 실데이터 결과 파일의 범위

최종 전달서 `01_GPT_HANDOFF.md`와 동결 후보 `DEV_CHAMPION_CANDIDATES.json`을 먼저 읽는다. 아래 표는 249개 완료 후보를 기록한 모델 연구 체크포인트를 기준으로 한다. CSV에 쓰인 후보·폴드·시간 구간의 키와 수치는 `scripts/audit_result_exports.py`의 31개 검사를 통과했다. 근거는 `artifacts/validation/RESULT_EXPORT_AUDIT_REAL.json`이다.

| 파일 | 행 수 | 의미 |
|---|---:|---|
| `MODEL_RESULTS_REAL.csv` | 249 | 완료 후보 전체 요약. 이름과 달리 통계 baseline도 포함한다. |
| `CV_RESULTS_REAL.csv` | 1,245 | 249개 후보 × 고정 5개 DEV 폴드. 모델 간 비교의 기본 표다. |
| `BASELINE_RESULTS_REAL.csv` | 11 | B0–B7 및 보조 통계 검출기 3개. |
| `FEATURE_ABLATIONS_REAL.csv` | 102 | family ablation 실험. 다른 모델·표현·sampling을 섞어 marginal 차이를 계산하면 안 된다. |
| `TIME_FRACTURE_REAL.csv` | 8,715 | 249 × 5 folds × 7 online-time 구간. |
| `BREAK_AGE_DIAGNOSTICS.csv` | 6,225 | 249 × 5 folds × 5 break-age 구간. 해당 age의 positive를 모든 negative control과 비교한다. |
| `SAMPLING_WEIGHTING_RESULTS.csv` | 38 | sampling/weighting 목적의 명시된 비교 패널. 다른 연구에서 S3를 사용한 모든 실험의 목록은 아니다. |

`MODEL_RESULTS_REAL.csv`의 일반 `latency_us`는 연구 당시 측정이며 구현 및 측정 범위가 후보마다 다를 수 있다. 최종 세 후보의 실제 전체 DEV refit 모델 속도는 `LATENCY_REAL.json`이 권위다. 그 안에서도 `selection_benchmarks`의 CV fold-0 모델과 `candidates`의 최종 배포 모델을 구분한다.

`FEATURE_STANDALONE_RANKING.csv`와 `FEATURE_MARGINAL_CONTRIBUTIONS.csv`는 최종 전달서 생성 시 하나의 동일 설정 패널에서 만든다. 현재 주력인 고정 AR4/AR8 평균의 자체 family ablation은 수행하지 않았으므로 AR4 구성 모델의 패널을 사용한다. 선택한 패널의 정확한 설정은 `artifacts/real/FINAL_ABLATION_PANEL.json`에 있다.

초기 보조 실험 10개는 연구 체크포인트의 후보별 중첩 `time_profile`/`age_profile` 필드가 없다. 그러나 체크포인트의 `fractures`와 두 CSV에는 249개 후보 모두의 폴드별 진단이 있다. 내보내기 감사는 이 전체 행을 직접 비교했고, 나머지 239개 후보의 중첩 요약도 대조했다. 누락된 중첩 필드를 보충하려고 동결 체크포인트를 수정하지 않았다.

`RESEARCH_STATE.json`과 CSV의 `reduced_scored_uses`는 각 연구 체크포인트 당시 상태다. 현재 reduced 사용 횟수는 루트 `REDUCED_TEST_USAGE_LEDGER.json`만 권위로 삼는다. 최종 두 진단의 저장된 scalar는 모델 동결과 최종 감사 뒤 공개하며, 이 때문에 과거 체크포인트의 숫자가 소급 변경되지는 않는다.

`DETERMINISM_REAL.json`은 수치 재현 근거, `DEPLOYMENT_SMOKE_REAL.json`은 실제 배포 API와 공식 runner 근거를 묶는다. 별도 압축본에서 raw부터 수행한 세 후보의 CV 및 전체 DEV 재학습은 `artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json`에 연결한다. CV 성능 검증과 전체 DEV 학습 모델의 단순 수치 일치 검사를 구분해야 한다.

최종 ZIP에는 raw, feature cache, per-row OOF, 가상환경, 인증정보가 없다. `SOURCE_MANIFEST.json`의 바이트 수와 SHA256으로 포함된 코드를 검증한다. 해당 manifest 두 파일은 자기 해시 계산에서 제외된다. `LOCAL_FINAL_SEAL_OPENED = false`이며 독립 seal 성능은 없다.

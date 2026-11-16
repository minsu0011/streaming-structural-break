# Structural Break 실데이터 연구 전달서

DEV 연구 및 최종 검증을 완료했다. 공식 인증으로 확보한 데이터에서 249개 후보의 5-fold 결과를 기록했다. 주력 평균 TS-AUC는 **0.611293**, 공식 EWMA는 **0.514354**다. 이 수치는 반복 선택에 사용한 DEV 성능이며 독립 최종 성능이 아니다.

| 역할 | 후보 | mean | median | worst |
|---|---|---:|---:|---:|
| PRIMARY_DEV_CANDIDATE | M7_cross_order_AR4_LGB_AR8_HGB_equal | 0.611293 | 0.609959 | 0.596411 |
| BACKUP_ROBUST_CANDIDATE | M3_histgb_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae | 0.612049 | 0.606125 | 0.598774 |
| BACKUP_FAST_CANDIDATE | M4_lightgbm_ABCD_S3_122668fc55e8 | 0.607055 | 0.604984 | 0.595586 |

주력 승격 기준은 median +0.001, worst loss ≤0.002, 속도 배율 기준을 유지했다. 평균이 가장 높은 후보와 주력은 다를 수 있다. 안정성 백업과 속도 백업은 `FINAL_CANDIDATE_SELECTION_POLICY.json`의 사전 정책으로 골랐다. 최종 모델은 `artifacts/deployment/final/` 아래에 있다.

| 항목 | 실측 결과 |
|---|---|
| 1. Train series | 10,000 |
| 2. Break / no-break | 4,967 / 5,033 |
| 3. Historical 길이 | 1,000–5,000 |
| 4. Online 길이 | 10–999 |
| 5. 중복 / group | exact 0, normalized exact 0, near merge 0; groups 10,000 |
| 6. DEV / seal | 8,000 / 2,000 series; DEV online 4,014,405 rows |
| 7. Official EWMA DEV mean | 0.514354 |
| 8. Best statistical mean | C2_ar_residual_scale_lr: 0.546456 |
| 9. Best supervised by mean | M4_lightgbm_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae: 0.612535 |
| 10. PRIMARY | M7_cross_order_AR4_LGB_AR8_HGB_equal |
| 11. PRIMARY fold 0–4 | 0.605747 / 0.627883 / 0.609959 / 0.616464 / 0.596411 |
| 12. Mean / median / worst | 0.611293 / 0.609959 / 0.596411 |
| 13. Standalone family 순위 | B 0.602155 > D 0.601932 > C 0.585352 > A 0.538113 |
| 14. 가장 큰 matched marginal | D: A→AD mean +0.066591 |
| 15. 가장 약한 online time | 0-9: 0.510182 |
| 16. Break age 0–4 / 50+ | 0.522997 / 0.631589 |
| 17. Inference μs/point | 5.363 (isolated, continuous stream) |
| 18. 10M 예상 시간 | 53.63s point-only; 174.02s historical 초기화 포함 추정 |
| 19. Reduced 사용 | 2/2; baseline 1회 + frozen primary 1회 |
| 20. LOCAL_FINAL_SEAL_OPENED | false; 성능 평가 0회 |
| 21. Leakage / API 검사 | PASS; 금지 feature 식별자, prefix, single pass, no read-ahead, no inference writes |
| 22. Determinism | PASS; DEV 4,014,405 OOF 재학습 및 별도 전체 DEV refit 예측 동일 |
| 23. Crunch runner | PASS; baseline / primary 각각 100 series, 50,983 predictions; 제출 0회 |
| 24. 병목 3개 | 변화 직후 약한 구별력; 반복 DEV 선택의 낙관 편향과 fold 편차; historical 초기화 및 미검증 cloud IPC 비용 |

## 점수와 진단의 해석

모든 CV 예측은 공식 전송 정밀도인 float32로 평가했다. TS-AUC는 동일 online step 안의 positive/negative pair 수로 가중한 AUC이고, 폴드 보고값은 고정 5개 폴드의 산술 평균이다. Break-age 진단은 해당 age의 positive와 모든 negative control을 비교한다. 이는 임계값 기반 검출 지연이나 오경보율이 아니다.

별도 첫 경보 진단: `M7_cross_order_AR4_LGB_AR8_HGB_equal`에서 TRAIN 무변화 시계열의 상위 5% 최대값으로 fold별 임계값을 정했을 때, held-out 무변화 시계열의 any-alarm 비율은 5.72%, 사전 경보 없이 변화를 검출한 비율은 13.18%, 검출 사례에 한정한 지연 중앙값은 142.5 관측치였다. 미검출·관측 종료 사례는 지연 중앙값에서 제외되므로 검출률과 함께 해석한다. 이는 사후 DEV 진단이며 competition 출력이나 모델 선택에 임계값을 적용하지 않았다. `artifacts/real/first_alarm_diagnostics/RESULTS.json`에 1%/5% 결과와 정확한 범위가 있다.

Historical 구조별 사후 진단에서는 AR8 제곱 잔차의 lag-1 상관이 0.05 이상 0.2 미만인 1,955개 DEV 시계열의 주력 TS-AUC가 0.587927였다. 구간은 historical 값만으로 정했으며 모든 결과는 `artifacts/real/historical_cohorts`에 보존했다. 구간 간 pair는 점수에서 제외되므로 전체 점수의 가산 분해가 아니다. 400회 group bootstrap 구간은 다중 비교·모델 선택 편향을 보정하지 않은 조건부 진단이다.

Standalone 및 marginal 표는 같은 모델·표현·feature 설정·sampling에서 비교한 별도 패널이다. 고정 AR4/AR8 평균이 주력이면 AR4 구성 모델의 패널이며, 평균 모델 자체의 family ablation은 수행하지 않았다. 정확한 설정은 `artifacts/real/FINAL_ABLATION_PANEL.json`을 확인한다. feature importance만으로 family 효용을 단정하지 않았다. 선정된 모델의 각 표현은 57개 특징을 계산하고, 두 표현의 평균은 총 114개를 계산한다.

최종 동결 뒤 저장된 reduced 진단을 공개했다: baseline **0.517007**, primary **0.554369**. 이 점수로 모델·feature·hyperparameter를 변경하지 않았다. LOCAL_FINAL_SEAL_OPENED = false.

공개된 scalar 보고서의 원래 바이트는 `artifacts/real/final_reduced_scalars`에 보존하며 quota ledger의 SHA256과 일치한다. 이 파일은 target 행을 포함하지 않으며 새 폴더에서 전달 문서를 재구성할 때 추가 target 조회를 대신한다.

사전 전체 기술 통계 감사에서는 모든 train series의 길이와 break/tau 메타데이터를 요약했다. split 동결 후 모델 학습·CV·진단은 DEV에 한정했으며 seal 성능을 계산하지 않았다. 원시 파일 SHA256 검사는 파일 전체 바이트를 읽는 작업이다.

## 재현과 전달 범위

`23_REPRODUCTION_COMMANDS.md`, `SOURCE_MANIFEST.json`, `artifacts/validation/FINAL_AUDIT_REAL.json`을 먼저 확인한다. 코드·설정·테스트·선정 모델·작은 결과를 ZIP에 포함하고, raw/cache/대용량 OOF/가상환경/인증정보는 제외한다. 원본 데이터가 필요한 CV 재현은 manifest와 일치하는 로컬 `data/raw`에서 실행해야 한다.

수치의 전체 구조는 `artifacts/real/figures/DEV_RESULTS_OVERVIEW.pdf`와 같은 이름의 PNG/SVG에 요약했다. 그래프 해석은 `docs/FIGURE_GUIDE_REAL.md`, CSV별 행 수와 권위는 `docs/RESULTS_CATALOG_REAL.md`를 확인한다. 31개 내보내기 검사는 모든 완료 후보·폴드·시간 구간이 동결 체크포인트와 일치함을 확인했다.

추가 DEV 통신 검증: 원래 fold 모델 5개를 실제 `infer()`와 변경 없는 공식 소켓/보호 코드로 실행하여 4,014,405개 예측이 OOF와 정확히 일치했다. 역순 시계열 검증도 통과했다. Windows loopback의 별도 Python consumer 프로세스 방식에서 초기화·전송·예측·수집을 포함한 비용은 71.054 μs/point였다. 이는 로컬 통신 실측이며 cloud IPC·병렬화 SLA 검증은 아니다. reduced 입력과 target은 사용하지 않았다. 위 전체 시간에는 fold별로 새로 만든 5개 프로세스의 시작/JIT 준비 비용이 포함된다(각 8.08–8.43초). 준비 신호 이후의 초기화·통신·예측 비용은 60.349 μs/point였다. 최종 전체 DEV refit 모델도 별도로 4,014,405개 예측의 배치/스트리밍 일치와 새 consumer 프로세스 1개의 공식 소켓 일치를 통과했다. 최종 모델의 소켓 전체 실행은 250.56초, 그중 준비 완료까지 8.18초였다. 이 검사는 같은 모델의 수치 일치를 확인하며 학습 데이터 성능 점수를 계산하지 않는다.

10M point-only 측정은 실제 DEV 값의 작은 버퍼를 반복한 하나의 긴 스트림이다. 초기화 포함 174.02초 추정은 최종 모델의 128개 DEV historical 초기화와 관측 online 길이를 사용한다. 별도로 원래 CV 모델을 DEV 8,000개 실제 길이로 3회 실행한 검사는 매번 4,014,405개 OOF 예측과 일치했고, 초기화 포함 중앙 실행은 18.032 μs/point였다. 그 실행의 초기화 비중은 68.79%다. 이 계산 구간에는 최초 JIT·모델 로딩·입력 파일 I/O·소켓 통신이 포함되지 않는다.

최종 동결 후 소스 ZIP `SRC_REAL_SNAPSHOT_20260908T145235Z.zip`(SHA256 `4faedf71168e83f4d2cad59d4e40fcb66c43ac960a339c9dfe05606c2d2dd4c8`)을 별도 경로·가상환경에 풀어 최종 세 역할 모두 원본 raw부터 재현했다. 각 후보의 5개 cache 파일은 원래 바이트와 일치했고, 독립 5-fold 재학습과 전체 DEV 배포 모델 재학습에서 각각 4,014,405개 예측 차이가 0이었다. CV 모델 바이트 일치 여부는 PRIMARY_DEV_CANDIDATE: False; BACKUP_ROBUST_CANDIDATE: True; BACKUP_FAST_CANDIDATE: False이며 전체 DEV 모델 바이트 일치 여부는 PRIMARY_DEV_CANDIDATE: True; BACKUP_ROBUST_CANDIDATE: True; BACKUP_FAST_CANDIDATE: True다. 예측 동일성과 직렬화 파일 동일성은 구분한다. 원본 OOF 배열과 cache는 ZIP에 없었으며, 상세 증거는 `artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json`과 그 안의 보고서 경로에 있다. 이 raw 재현 당시 모델·학습·추론 소스가 현재와 일치하는지도 최종 감사에서 검사한다.

직렬화 차이는 추가로 검사했다. 선정 후보의 15개 CV 모델과 3개 최종 모델, 총 18쌍에서 실제 compact 추론에 쓰는 모든 트리 분기·값·사용 열·표현 설정의 정확한 상태가 일치했다. LightGBM 본문 및 HistGB 노드·bias 진단은 `artifacts/real/PREDICTIVE_STATE_REPRODUCTION_REAL.json`에 따로 기록했다. 이는 전체 native 추정기 객체나 다른 플랫폼의 실행까지 동일하다는 주장은 아니다.

실행 예산: 2026-09-08T06:49:52+00:00 시작, 2026-09-08T16:04:52+00:00 새 실험 마감, 2026-09-08T16:49:52+00:00 전체 마감. 최종 문서 생성 시각 2026-09-08T16:06:46.607006+00:00. 테스트 172개 통과. 별도 소스 복사와 가상환경 검증은 `artifacts/validation/REPRODUCTION_REPORT.json`에 기록하며, 최종 ZIP의 실행 소스 일치 여부는 `SOURCE_MANIFEST.json`이 권위다.

이전 인증 차단 단계의 문서는 역사 기록이다. 현재 상태와 실측 결과는 이 전달서와 이름에 REAL이 있는 보고서를 우선한다. 다음 작업은 `NEXT_SESSION_PLAN.md`, 한계는 `LIMITATIONS.md`에 적었다.

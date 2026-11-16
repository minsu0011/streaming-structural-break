# NEXT4 전달 파일 사용 안내

20개 파일 폴더는 GPT에 과정과 결과를 전달하기 위한 요약본이다. 소스, 모델, 상세 검증 기록은 `STRUCTURAL_BREAK_GPT_HANDOFF_NEXT4_<timestamp>.zip` 전체본에 있다. 전체본을 펼칠 때 내부 상대 경로를 유지한다.

먼저 루트의 `00_GPT_READ_FIRST_NEXT4.md`, `01_GPT_HANDOFF_NEXT4.md`, `DEV_CANDIDATES_NEXT4.json`, `FINAL_AUDIT_NEXT4.json`을 읽는다. 최종 감사와 ZIP 검증은 서로 다른 기록이다. ZIP의 실제 CRC 및 각 파일 SHA 검증 결과는 원래 프로젝트의 `artifacts/next4/handoff/DESKTOP_DELIVERY_POINTER.json`이 가리키는 외부 receipt에 남는다. ZIP 자신을 포함한 재귀 해시는 만들지 않는다.

| 역할 | 추론 entry | model directory |
|---|---|---|
| 기존 주 성능 BL80 | `next2_shared_performance_main.py` | `artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1` |
| 별도 SCALE 강건성 D2 | `next4_robustness_main.py` | `artifacts/next4/refits/D2_DISCRETE_SCALE_MIXTURE/nominated_all_dev_v1` |

D2는 원래 primary 평균 0.6235961690191665이지만 전체 차이의 95% CI가 0을 포함한다. 주 성능 BL80의 대체 모델이나 새 generation으로 해석하지 않는다. 지원되는 SCALE 변화 132개에서의 이득과 실제 all-DEV artifact의 반복 이상치 악화를 함께 읽는다. `docs/NEXT4_FINAL_TRANSFER_REVIEW.md`가 채택 범위와 손실을 설명한다. B2의 탐색 최고 점수도 승격된 역할이 아니다.

D2 entry는 자신의 expert 파일 외에 위 BL80 directory를 함께 읽는다. `model.joblib` 하나만 이동하면 충분하지 않다. 선택 JSON의 model SHA는 다음과 같다.

- BL80: `1cdfcde2c715db69ab2bf91b8e9803ac1dd24a5e146bea8f94cbc7e16e7b1d90`
- D2: `ce04a393a127506291c1d7ee7bff75a11781dac435fc9dee89684a69c754568c`

API는 `infer(datasets, model_directory_path)`다. `datasets`가 제공하는 각 항목은 `(historical_array, online_iterator)`이며 첫 출력은 `None`, 이후 현재 관측마다 scalar 하나다. 초기 historical 구간과 현재까지의 online 관측만 전달한다. 시계열 ID, tau, 전체 online 길이, 미래 suffix, 사후 기전 label은 모델 입력이 아니다.

원본 데이터 없이 고정 합성 입력으로 cold inference를 확인하는 기존 명령은 다음과 같다. 먼저 전체 ZIP을 펼친 프로젝트 루트로 이동하고, 정확한 pin이 준비된 환경의 Python을 사용한다. `--output` 또는 `--cold-worker-output`에는 새 결과 JSON 경로를 지정한다.

```text
python -B scripts/verify_next4_cold_api.py --worker BL80 --output <new_BL80_result.json>
python -B scripts/verify_next4_b2_readonly.py --worker-output <new_B2_result.json>
python -B scripts/verify_next4_candidate_api.py --model D2_DISCRETE_SCALE_MIXTURE --cold-worker-output <new_D2_result.json>
```

이 명령의 검증 범위는 고정 합성 시계열 20개, 2,170개 출력과 cold import/추론 중 쓰기 시도다. 실제 DEV 8,000개에 대한 검증과 혼동하지 않는다. 모델을 검증한 뒤 결과 JSON을 기록하는 harness의 쓰기는 추론 중 모델의 쓰기와 구별한다. 기존 고정 fingerprint는 다음과 같다.

- B2: `36741348a661af08613a520a0d8c7b8e82d4618f1d5be9cf9a48b5a56cd7df84`
- D2: `26c83904f8e9a8482b59126d5ed61ec5b860ac829a770016045e86d7b751906a`
- BL80: 포함된 `artifacts/next4/cold_api/BL80_1.json`의 `prediction_sha256`.

복사본에서 새 Python 프로세스로 실행한 실제 결과는 `artifacts/next4/portable_verification/final_v1/RESULTS.json`에 기록한다. 이 검증은 기존에 pin을 확인한 interpreter를 재사용한다. 다른 PC에서 새 환경을 설치해 검증했다는 뜻은 아니다. 실행 pin은 `requirements-lock.txt`, 그림 pin은 `requirements-next2-figures-lock.txt`에 있다. 환경 자체는 ZIP에 포함하지 않는다.

학습 재현에는 별도 raw 데이터와 cache 재생성이 필요하다. 루트의 `RAW_DATA_REQUIRED_NEXT4.json`에서 원본 파일의 목적 경로 및 SHA를 확인한다. ZIP에는 raw 관측/label, 실제 point OOF 전체, feature cache가 없다. 이미 완료한 D2 fresh raw 재현의 결과는 `REPRODUCTION_NEXT4.json`과 `artifacts/next4/reproduction/D2_DISCRETE_SCALE_MIXTURE/`에 있다. 재현용 joblib는 metadata 때문에 배포 joblib와 파일 SHA가 다를 수 있으며, 검증한 동등성은 새 feature·학습 좌표·4,014,405개 예측의 정확한 일치다.

완료된 immutable 출력 directory에 기존 연구 scheduler를 그대로 재시작하지 않는다. 포함된 이전 단계의 중단/실패 기록은 추적 근거다. NEXT3의 사용자 중단은 유지되며, 새로운 seal/reduced/public/cloud 실행은 이 인계의 후속 자동 작업이 아니다.

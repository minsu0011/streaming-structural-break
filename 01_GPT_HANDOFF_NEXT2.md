# Structural Break NEXT2 연구 인계

이전 프로젝트를 이어 진행한 DEV 전용 연구다. 최종 역할은 `DEV_CANDIDATES_NEXT2.json`에 동결했다. 기존 formal PRIMARY J102의 원본 파일을 교체하지 않았으며, 새 80/20 모델은 성능 후보 역할로 제공한다. generation-level 교체는 승인되지 않았다.

| 역할 | 과학 모델 | 실행 구현 | mean DEV TS-AUC | canonical 파일 |
| --- | --- | --- | ---: | --- |
| DISTILLED | J102_SERIAL_VARIANCE_512 | tree_used_v1 | 0.62163175031248 | artifacts/next2/canonical_deployment/DISTILLED_tree_used_v1/model.sbnext2 |
| PERFORMANCE | BL2_C205_DEPENDENCE_80 | shared_blend_v1 | 0.6232684128173095 | artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2 |

공통 학습 규칙은 고정 100-tree LightGBM, W3, 기존 group 5-fold다. 성능 모델은 100-tree 두 개의 80/20 산술 혼합이다. synthetic 점수로 weight/threshold를 고르지 않았다. 실제 두 entry의 train/infer를 새 환경에서 원본 DEV로 재실행했으며, model bytes·학습 feature/ID/target hash·전체 4,014,405개 예측 fingerprint를 맞췄다. 학습 시 모든 seal ID는 값 접근을 거부하는 opaque object로 주었고 2,000개를 건너뛰었다. 재학습의 all-DEV 예측은 동일성 확인이며 in-sample 품질 점수가 아니다.

`model.sbnext2`는 JSON+zlib의 검증 가능한 inference 전용 형식이다. 기존 joblib은 pickle alias metadata 때문에 byte 동일성이 실패한 기록이 있어, predictive arrays/Booster 내용이 같은 것을 먼저 확인한 뒤 canonical 형식으로 해결했다. 기존 실패 결과를 삭제하지 않았다. 모델 파일 크기는 distilled 35,180 bytes, performance 68,999 bytes다. 프로세스/JIT 메모리 크기와 혼동하지 않는다.

연구 결론은 세 가지다. 첫째, 기존 0.6216대 상승에는 ARCH/innovation energy/forecast와 long-age Bayes가 기여하며 serial correction 자체의 이득은 거의 0이다. 다만 GLR/serial 제거 후보는 사전 비열등성 확률 조건을 통과하지 못했다. 둘째, 학습 예측을 바꾸지 않고 J102 private state를 약 59.80% 줄이고 H 초기화를 약 40.89% 단축했다. 셋째, 낮은 상관의 조건부 dependence 신호 C205가 실제 성능 혼합에 기여했다. 성공 조건 B와 D를 충족했지만 전반적 robustness 개선이나 generation jump는 아니다.

합성 연구는 18개 원래 시나리오와 4개 long-null 배경을 고정했다. 원래 all-DEV J/Y/B/C/blend, 원래 fold-0의 component 28개, 기존 5개 CV fit의 민감도, 교차 배경 scale, 늦은 변화/첫 경보, 단일 이상치 잔류, 내부 증거와 predictive TreeSHAP을 구분했다. 표의 `fitted_fold`, `artifact_kind`, `record_type`, `diagnostic_suite`를 함께 읽는다. H heldout을 다시 학습한 점수와 원래 OOF를 H별로 자른 점수도 구별한다. runtime 비용의 28개 component 진단은 CV0 모델이며, 최종 all-DEV runtime은 별도의 12개 측정 표를 기준으로 한다.

ARCH/tail scale-down 실패를 단순한 신호 부재로 설명하지 않는다. H calibration에서 extreme energy가 표준편차에 큰 영향을 주고, 음의 current-energy 범위가 좁아졌다. 그러나 paired current-energy와 forecast는 모두 감소에 맞는 signed 반응을 유지했다. 뒤의 evidence/memory/트리 결합이 다음 연구 대상이다. limiting ARCH innovation의 fourth moment 계산은 고정 DGP에 대한 수학적 성질이며, 유한 simulation sample이나 real competition DGP의 무한 모멘트를 주장하지 않는다. `docs/NEXT2_SCALE_DOWN_MECHANISTIC_SYNTHESIS.md`에 근거와 한계를 연결했다.

rank를 보존하는 odds age-decay는 TS-AUC를 바꾸지 않지만, 49k의 늦은 변화에서 clean detection을 없앨 수 있었다. 높은 TS-AUC, finite-horizon FPR, 유용한 first alarm은 다른 평가다. 누적 GLR memory는 null에서 자라며, 이상치 하나의 영향은 수백 포인트 남는다. 제거 ablation은 일부 반례를 완화하지만 real DEV 조건을 충족하지 못해 최종 모델에 반영하지 않았다.

## 재현 명령

Windows PowerShell에서 압축 해제한 디렉터리로 이동한 뒤:

```powershell
py -3.13 -m venv .review-venv
.\.review-venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.review-venv\Scripts\python.exe -m pip check
.\.review-venv\Scripts\python.exe scripts/replay_next2_portable.py --output portable_replay.json
.\.review-venv\Scripts\python.exe scripts/replay_next2_report_bootstrap.py
```

첫 replay는 raw competition 데이터 없이 18개 원래 synthetic scenario의 64개 role/control cell, 524,288개 public-API 출력을 기존 결과와 비교한다. 두 번째는 최고 모델의 24개 scope paired bootstrap 수치를 재계산한다. 압축본에는 필요한 두 모델, source/default config, 고정 generator, 작은 bootstrap 통계 배열이 포함된다. 원래 프로젝트 전체 테스트 536개 중 raw/old cache에 의존하는 검증은 원본 workspace가 필요하므로, ZIP만으로 모든 과거 테스트가 실행된다고 주장하지 않는다.

그림을 다시 만들려면 별도 figure 환경에 `requirements-next2-figures-lock.txt`의 13개 고정 패키지를 설치한다. 실제 그림 환경의 버전 목록과 일치하며, main 학습 환경의 63개 lock을 바꾸지 않는다. `scripts/plot_next2_research.py`, `scripts/plot_next2_internal_evidence.py`, `scripts/plot_next2_outlier_persistence.py`, `scripts/plot_next2_scale_channels.py`가 포함된 요약 CSV에서 PNG/PDF를 생성한다.

원본 재학습 검증은 `scripts/prepare_next2_independent_reproduction.py`, `scripts/complete_next2_reproduction_dependencies.py`, `scripts/run_next2_independent_role.py` 및 독립 child script에 기록했다. 기존 결과를 덮어쓰지 않도록 실행을 거부하는 시간/경로 guard가 있다. 그대로 재실행하려면 새 audit session/새 출력 경로가 필요하다. 원본 raw 파일의 필요 SHA는 `RAW_DATA_REQUIRED_NEXT2.json`, 실제 새 환경·module 경로·fit·예측 확인은 `REPRODUCTION_NEXT2.json`과 `artifacts/next2/independent_reproduction/`에 있다. 학습 iterator API로 재현할 때도 frozen DEV boundary를 유지해야 한다.

## 읽을 파일

`00_GPT_READ_FIRST_NEXT2.md`는 요청한 35개 답이다. `MODEL_RESULTS_NEXT2.csv`/`CV_RESULTS_NEXT2.csv`/`PAIRED_BOOTSTRAP_NEXT2.csv`는 전체 과학 결과, `RUNTIME_PRUNING_RESULTS.csv`/`LATENCY_NEXT2.json`/`STATE_MEMORY_NEXT2.json`은 실행 비용, 나머지 component/robustness CSV는 세부 근거다. `docs/NEXT2_*PROTOCOL.md`와 각 `POLICY.json`은 시점·메뉴·seed를 보존한다. 원래 experiment receipt의 `*_PENDING`은 당시 학습 단계 상태일 수 있으므로 최종 역할은 `DEV_CANDIDATES_NEXT2.json`과 최종 표의 disposition을 기준으로 읽는다.

`FINAL_AUDIT_NEXT2.json`은 별도 감사 단계의 결과, `SOURCE_MANIFEST_NEXT2.json`은 ZIP entry별 SHA/크기다. 모든 ZIP entry SHA와 CRC를 다시 읽어 확인한 외부 기록은 원본 프로젝트 `artifacts/next2/handoff/DESKTOP_ZIP_VERIFICATION_NEXT2.json`에 남긴다. ZIP 자기 hash를 ZIP 내부에 재귀적으로 넣지 않는다. 같은 agent가 연구 후 검증 단계를 분리했으며 독립된 인간/다른 agent의 심사를 받았다고 주장하지 않는다.

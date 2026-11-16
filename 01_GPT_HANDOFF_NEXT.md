# Structural Break NEXT — GPT 전달 문서

| 역할 | 후보 | 평균 | 중앙값 | 최저 fold |
| --- | --- | --- | --- | --- |
| PRIMARY_DEV_CANDIDATE | J102_SERIAL_VARIANCE_512 | 0.62163175031248 | 0.6214373863953904 | 0.6101464788170321 |
| ROBUST_DEV_CANDIDATE | B104_LONG_VARIANCE_NO_GLR_MEMORY | 0.6214890347069302 | 0.6201193885482584 | 0.6114405720853687 |
| FAST_DEV_CANDIDATE | F101_VARIANCE_TREES50 | 0.6171824646565767 | 0.6178571134418307 | 0.6072673445305061 |

먼저 00_GPT_READ_FIRST_NEXT.md의 26개 항목을 읽는다. 모든 점수는 고정된 DEV 8,000개 series의 원본 5-fold TS-AUC 또는 별도로 표시한 stress fit에서 나왔다. seal 2,000개는 관측/라벨/예측/평가에 사용하지 않았다.

ZIP 자체 검증: `python scripts/verify_next_handoff_manifest.py "ZIP의 전체 경로"`. Python 표준 라이브러리만 사용하고 파일을 풀거나 모델을 실행하지 않는다. 출력된 archive_sha256을 별도 전달된 ZIP 검증 영수증과 비교한다. 내부 manifest 일치는 발신자 인증을 뜻하지 않는다.

최종 구현 검토와 serial Bayes 수식의 독립 적분 확인은 docs/NEXT_FINAL_IMPLEMENTATION_REVIEW.md 및 tests/test_next_serial_bayes_integral.py에 있다.

원본 실험은 artifacts/next/experiments, 20개 추가 split fit은 robustness/candidates, 24개 절단의 2,000회 paired bootstrap은 forensic, 독립 원시 재현은 independent_raw_validation, 실제 모델의 API/속도 연결 감사는 final_engineering, 생산 train 재현은 final_training에 있다. 각 RESULTS.json 및 SOURCE_MANIFEST_NEXT.json에 SHA256이 있다.

MODEL_RESULTS_NEXT.csv와 EXPERIMENT_LEDGER.csv의 final_evidence_status가 최종 종합 상태다. original_stage_status는 당시 screen/full 단계의 불변 기록으로 일부 ROBUSTNESS_PENDING이 남는다. engineering 미완료의 미선정 후보는 임의로 PASS 처리하지 않는다. SCREEN_FAIL 후보의 품질 숫자를 전체 5-fold 평균과 비교하지 않는다.

모델 사용: DEV_CANDIDATES_NEXT.json의 delivery_model_directory와 entrypoint를 함께 사용한다. variance는 next_readonly_main.py, log-energy는 next_registered_main.py, serial-variance는 next_serial_main.py다. 기존 main.py는 기존 REAL 모델용으로 보존되어 있다. infer는 먼저 None을 yield하고 각 O 관측마다 Python float 하나를 내보낸다.

역할별 train specification은 configs/next_roles/<ROLE>.json이다. 각 family entry의 기본 train은 그 family에서 우선순위가 가장 높은 선택 역할을 학습한다. 같은 family의 다른 역할은 해당 training 모듈 train_frozen_candidate에 명시적 spec을 넘긴다. 검증 명령은 scripts/train_next_final_roles.py에 기록되어 있다. 이미 만든 최종 디렉터리에 재학습하면 fail-closed로 거절한다.

핵심 attribution은 artifacts/next/fixed_ablation_contrasts/research_v1, 학습량 곡선은 learning_curve, H forward 진단은 historical_holdout, 제한된 nested 선택 진단은 nested_menu_cv, alarm 진단은 alarm_calibration, 합성 counterexample은 synthetic_variance_stress, 장기 null 실험은 long_null_stress를 확인한다. 장기 오경보·방향별 반례·앙상블 추가 기여를 함께 해석한 문서는 artifacts/next/research_synthesis/ADDITIONAL_DIAGNOSTICS.md다.

Zip에는 원시 데이터와 환경, 인증정보, 모든 OOF point 파일을 포함하지 않는다. canonical raw SHA는 REPRODUCTION_NEXT.json과 src/next/data.py에 있다. 모델 환경은 requirements-lock.txt, 별도 그림 환경은 requirements-next-figures-lock.txt에 고정했다. 그림은 작은 CSV 보고서에서 생성하며 원시 데이터가 필요하지 않다. 분석은 잠근 환경에서 재현했으며 독립 seal 일반화 성능은 아직 확인하지 않았다. 상세 한계와 다음 세션 경계는 LIMITATIONS_NEXT.md 및 SEAL_OPEN_RECOMMENDATION.md에 있다.

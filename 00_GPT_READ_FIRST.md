# GPT에게 먼저 전달할 내용

`01_GPT_HANDOFF.md`부터 읽는다. 실데이터 DEV 연구와 최종 검증을 완료했다. 주력 `M7_cross_order_AR4_LGB_AR8_HGB_equal`의 5-fold mean / median / worst는 **0.611293 / 0.609959 / 0.596411**다.

Train 10,000 series, DEV 8,000 / seal 2,000. **LOCAL_FINAL_SEAL_OPENED = false**. Reduced scored uses **2/2**를 소진했으므로 추가 조회·모델 선택에 사용하지 않는다. 최종 후보 3역할과 원본 모델은 `DEV_CHAMPION_CANDIDATES.json` 및 `artifacts/deployment/final/`에 있다.

172 tests PASS. 독립 5-fold 및 전체 DEV 재학습, 실제 streaming parity와 인증된 공식 runner 검사 PASS. DEV 반복 선택 편향, 약한 초기 검출, cloud 실행 미검증을 한계로 유지한다. 이 ZIP은 이전 `BLOCKED_AUTH` / 20260908_080300 ZIP을 대체한다. 원시 데이터·캐시·인증정보는 ZIP에 넣지 않았다.

핵심 파일: `01_GPT_HANDOFF.md`, `DEV_CHAMPION_CANDIDATES.json`, `CV_RESULTS_REAL.csv`, `FEATURE_STANDALONE_RANKING.csv`, `LATENCY_REAL.json`, `DETERMINISM_REAL.json`, `DEPLOYMENT_SMOKE_REAL.json`, `SOURCE_MANIFEST.json`.

요약 그림: `artifacts/real/figures/DEV_RESULTS_OVERVIEW.pdf` 또는 PNG. 데이터 표 안내: `docs/RESULTS_CATALOG_REAL.md`.

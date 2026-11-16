# NEXT4 최종 감사 계약 — 실행 전 계획

이 문서는 완료 증명이 아니다. 실제 완료 시각·검사 결과·파일 해시는 별도의 `FINAL_AUDIT_NEXT4.json`에 기록한다.

최종 감사는 `artifacts/next4/runtime/ACCOUNTING.json`의 실제 productive research가 21,600초 이상이고, 필요한 연구·검증 작업이 끝난 뒤 시작한다. `SESSION.json`에 실제 시작 시각을 기록하며 소급하지 않는다. 그 이후 별도 45~60분 동안 검증·검토·문서화한다. CPU 작업 또는 구체적인 감사 활동이 관측되지 않은 공백을 완료 시간으로 바꾸지 않는다.

승격 결정은 처음 고정한 policy와 원래 primary OOF를 따른다. B2의 original primary CI는 0을 포함하므로 ALT 개선, 시계열 삭제 민감도, 합성 점수, 추가 bootstrap 해석 또는 빠른 runtime으로 그 실패를 덮지 않는다. Robustness 역할은 실제 supported SCALE 개선과 나머지 고정 조건을 통과한 경우에만 고려하며, 최대 한 후보를 선택한다. 새 후보가 없으면 frozen BL80을 유지한다.

감사에서 다음 항목을 확인한다.

1. Raw X/y 전체 바이트 SHA, 원래 grouped split, ALT/H lock, seal membership와 unopened 상태, reduced ledger를 시작 snapshot과 대조한다. Raw 바이트 hashing과 DEV predicate 밖의 값 관측을 구별한다.
2. 기존 소스·설정 908개와 사용자의 기존 staged 변경이 보존됐는지 확인한다. NEXT3의 사용자 중단 상태 및 바탕화면의 기존 20개 파일·ZIP도 보존한다.
3. 선택 근거를 원래 fold 좌표와 official TS-AUC로 확인한다. Paired bootstrap의 seed, multiplicity, 2,000회, 참조 배열, 집계 방식이 동일한지 점검한다. 분석을 새로운 학습 불확실성이나 적응적 선택 보정으로 표현하지 않는다.
4. Guard/scorer, 새 모델에 관련된 검사와 가능한 전체 pytest suite를 실행한다. 개발 중 실패한 unit fixture·정밀도 수정은 원본과 실패 XML을 남긴 상태로 설명한다. 이전 NEXT3의 중단된 전체 검사를 완료로 바꾸지 않는다.
5. 선택하거나 중요 finalist로 보고한 모델의 prefix, 현재 점당 하나의 출력, 첫 None, 순서 재설정, offline/stream, save/load, deterministic fingerprint와 직접 측정한 비용을 검토한다. 준비된 모델의 추론 검사와 새 프로세스의 cold import 검사를 구별한다. Cold 검사에서 CPython bytecode 쓰기를 끄는 조건을 명시한다.
6. 실제 새 NEXT4 후보를 승격한다면 fresh output에서 raw DEV → 새 feature → 동일 learner fit → inference 재현을 수행한다. 캐시를 사용한 B2 진단용 all-DEV fit은 이 재현으로 세지 않는다. 새 승격이 없으면 frozen 모델을 다시 적합하지 않는다.
7. 학습 수는 개별 fit 시작·완료 receipt와 모델 파일에서 집계한다. Parent summary, 같은 inner OOF의 반복 사용, bootstrap과 threshold calibration을 새 learner fit으로 중복 집계하지 않는다. 실제 DEV 적합, permuted-label control, 합성 unit fixture를 구별한다.
8. 명세 §68의 모든 출력과 한국어 42개 답변을 검토한다. Screen/full, standalone/learned combination, 조건부 H slice/실제 H holdout, eventual-break membership/실제 scoring-positive support, fold-0/all-DEV 합성 비교를 명확하게 구별한다.
9. 전체 프로젝트 handoff ZIP에는 소스·설정·검증·결과·필요한 모델을 넣고, raw 경쟁 데이터·credentials·seal labels·실제 point OOF 전체 행렬·환경·이전 ZIP을 제외한다. 모든 archive member의 SHA와 CRC를 확인하고 필요한 portable 모델 의존성을 검토한다. 주요 결과 20개 파일 폴더는 별도로 제공할 수 있다.
10. Actual seal open, reduced test, public/cloud submission은 모두 새 사용 0이다. 이번 세션에서는 권고 문서만 작성한다. 마지막 결과를 고정하고 감독 프로세스의 종료 상태를 확인한 후 완료를 보고한다.

관측된 시간, 필수 검증 또는 산출물 중 하나라도 부족하면 최종 완료로 표시하지 않는다. 실패나 미확인 항목을 빈 `PASS`로 대체하지 않는다.

# NEXT3 원본 데이터 없는 통계 요약 재생

기존 D002/D007의 조건부 bootstrap 요약은 저장된 작은 aggregate 배열만으로 다시 계산할 수 있다. 최종 ZIP을 일반 폴더로 풀고 handoff 문서의 고정 환경을 준비한 뒤 실행한다.

```powershell
./.review-venv/Scripts/python.exe -I scripts/replay_next3_conditional_statistics.py --output review_next3_conditional_statistics.json
```

출력은 새 JSON 파일에만 쓴다. 이미 같은 이름의 파일이 있으면 다른 새 이름을 지정한다. 원래 파일들을 manifest와 대조하므로 포함 파일을 임의로 수정한 복사본에서는 실행하지 않는다.

## 다시 확인하는 수치

`selection_uncertainty/AGGREGATE_CONTRAST_DRAWS.npz`의 shape는 2,001 × 12다. 첫 행은 원래 관측 대비이고, 나머지 2,000개 행은 이미 저장된 paired whole-series bootstrap 대비다. 첫 행을 draw 표준편차·percentile·양수 비율에 넣지 않는다. 후보 간 동일한 draw 행의 연결을 유지한다.

sample 표준편차는 ddof=1, percentile은 linear 보간을 사용한다. 각 draw에서 후보별 표준화 오차 절댓값의 최대값을 계산하고 그 95% quantile로 기존 공동 envelope를 재구성한다. strict 양수 비율은 `> 0`으로 계산한다. 기존 효과 기준 0.0015도 바꾸지 않는다.

M672는 모델·원본 관측·정답·point prediction 배열이 없는 일반 sibling 폴더에 15개 소스·lock·집계 파일만 복사한 뒤 위 CLI를 실제 실행했다. 불러온 프로젝트 모듈 7개가 복사본에서 왔음을 확인했다. 12개 후보의 D002/D007 요약 및 critical value에 대한 169개 수치 대조가 절대 오차 1e-14 이내로 통과했고, 원래 파일 SHA는 유지됐다. 별도 새 환경을 설치한 실행은 아니다.

primitive 테스트 20개는 손으로 계산할 수 있는 표준편차·percentile·strict 양수 비율, 관측 행 제외, 후보 간 draw 대응, 변형된 NPZ와 잘못된 입력을 검증했다. 후보별 marginal 값이 같더라도 draw 대응을 바꾸면 공동 envelope가 바뀌는 사례를 포함했다.

근거는 `artifacts/next3/maintenance/statistics_copy_replay_v1/RESULTS.json`, `CHILD_RESULT.json`, `INPUT_MANIFEST.json`, `CONDITIONAL_STATISTICS_PRIMITIVE_TESTS.xml` 및 두 `CONDITIONAL_STATISTICS_REPLAY_*` 등록 파일이다.

## 해석의 한계

새 bootstrap draw·학습·point 예측·품질 평가를 추가하지 않았다. 원래 bootstrap 표본추출이나 raw scorer 계산을 이 CLI가 다시 수행하는 것은 아니다. 원래 raw/OOF를 통한 다른 scorer 재구성은 D007의 별도 근거다.

저장된 fitted OOF와 탐색한 후보 pool에 조건부인 수치다. 적응적 후보 생성·screen 통과·학습·이전 세션 DEV 재사용·이후 선택을 보정하지 않으며, 무조건 95% coverage나 독립된 일반화 검증을 주장하지 않는다. 실제 ZIP bytes의 검증도 이 명령과 별개의 사후 receipt를 따른다.

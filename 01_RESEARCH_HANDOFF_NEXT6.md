# NEXT6 인수인계

`00_READ_FIRST_NEXT6_KO.md`에 결과와 한계를 정리했다. 기존 BL80/D2 canonical은 수정하지 않았다. NEXT6는 ResearchPool 6,400 series의 원래 grouped 5-fold 연구이며 Shadow와 Seal을 사용하지 않았다.

## 파일과 역할

- `src/next6/nodes.py`: H 앞 60%로 기존 엔진을 fit하고 나머지 H를 pseudo-online으로 재생해 16개 노드의 경계와 정규화를 계산한다.
- `src/next6/graph.py`: 자기 과거/공통 제어 대비 sparse cross-lag 예측, factorized 상태 확률, bounded graph/state 요약. 경제적 인과 모델이 아니다.
- `src/next6/model.py`: 고정 LightGBM head. graph 피처는 training head에서도 cross-fit한다.
- `src/next6/runtime.py`: 같은 수학을 실행하는 실제 scalar API. 입력은 H와 현재 scalar뿐이다.
- `artifacts/next6/fold_f/outer/M*/model.joblib`: 기존 champion을 대체하지 않는 fold별 DEV 연구 모델.
- `FIGURES/index.html`: 실제 계산한 PNG/SVG 그림 갤러리. 그림 재생성에는 별도의 matplotlib 환경을 사용했다.

## 실행 순서

아래 명령은 프로젝트 루트에서 실행한다. Windows의 실제 실험 환경은 `.venv\Scripts\python.exe -B`였다. 완료된 학습 단계는 재사용한다. 결과 집계·audit·package 명령은 기본적으로 기존 완료 산출물을 덮어쓰지 않으므로 재실행 전 새 output 경로 또는 보존된 완료 receipt를 확인한다.

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/demo_next6_streaming.py --method M3 --fold 0
& '.\.venv\Scripts\python.exe' -B scripts/prepare_next6_nodes.py
& '.\.venv\Scripts\python.exe' -B scripts/run_next6_nested.py --stage inner
& '.\.venv\Scripts\python.exe' -B scripts/run_next6_nested.py --stage outer
& '.\.venv\Scripts\python.exe' -B scripts/score_next6.py
& '.\.venv\Scripts\python.exe' -B scripts/prepare_next6_controls.py
& '.\.venv\Scripts\python.exe' -B scripts/prepare_next6_controls.py --null-graphs
& '.\.venv\Scripts\python.exe' -B scripts/evaluate_next6_controls.py
& '.\.venv\Scripts\python.exe' -B scripts/analyze_next6_stability.py
& '.\.venv\Scripts\python.exe' -B scripts/audit_next6_streaming.py
& '.\.venv\Scripts\python.exe' -B scripts/audit_next6_streaming.py --latency
```

숫자 재현은 기존 split과 hash가 일치하는 원래 ResearchPool cache·reference 피처·모델을 요구한다. 이 자료는 ZIP에 넣지 않았다. `SOURCE_MANIFEST_NEXT6.json`의 외부 의존성 hash를 확인한다. 새 raw scanner나 새 split을 만들며 우회하지 않는다. ZIP의 synthetic scalar demo는 경쟁 데이터 없이 실행할 수 있다. 깨끗한 새 Python 설치까지 검증한 것은 아니며, 복사한 소스와 모델을 기존 고정 가상환경으로 검증했다.

## 설계와 해석상 제한

모든 outer 결과를 열기 전에 5개 fold의 inner 결정을 동결했다. outer 점수를 본 뒤 threshold·weight·node·lag·조합을 바꾸지 않았다. M4가 일부 fold에서만 허용되면 그 결과는 전체 5-fold 후보가 아니다. OWN 대조도 고정 common controls를 유지한다. RANDOM은 같은 edge 수와 target별 parent 수를 쓰지만 관계+전이 24개 요약을 모두 쓰므로 M2/M3 각각의 12개 요약과 head 피처 수가 완전히 같지는 않다. 같은 원 입력·lag·state·memory의 주요 대조는 M1_STRICT이다. 원래 M1에 명시적 상태와 현재 시점이 빠졌음을 감사에서 발견하고, outer 점수 공개 전에 390열 M1_STRICT와 더 엄격한 추가 gate를 고정했다. 학습 recipe는 대조 포함 8개, graph 후보와 기존 inner 결정은 그대로다. M1_INPUT_PARITY_CORRECTION_LOCK.json과 STRICT_INPUT_PAIRED_NEXT6.csv를 확인한다.

동일 raw에서 파생한 두 노드의 확인은 두 독립 원인의 확인이 아니다. 특히 여러 energy/variance 필터의 긴 반응이 outlier를 지속 변화처럼 보이게 할 수 있어 실제 paired 반례를 함께 읽어야 한다. H pseudo-online 비교, 기계적 raw null, 고의로 잘못 연결한 series 경계 trap을 별도 보존했다. state3은 결측/퇴화이며 낮음 state0과 구분한다. 정상/post 확률은 매 시점 함께 계산하고 tau로 어느 모형을 사용할지 고르지 않는다.

paired bootstrap은 series가 단위이며 고정 예측에 조건부다. 전체 CV 재학습 bootstrap이나 과거 DEV 적응성을 보정한 CI가 아니다. 기존 mechanism signature는 사후 관측 지표이고 잠재 DGP 정답이 아니다. 기존 historical regime에는 과거 H transform의 transductive 한계가 있어, 새 H quartile은 별도로 각 outer training에서만 fit했다.

## 오류와 복구

초기 pytest 원본 접근 guard가 ID-only metadata도 차단한 오류는 기존 검토된 compatibility를 재사용해 해결했다. 초기 Numba bool-to-float 문제는 첫 실제 graph fit 전에 수정했다. 이후 float32 범위 밖 scalar/batch 입력 규칙 불일치는 실행 adapter에서 수정했고 과학적 source·학습 파라미터는 바꾸지 않았다. 실패 로그와 수정·재검사 receipt를 보존했다.

## 보존 및 ZIP 범위

ZIP에는 소스·설정·테스트·작은 결과·그림·외부 fold 연구 모델과 비교용 frozen reference weights를 넣었다. raw 관측, credentials, Seal 값, 대용량 feature cache, 전체 OOF/예측 배열, 가상환경, Numba cache는 제외했다. 공식 tau의 전체 per-series 표는 작업 폴더에만 있으며 ZIP에서 제외했다. 그림을 재현하는 소수의 held-out 사례 피처/점수 JSON은 작은 결과 자료다. ZIP 파일별 hash는 `SOURCE_MANIFEST_NEXT6.json`, 별도 smoke·ZIP 무결성 검사는 `HANDOFF_AUDIT_NEXT6.json`이다.

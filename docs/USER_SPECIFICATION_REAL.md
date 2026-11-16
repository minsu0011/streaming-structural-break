이전 Structural Break 프로젝트 작업을 이어서 수행한다.

프로젝트 위치:
C:\Users\minsu\Documents\Structural Break

중요한 상태 변경:
CrunchDAO `structural-break-real-time` setup 인증이 방금 성공했다.

공식 CLI 결과:
"Success! Your environment has been correctly setup."

따라서 이전 BLOCKED_AUTH 상태는 해소되었다.

기존 프로젝트 파일, 소스, 테스트, handoff, audit 문서를 모두 먼저 읽고 현재 상태를 정확히 파악한 후 작업하라.

특히 기존 전달물/문서에 기록된 내용:
- scorer parity 완료
- streaming feature engine 존재
- prefix invariance/leakage guard 존재
- synthetic runner smoke 존재
- 57개 정도의 feature bank 구현
- B0~B7 baseline scaffold 존재
- Logistic / Ridge / HistGB / LightGBM / XGBoost scaffold 존재
- LOCAL_FINAL_SEAL 미개봉
- reduced target 반복 사용 0회
를 코드와 artifact로 직접 재확인한다.

기존 결과를 맹신하지 말고 실제 소스와 테스트로 검증한다.

이번 작업은 약 10시간 autonomous research run이다.

사용자에게 중간 질문하지 말 것.
사소한 오류로 중단하지 말 것.
한 실험이 실패하면 이유를 ledger에 기록하고 다음 합리적 실험으로 진행할 것.
마지막 30~45분은 반드시 audit/reproduction/handoff에 남길 것.

================================================================
0. 가장 먼저 — 인증 성공 및 실제 데이터 위치 확인
================================================================

Crunch setup이 완료되었으므로 실제 competition data가 어디 생성되었는지 탐색한다.

예상 이름:
- X_train.parquet
- y_train.parquet
- y_train_index.parquet
- X_test.reduced.parquet
- y_test.reduced.parquet
- y_test_index.reduced.parquet

단 파일 위치와 이름은 추측하지 말고 실제 filesystem 및 crunch_tools contract를 확인한다.

데이터가 존재하면:
1. 프로젝트의 canonical `data/raw` 위치 결정
2. 필요하다면 안전하게 reference/copy
3. 원본 byte를 임의 수정하지 않음
4. SHA256 manifest 생성

setup credential/token은 어떤 로그/JSON/git에도 기록하지 마라.

데이터 확보가 확인되면 기존:
BLOCKED_AUTH
상태를 폐기하고 새로운 상태를 기록한다.

================================================================
1. 이전 코드의 중요한 결함부터 수정
================================================================

실데이터 성능을 보기 전에 아래 인프라 문제를 수정한다.

A. run_research.py의 장기 실행 구조

현재 `--hours 8` 등이 최대 runtime 역할만 하고,
미리 정의된 experiment list가 끝나면 조기 종료하는 구조라면 수정한다.

진짜 wall-clock autonomous research loop로 만든다.

형태:

while elapsed < research_deadline - final_audit_buffer:

    1. 현재 champion 및 weakest fold/slice 파악
    2. 원인 가설 생성
    3. 한 번에 1~2개의 변경만 candidate로 생성
    4. development CV
    5. baseline/champion과 비교
    6. promote/reject
    7. ledger/checkpoint 저장
    8. 다음 합리적 candidate 선택

실험 후보가 일찍 소진되면 자동으로:
- feature scale refinement
- feature family refinement
- robust normalization
- sampling/weighting
- shallow model tuning
- latency optimization
- complementary detector
순으로 계속 연구한다.

무한 random search는 금지.

실험이 진짜로 더 이상 의미 없을 때만 남은 시간을
robustness/reproduction/audit에 사용한다.

B. raw manifest

mtime을 immutable authority로 사용하지 마라.

SHA256 + byte size를 primary immutable identity로 사용한다.

mtime은 informational metadata일 뿐이다.

C. config single source of truth

현재 research.json과 Python FeatureConfig 간에 설정이 중복되어 있다면 정리한다.

feature scales / alphas / thresholds / lag settings 등은
한 authoritative config에서 생성되도록 한다.

config hash를 feature cache version에 포함시킨다.

================================================================
2. 실데이터 감사부터 다시 수행
================================================================

이번이 실제 데이터 최초 개봉이다.

모델 점수를 보기 전에 데이터를 감사한다.

다음 전체 metadata를 만든다.

- number of series
- historical length
- online length
- break/no-break
- tau distribution
- pre/post break lengths
- historical location/scale statistics
- NaN / Inf
- constant / near-constant
- extreme scales
- heavy tails
- autocorrelation signatures
- duplicate possibility
- identical historical prefix
- identical full historical
- near-duplicate historical

특히 split freeze 전에 duplicate/group contamination을 먼저 조사한다.

이전 코드가:
split freeze → duplicate audit
순이라면 수정한다.

이번에는:

RAW AUDIT
    ↓
EXACT DUPLICATE AUDIT
    ↓
OUTCOME-BLIND NEAR-DUPLICATE / FAMILY GROUPING
    ↓
SPLIT FREEZE

순서로 한다.

================================================================
3. duplicate / near-duplicate guard
================================================================

동일 또는 사실상 동일한 series가 train과 validation으로 갈라지는 것을 막는다.

target/tau를 이용해서 group을 만들지 마라.

outcome-blind하게 historical data만 사용한다.

최소:
- exact hash
- normalized exact hash
- coarse statistical signature

추가적으로 실용적이면:
- downsampled normalized historical fingerprint
- low-dimensional FFT/autocorrelation signature
- random projection 또는 PCA-like signature
- nearest-neighbor similarity

등으로 지나치게 유사한 series를 탐색한다.

하지만 RAM 32GB 환경이므로 O(N²) brute-force pairwise comparison 금지.

근접 grouping threshold는 점수를 보고 고르지 않는다.
데이터 구조/false merge 가능성을 분석해 사전 결정한다.

결과:
DUPLICATE_AUDIT
GROUP_ASSIGNMENTS
GROUP_POLICY

저장.

================================================================
4. split을 실제 ID 기준으로 새로 freeze
================================================================

이전에는 실제 dataset ID가 없었으므로
이제 실데이터 기준 split을 확정한다.

seed를 새로 shopping하지 말고 기존 deterministic seed를 유지할 수 있으면 유지한다.

권장:

전체 series
    ↓
DEV_POOL ≈ 80%
LOCAL_FINAL_SEAL ≈ 20%

반드시 group-aware.

같은 exact/near duplicate group은 같은 role.

DEV_POOL 내부 5 folds.

LOCAL_FINAL_SEAL의 target performance는 이번 run에서도 보지 마라.

LOCAL_FINAL_SEAL ID/hash만 freeze.

이후 split 변경 금지.

점수를 본 뒤 split 수정 절대 금지.

================================================================
5. scorer parity를 실데이터에서도 확인
================================================================

기존 synthetic 128-case/256 comparison parity를 다시 검증.

그 다음 실 TRAIN의 일부 prediction을 사용하여:
- our scorer
- official scorer

의 parity도 확인할 수 있으면 확인한다.

max abs error 기록.

scorer parity FAIL이면 모델 연구를 중지하고 scorer부터 수정.

================================================================
6. official EWMA baseline 실데이터 재현
================================================================

가장 먼저 실제 official-style EWMA baseline을 돌린다.

B0:
constant

B1:
official EWMA-like baseline

DEV 5-fold TS-AUC 측정.

반드시 기록:
- mean
- median
- fold scores
- worst fold
- std
- time bucket profile
- latency

이 시점의 결과를 CHECKPOINT로 저장.

이것이 최초 실제 benchmark다.

================================================================
7. 기존 feature bank를 실데이터에서 검증
================================================================

현재 약 57개 streaming feature가 구현되어 있을 것이다.

이를 그대로 믿지 말고 각 family를 실데이터에서 검사한다.

A LOCATION
B SCALE
C TAIL/DISTRIBUTION
D DEPENDENCE
E CROSS-SCALE
F EVIDENCE MEMORY

각 feature에 대해:
- finite rate
- variance
- saturation
- extreme quantile
- inter-feature correlation
- CPU cost
- useful dynamic range

constant/nearly constant feature 제거.

99.99% saturation 등 비정상 feature 진단.

feature 값에 tau/online length가 들어갈 수 없는지 재확인.

================================================================
8. Statistical baseline ladder
================================================================

다음을 실제 DEV CV에서 모두 평가.

B0 constant
B1 official EWMA
B2 multi-scale mean shift
B3 CUSUM
B4 variance/scale
B5 tails
B6 dependence
B7 manual combined detector

각 candidate:
fold-wise TS-AUC
worst fold
time fracture
break-age diagnostic
latency

저장.

여기서 처음으로 실제 데이터에서 어떤 family가 강한지 파악한다.

결과를 보기 전 가정:
"volatility가 중요할 것이다"
등을 authority로 삼지 마라.

실데이터가 authority다.

================================================================
9. Supervised 모델 1차
================================================================

Streaming feature bank로 만든 causal row feature만 사용.

dataset_id feature 사용 금지.
tau feature 사용 금지.
online final length 사용 금지.

M1 Logistic
M2 Ridge/logistic-like linear
M3 HistGradientBoosting
M4 LightGBM
M5 shallow XGBoost

우선 작은 reasonable parameter set만 비교한다.

GPU 사용이 가능하더라도
10M inference latency가 나빠지는 모델을 무작정 champion으로 만들지 않는다.

각 모델:
- CV TS-AUC
- folds
- worst fold
- model size
- single-point latency
- deterministic save/load
평가.

================================================================
10. 중요한 신규 연구 — TS-AUC와 training objective alignment
================================================================

현재 supervised training이 일반 row-wise binary classification 위주라면
실제 metric과 mismatch가 존재할 수 있다.

먼저 ordinary baseline 결과를 확보한 후 아래를 비교한다.

S0 ordinary rows

S1 deterministic per-series capped sample

S2 time-stratified deterministic sampling

S3 TS-AUC contribution-aware weighting

TS-AUC는 같은 online step에서 positive/negative series를 ranking하므로
time step별 class balance 및 pair contribution을 고려한 sample weighting을 설계한다.

예를 들면 training weight 후보:
각 time t에서
effective pair contribution에 대응하도록
class/time weighting

단 metric 자체를 training data로 직접 계산해 label leakage를 일으키지 않는다.

결과를 보고 weight formula를 계속 변형하지 말고
몇 개의 사전 정의된 weighting family만 비교.

가능하면 LightGBM/XGB ranking objective도 별도 exploratory candidate로 평가하되
series/time group 정의가 metric과 논리적으로 맞는지 검토 후 수행.

================================================================
11. feature-family ablation
================================================================

A
B
C
D

A+B
A+D
B+D
A+B+D
A+B+C+D
ALL

필요시 E/F.

각 family의:
- standalone performance
- marginal contribution
- worst-fold impact
- latency
를 분석한다.

feature importance 하나만 보고 결론내지 않는다.

permutation/ablation evidence를 우선.

================================================================
12. 시간 fracture / detection delay
================================================================

TS-AUC 전체 점수만 보지 않는다.

online time:
0-9
10-24
25-49
50-99
100-199
200-399
400+

break 이후 age:
0-4
5-9
10-19
20-49
50+

가능한 범위에서 분석.

특히:
높은 전체 점수가
break가 오래 지난 뒤에만 잘 맞혀서 생긴 것인지 확인.

early detection이 약한 모델의 원인을 분석.

================================================================
13. 실데이터 중심 adaptive research loop
================================================================

기본 ladder 이후 남은 시간에는
실데이터 결과에 따라 연구 방향을 고른다.

예:

CASE 1
mean/location strongest
→ normalization / CUSUM / fast-slow evidence 개선

CASE 2
scale strongest
→ robust scale / EWMA squared / abs-residual / volatility persistence 개선

CASE 3
dependence strongest
→ AR residual / whiteness / autocorrelation streaming approximation 집중

CASE 4
tail strongest
→ quantile/exceedance features 개선

CASE 5
family별 서로 보완
→ calibrated shallow ensemble

CASE 6
fold instability
→ 해당 fold의 historical structure와 feature fracture 분석

한 번에 한 family를 크게 바꾸지 말 것.
각 candidate는 가설을 명확히 기록.

================================================================
14. complementary detector 연구
================================================================

시간이 충분히 남고 기본 모델이 안정되면
다른 메커니즘 detector를 1~3개 추가한다.

예:
- Page-Hinkley style
- GLR-like cheap statistic
- robust sequential likelihood
- AR residual cumulative evidence
- rank/quantile based detector

매 point O(window) full recompute는 피한다.

streaming incremental update 중심.

candidate가 개선되지 않으면 즉시 폐기/ledger 기록.

================================================================
15. latency 최적화
================================================================

실제 모델에서 benchmark.

1M/10M point extrapolation.

특히 XGBoost가 이전 synthetic에서
single-row inference가 다른 모델보다 상당히 느렸으므로
실제 모델에서도 확인한다.

점수 차가 작은 경우:
fast candidate를 별도로 유지.

최대 세 후보:
PRIMARY_DEV_CANDIDATE
BACKUP_ROBUST_CANDIDATE
BACKUP_FAST_CANDIDATE

================================================================
16. reduced test 사용 정책
================================================================

X_test.reduced / y_test.reduced가 실제로 존재하더라도
튜닝용으로 사용하지 않는다.

오늘 허용:
- official baseline sanity 1회
- 종료 직전 primary dev candidate smoke 1회

최대 2회.

각 사용시:
REDUCED_TEST_USAGE_LEDGER
에 기록.

reduced 결과를 보고 feature/hyperparameter 수정 금지.

================================================================
17. LOCAL_FINAL_SEAL
================================================================

오늘도 열지 마라.

이번 목표는 실데이터 DEV research pipeline 구축 및 champion 후보 선정이다.

LOCAL_FINAL_SEAL target은 untouched.

최종 문서에:
LOCAL_FINAL_SEAL_OPENED = false
명시.

================================================================
18. 실험 종료 조건
================================================================

`--hours 10`을 진짜 wall-clock 연구 budget으로 사용한다.

다만 의미 없는 반복으로 시간을 채우지는 않는다.

남은 시간이 45분 이하가 되면 새 실험 시작 중단.

그 뒤:
- 전체 tests
- deterministic rerun
- selected CV rerun
- streaming/offline parity
- official crunch_tools.test 가능하면 수행
- artifact audit
- SHA256
- handoff
수행.

================================================================
19. 최종 산출물
================================================================

기존 파일을 업데이트하고 다음을 추가/갱신.

00_GPT_READ_FIRST.md
01_GPT_HANDOFF.md

REAL_DATA_MANIFEST.json
REAL_DATA_AUDIT.md
REAL_DATA_AUDIT.json

DUPLICATE_AUDIT.json
GROUP_POLICY.md
GROUP_ASSIGNMENTS.parquet

SPLIT_LOCK_REAL.json

SCORER_PARITY_REAL.json

BASELINE_RESULTS_REAL.csv
MODEL_RESULTS_REAL.csv
FEATURE_ABLATIONS_REAL.csv
CV_RESULTS_REAL.csv
TIME_FRACTURE_REAL.csv
BREAK_AGE_DIAGNOSTICS.csv

SAMPLING_WEIGHTING_RESULTS.csv

LATENCY_REAL.json
DETERMINISM_REAL.json
DEPLOYMENT_SMOKE_REAL.json

REDUCED_TEST_USAGE_LEDGER.json

DEV_CHAMPION_CANDIDATES.json

EXPERIMENT_LEDGER.csv

NEXT_SESSION_PLAN.md
LIMITATIONS.md

SOURCE_MANIFEST.json

그리고 최종 GPT 전달 ZIP.

================================================================
20. 최종 보고서에 반드시 숫자로 적을 것
================================================================

1. 실제 train series 수
2. break/no-break 수
3. historical 길이 범위
4. online 길이 범위
5. duplicate/group 수
6. dev/seal 수
7. official baseline DEV TS-AUC
8. best statistical baseline
9. best supervised model
10. PRIMARY_DEV_CANDIDATE
11. 5-fold 각각 TS-AUC
12. mean / median / worst
13. feature family standalone 순위
14. best marginal feature family
15. 가장 약한 time bucket
16. break 직후 detection 능력
17. inference µs/point
18. 10M point 예상 시간
19. reduced test 사용 횟수
20. LOCAL_FINAL_SEAL 개봉 여부
21. leakage tests
22. determinism
23. Crunch runner smoke
24. 현재 가장 큰 병목 3개

성과가 나쁘더라도 그대로 보고.

================================================================
21. 하드 금지
================================================================

LOCAL_FINAL_SEAL 개봉 금지.
reduced target 반복 조회 금지.
dataset ID target encoding 금지.
future suffix 금지.
online final length 금지.
tau feature 금지.
row-random CV 금지.
fold seed shopping 금지.
score 본 뒤 duplicate threshold 수정 금지.
leaderboard chasing 금지.
기존 금융모델 직접 이식 금지.
Transformer 대규모 sweep 금지.

================================================================
22. 지금 시작
================================================================

계획 설명만 하고 멈추지 말고 실제 작업을 시작한다.

첫 순서:

1. 이전 프로젝트/결과 읽기
2. Crunch 인증 성공 여부와 데이터 파일 확인
3. actual raw manifest
4. duplicate-first audit
5. real split freeze
6. scorer parity
7. official baseline
8. statistical baseline ladder
9. supervised ladder
10. weighting/sampling
11. ablation/fracture
12. adaptive research loop
13. final audit/handoff

약 10시간 동안 사용자 개입 없이 계속 진행하라.
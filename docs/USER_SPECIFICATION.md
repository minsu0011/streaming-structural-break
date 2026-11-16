너는 이 프로젝트의 수석 ML 연구원 + 데이터 엔지니어 + 검증 담당자다.

이번 작업은 ADIA Lab / CrunchDAO의 `structural-break-real-time` 대회를 위한 완전히 새로운 프로젝트의 첫 장기 연구 세션이다.

목표는 기존 금융 AI 프로젝트를 억지로 이식하는 것이 아니다.

기존 프로젝트에서 검증된 좋은 연구 습관과 일부 일반적인 시계열 아이디어만 최소한 참고하고, 데이터 획득부터 평가기, 검증 구조, streaming feature engine, baseline, 초기 모델 후보까지 이 대회에 맞게 처음부터 다시 구축한다.

이번 세션은 약 10시간 동안 사용자가 개입하지 않아도 계속 진행하는 autonomous run으로 수행한다.

사용자에게 중간 질문을 하지 마라.
사소한 장애가 발생했다고 작업을 중단하지 마라.
한 실험이 실패하면 원인을 기록하고 다음 합리적인 실험으로 넘어가라.
실험 결과를 본 뒤 평가 기준이나 봉인 규칙을 완화하지 마라.
시간이 남으면 반복 실험과 검증을 계속 수행하라.
마지막 30~45분은 반드시 결과 정리, 재현성 검사, 다음 세션 handoff에 사용한다.

======================================================================
0. 하드웨어 / 실행 환경
===============

주 실행 환경은 Windows 기반 노트북이다.

대략적인 자원:

* Intel Core Ultra 7 258V
* RAM 32GB
* RTX 3080 Ti eGPU 사용 가능
* Windows 11

단, 위 사양을 맹신하지 말고 실제 실행 시 CPU/RAM/GPU/CUDA 상태를 먼저 탐지해서 기록한다.

중요:

1. 기존 Python 환경, conda 환경, 다른 프로젝트 패키지를 삭제하거나 변경하지 마라.
2. 가능하면 프로젝트 전용 `.venv`를 생성한다.
3. 글로벌 pip uninstall 금지.
4. 기존 프로젝트의 환경을 깨뜨리지 마라.
5. GPU가 정상적으로 사용 가능하면 학습에는 활용하되 GPU가 없다고 전체 파이프라인을 중단하지 마라.
6. 이 대회는 streaming inference latency가 중요하므로 GPU 학습 성능보다 실제 CPU inference 비용을 더 중요하게 본다.
7. RAM 32GB를 고려하여 수백만 row짜리 feature matrix를 무조건 pandas float64로 메모리에 적재하지 마라.
8. float32, int32, chunk processing, parquet cache 등을 적극 활용한다.

CPU thread oversubscription을 방지한다.

numpy / sklearn / xgboost / lightgbm 등이 동시에 모든 thread를 잡아먹지 않게 실험 runner 차원에서 합리적인 thread 제한을 둔다.

======================================================================

1. 프로젝트 철학
   ======================================================================

이 프로젝트는 새 프로젝트다.

이전 EPS, Expected P/E, PE, 뉴스 모델, SEC crawler, KR/US transfer 모델 등을 이 대회에 직접 연결하지 마라.

특히 다음은 가져오지 않는다.

* EPS feature
* P/E feature
* 주가 방향 예측 label
* OHLCV 전용 feature
* 뉴스 / SEC event feature
* 종목 ticker logic
* 금융시장 domain-specific hard coding
* 기존 champion model
* 기존 학습 데이터

이번 대회의 입력은 generic univariate time series이고 목표는 실시간 structural break detection이다.

기존 프로젝트에서 참고할 수 있는 것은 다음 정도뿐이다.

A. 검증 철학

* train/dev/seal 분리
* untouched seal
* 결과를 본 뒤 gate 변경 금지
* 미래누수 방지
* group 단위 split
* deterministic seed
* experiment ledger
* manifest/hash
* feature family ablation
* worst-fold 검사
* 시간축 fracture 분석

B. 일반적인 시계열 사고방식

* rolling / expanding
* fast vs slow regime comparison
* volatility/variance 변화
* autocorrelation/dependency 변화
* robust statistics
* multi-scale feature

C. 범용 인프라

* hashing
* logging
* deterministic run
* experiment table
* artifact manifest
* config management

기존 금융 프로젝트가 현재 workspace 주변에서 명확하게 발견되는 경우에만 최대 20분 동안 READ-ONLY로 살펴봐도 된다.

목표는 재사용 가능한 범용 utility가 있는지 확인하는 것이다.

기존 프로젝트 전체를 복사하지 마라.
기존 model artifact를 가져오지 마라.
기존 데이터 파일을 가져오지 마라.
기존 프로젝트를 수정하지 마라.

재사용하려는 코드가 있다면 반드시:

1. generic utility인지 확인
2. 금융 domain dependency가 없는지 확인
3. 새 프로젝트에 작은 독립 모듈로 복사
4. 원본 provenance 기록

애매하면 재사용하지 말고 새로 작성한다.

======================================================================
2. 공식 대회 contract를 가장 먼저 고정
===========================

대회 slug:

`structural-break-real-time`

공식 quickstarter와 공식 Crunch competition repository를 authority로 사용한다.

첫 세션에서는 Reddit, Kaggle write-up, leaderboard solution, 참가자 포럼의 고득점 전략 등을 뒤지며 따라 하지 마라.

이번 목표는 independent baseline과 검증 체계를 먼저 확보하는 것이다.

공식 contract를 직접 확인하고 아래 내용을 문서로 freeze한다.

각 training series:

(dataset_id, x_historical, x_online, tau_index)

* historical은 전체가 처음부터 주어진다.
* historical에는 structural break가 없는 것이 보장된다.
* online은 실제 inference에서 한 observation씩 들어온다.
* tau_index는 training에서만 제공된다.
* tau_index=None이면 break 없음.
* prediction은 매 online observation마다 하나.
* prediction 범위 [0,1].
* prediction은 현재 observation까지의 정보만 사용.
* 다음 observation을 미리 볼 수 없음.
* online iterable은 cloud에서 one-pass.
* 전체 online 길이를 미래 정보로 사용 금지.
* 다른 series의 미래/상태를 현재 series 판단에 부적절하게 사용하지 않는다.
* infer에서 현재 point를 받은 뒤 그 point까지 사용해서 score를 출력한다.

OFF-BY-ONE 문제는 추측하지 말고 실제 official reduced labels와 contract를 이용해 test를 작성해서 확정한다.

예를 들어 tau_index가 k이면:

t < k → 0
t >= k → 1

인지 공식 자료와 실제 label alignment로 검증한다.

검증 결과를:

`docs/LABEL_CONTRACT.md`

에 기록한다.

======================================================================
3. 새 프로젝트 root 구축
=================

현재 디렉터리가 이미 이 대회 전용 프로젝트가 아니라면 안전한 위치에 새 폴더를 만든다.

권장 이름:

`STRUCTURAL_BREAK_RT_2026`

구조 예시:

STRUCTURAL_BREAK_RT_2026/
README.md
.gitignore
requirements.txt
pyproject.toml 또는 requirements-lock.txt

```
configs/
data/
    raw/
    interim/
    processed/
    cache/

docs/
src/
    data/
    features/
    models/
    scoring/
    validation/
    streaming/
    utils/

scripts/
notebooks/
experiments/
    configs/
    results/
    logs/

artifacts/
    audits/
    baselines/
    features/
    models/
    validation/
    deployment/
    handoff/

tests/
checkpoints/
```

raw 대회 데이터는 git에 넣지 않는다.

모델 binary, 거대한 parquet cache도 무조건 git에 넣지 않는다.

git repository를 초기화하고 단계별 commit을 남긴다.

예:

* bootstrap
* data-contract
* scorer-and-splits
* streaming-baselines
* feature-bank
* model-research
* validation
* deployment-smoke
* handoff

======================================================================
4. Crunch 환경 및 데이터 확보
=====================

먼저 다음을 확인한다.

* Python version
* pip
* git
* crunch-cli version
* numpy
* pandas
* pyarrow
* scipy
* scikit-learn
* xgboost
* lightgbm
* joblib
* numba 여부
* GPU / CUDA 상태

프로젝트 전용 venv를 우선 사용한다.

Crunch CLI가 없다면 설치한다.
설치되어 있지만 공식 runner 요구 버전보다 낮으면 프로젝트 환경 안에서 업그레이드한다.

개인의 setup token이나 credential을 로그, README, git, JSON 등에 절대 기록하지 마라.

이미 Crunch competition environment가 설정되어 있다면 그대로 사용한다.

설정이 필요한 경우:

* 현재 환경에 credential/setup 정보가 안전하게 존재하는지 확인
* credential을 출력하지 않는다
* 가능한 공식 setup 절차를 사용한다

인증 정보가 없어 실제 데이터 다운로드가 막히더라도 전체 run을 즉시 종료하지 않는다.

그 경우:

1. BLOCKER_AUTH.md 작성
2. 공식 public quickstarter와 scorer를 확보
3. 프로젝트 구조
4. scorer
5. feature engine
6. unit tests
7. synthetic streaming tests
   까지 계속 만든다.

단, 실제 데이터가 없는 상태에서 가짜 CV 점수나 가짜 성능을 생성하지 마라.

데이터 접근이 가능하면 즉시 공식 competition data를 확보한다.

예상 파일명을 맹신하지 말고 실제 파일을 조사하되 공식 runner 기준으로 다음 파일들이 존재할 가능성이 높다.

* X_train.parquet
* y_train_index.parquet
* X_test.reduced.parquet
* y_test.reduced.parquet
* 기타 setup이 생성한 파일

파일을 확보한 직후 raw data를 수정하지 않는다.

각 파일에 대해:

* size
* row count
* columns
* index
* dtype
* SHA256
* modification time

을 저장한다.

`artifacts/audits/RAW_DATA_MANIFEST.json`

생성.

======================================================================
5. 데이터 감사 — 모델보다 먼저
===================

모델링 전에 반드시 전체 데이터 구조를 감사한다.

training series마다 다음 metadata table을 만든다.

* dataset_id
* historical_length
* online_length
* has_break
* tau_index
* tau_fraction = tau / online_length
* pre_break_online_length
* post_break_length
* historical mean
* historical std
* historical median
* historical MAD
* historical min/max
* obvious NaN/Inf counts
* online NaN/Inf counts

단 `online_length`는 TRAIN 분석에서는 사용할 수 있지만 실제 inference feature에는 사용 금지라는 표시를 강제한다.

분석:

1. 전체 series 수
2. break / no-break 비율
3. historical length distribution
4. online length distribution
5. tau distribution
6. tau / online length distribution
7. pre-break length
8. post-break length
9. constant / near-constant series
10. missing / inf
11. duplicate series 가능성
12. identical prefix 가능성
13. scale normalization 상태
14. historical과 pre-break online의 stationarity 차이
15. historical 자체의 다양한 dependency structure
16. 매우 짧은 online series 비중
17. 극단적인 heavy-tail series

중요:

TRAIN 데이터의 true tau 이후를 이용해 feature가 “미래를 미리 아는” 형태가 되지 않도록 feature 생성 API를 처음부터 streaming API로 만든다.

offline feature extraction 함수가 x_online 전체 array를 받더라도 내부적으로 반드시 앞에서부터 point 하나씩 StreamingState에 feed하고 각 시점 feature를 생성하도록 한다.

훈련과 inference가 서로 다른 feature code path를 사용하지 마라.

TRAIN feature 생성과 CLOUD infer가 같은 streaming feature implementation을 호출하게 한다.

이게 매우 중요하다.

결과:

`artifacts/audits/DATA_AUDIT.json`
`artifacts/audits/DATA_AUDIT.md`
`artifacts/audits/SERIES_METADATA.parquet`

생성.

======================================================================
6. 누수 방지 contract
=================

다음 규칙을 코드 + test로 강제한다.

절대 금지 feature:

* 미래 x_online 값
* 최종 online length
* tau
* has_break
* post-break 총 길이
* 미래 observation count
* dataset ordering에서 유추한 target
* y_test.reduced target
* 다른 validation fold target 정보

training label 생성용 tau는 label을 만들 때만 사용한다.

FeatureState는 tau를 입력으로 받지 못하게 API 자체를 설계한다.

예:

state = StreamingFeatureState(x_historical)

for t, x in enumerate(x_online):
features = state.update_and_get(x)

형태.

`state.update_and_get()`에는 target/tau/online_length가 전달될 수 없게 한다.

누수 unit test를 만든다.

Prefix invariance test:

동일한 historical + 동일한 online prefix를 가진 두 input에서,
그 뒤의 future suffix를 완전히 바꾸더라도
prefix 시점까지의 features와 predictions는 bitwise 또는 tolerance 내 동일해야 한다.

이 검사는 핵심 hard gate다.

결과:

`docs/LEAKAGE_CONTRACT.md`
`artifacts/validation/PREFIX_INVARIANCE.json`

======================================================================
7. Local TS-AUC scorer를 공식 구현과 parity
=====================================

평가 함수는 직접 정확히 구현한다.

일반 ROC-AUC 하나를 전체 row에 계산하는 것은 primary metric이 아니다.

TS-AUC:

각 online step t마다 현재 t까지 살아 있는 series들을 모은다.

그 시점 label:

* break가 이미 발생 → positive
* 아직 발생하지 않았거나 no-break → negative

각 t에서 AUC_t 계산.

weight:
n_positive(t) * n_negative(t)

전체 weighted average.

공식 scorer source와 동작을 비교해서 parity test를 만든다.

아래 edge case도 검증:

* 한 class밖에 없는 t
* series length가 서로 다름
* duplicate score
* constant predictions
* perfect predictions
* reversed predictions
* NaN
* inf
* index ordering
* shuffled input rows

hard assertions:

constant score → 약 0.5
perfect target score → 1.0
reverse score → 약 0.0

가능한 범위에서 official scorer와 random synthetic cases 100회 이상 parity를 확인한다.

결과:

`src/scoring/ts_auc.py`
`tests/test_ts_auc.py`
`artifacts/validation/SCORER_PARITY.json`

scorer parity가 실패한 상태에서는 본격 모델 연구를 진행하지 않는다.
먼저 scorer부터 수정한다.

======================================================================
8. TRAIN 내부 split을 먼저 봉인
========================

데이터를 본 뒤 좋은 seed를 골라서는 안 된다.

고정 seed 하나를 즉시 결정한다.

예:
20260908

dataset_id에 deterministic hash를 적용해서 series 단위로 role을 배정한다.

권장:

DEVELOPMENT_POOL ≈ 80%
LOCAL_FINAL_SEAL ≈ 20%

LOCAL_FINAL_SEAL은 이번 10시간 런에서 성능 측정에 사용하지 마라.

즉:

* feature 선택에 사용 금지
* hyperparameter 선택에 사용 금지
* model ranking 금지
* ablation 판단 금지

metadata representation만 확인할 수 있지만 결과를 보고 seal allocation을 변경하지 않는다.

Development pool 내부에는 deterministic group folds를 만든다.

예:
5-fold series-level CV.

중요:
같은 series의 서로 다른 time row가 train/validation으로 갈라지는 row random split 절대 금지.

반드시 series_id 단위 split.

또한 한 번 만든 split은 저장하고 이후 변경하지 않는다.

생성:

`artifacts/validation/SPLIT_LOCK.json`
`artifacts/validation/SPLIT_ASSIGNMENTS.parquet`

파일 안에:

* seed
* algorithm
* ids
* hash
* creation timestamp
* counts

기록.

======================================================================
9. reduced local test의 역할 제한
============================

공식 setup에 `X_test.reduced` / `y_test.reduced`가 존재한다면 이것을 hyperparameter playground로 만들지 마라.

역할:

`PUBLIC_LOCAL_REDUCED_DIAGNOSTIC`

으로 정의.

초기 official baseline 확인 1회.
그리고 장기런 종료 직전 champion candidate smoke 1회 정도만 허용.

반복적으로 reduced test 점수를 보면서 튜닝하지 않는다.

모델 선택 primary authority는 DEVELOPMENT_POOL의 series-level CV TS-AUC다.

LOCAL_FINAL_SEAL은 아직 열지 않는다.

======================================================================
10. Label dataset 구축
====================

training data에서 각 online point를 supervised example로 만들 수 있다.

단 feature는 streaming engine에서 생성.

row:

series_id
time_online
features...
target

target:
break가 해당 time까지 이미 발생했으면 1
아니면 0.

full 5~10 million row가 만들어질 수 있으므로 메모리 사용을 측정한다.

가능하면:

* float32
* chunk parquet
* partition by fold 또는 shard
* uint8 target
* int32 id/t

로 저장.

처음부터 모든 feature를 대규모 DataFrame으로 중복 복사하지 않는다.

Feature cache에는 feature contract/version hash를 기록한다.

Feature logic 변경 시 오래된 cache를 자동으로 잘못 재사용하지 않도록 한다.

======================================================================
11. Baseline ladder
===================

바로 XGBoost로 뛰지 마라.

먼저 독립 statistical baseline ladder를 만든다.

B0
constant 0.5

B1
공식 EWMA mean z-score baseline을 최대한 동일하게 재현

B2
multiple EWMA mean shifts

B3
CUSUM mean detector

B4
scale / variance detector

B5
tail detector

B6
dependence detector

B7
simple manually normalized ensemble

각 baseline에 대해 5-fold development TS-AUC를 계산.

기록:

* fold scores
* mean
* median
* std
* worst fold
* best fold
* inference us/point
* memory/state bytes per series

공식 quickstarter의 약 0.517이라는 예시 숫자는 sanity reference일 뿐 hard expected score로 사용하지 않는다.
현재 데이터 / split / implementation에 따라 다를 수 있다.

======================================================================
12. Streaming Feature Bank V1
=============================

Feature family를 명확히 분리한다.

---

## FAMILY A — LOCATION / MEAN

예:

* standardized current residual
* EWMA residual means with several alphas
* EWMA z scores
* positive/negative CUSUM
* running mean difference
* short-vs-long mean
* absolute mean departure
* signed mean departure

multi-scale를 사용하되 feature explosion은 막는다.

예시 effective scales:
~5
~10
~20
~40
~80
~160

모두 쓸 필요는 없으며 correlation과 ablation으로 줄인다.

---

## FAMILY B — SCALE / VOLATILITY

기존 금융 프로젝트의 realized-vol 사고방식만 참고하고
OHLCV 코드는 가져오지 않는다.

예:

* EWMA squared residual
* EWMA absolute residual
* variance ratio vs historical
* short/long scale ratio
* running second moment
* robust scale departure
* volatility CUSUM
* log variance ratio

historical variance가 극소인 경우 numerical guard.

---

## FAMILY C — DISTRIBUTION / TAIL

예:

* historical ±1σ / ±2σ / ±3σ threshold exceedance
* absolute extreme count
* positive extreme count
* negative extreme count
* EWMA exceedance rates
* empirical historical quantile bin position
* running quantile-bin frequency deviation
* sign imbalance
* robust outlier score

full KS를 매 point마다 다시 계산하지 않는다.

필요하다면 historical quantile thresholds를 고정해 cheap approximation을 만든다.

---

## FAMILY D — DEPENDENCE

이 family는 중요하다.

예:

* lag-1 product EWMA
* lag-2 / lag-4 / lag-8 dependency proxies
* running autocorrelation deviation
* historical lag correlation과 online estimate difference
* historical AR(1) / small AR model residual
* predictive residual magnitude
* residual autocorrelation
* likelihood-related statistic
* whiteness deviation

historical AR fitting은 series당 한 번만 한다.

매 point마다 AR 재학습 금지.

---

## FAMILY E — CROSS-SCALE / FRACTURE

기존 프로젝트의 시간축 fracture 개념을 일반화해서 사용.

예:
fast mean / slow mean disagreement
fast volatility / slow volatility disagreement
short dependence / long dependence disagreement
location evidence vs scale evidence
current shock vs accumulated evidence

단 불필요한 pairwise interaction 폭발은 금지.

---

## FAMILY F — EVIDENCE MEMORY

structural break는 한 번 발생하면 target이 이후 계속 1이므로,
누적 evidence memory가 도움이 될 수 있다.

후보:

* cumulative max detector evidence
* decayed max
* accumulated log evidence
* hysteresis-like state

하지만 false positive도 영구화될 수 있다.

따라서 raw instantaneous score와 cumulative evidence를 둘 다 비교하고 ablation으로 결정한다.

======================================================================
13. Historical normalization
============================

각 series는 자체 historical segment가 reference다.

가능한 한 series-specific normalization을 활용한다.

하지만 단순 mean/std만으로 끝내지 않는다.

historical에서 사전에 계산 가능한 값:

* mean
* std
* median
* MAD
* quantiles
* moments
* lag correlations
* small AR coefficients
* residual variance
* tail rates

이 정보들은 online 시작 전에 계산 가능하므로 합법적이다.

중요:

historical을 매 online point마다 다시 scan하지 마라.

series 시작 시 한 번 계산해서 state에 저장.

O(H + T) 구조를 목표로 한다.

O(H*T), O(T²) 금지.

======================================================================
14. Feature unit test
=====================

각 feature family마다 synthetic series를 만든다.

예:

1. no break
2. positive mean shift
3. negative mean shift
4. variance increase
5. variance decrease
6. heavy-tail change
7. autocorrelation increase
8. autocorrelation sign reversal
9. AR parameter change
10. isolated outlier but no permanent break

기대 반응을 검사.

목표는 synthetic score를 높이는 것이 아니라 feature가 의도한 물리적 의미로 반응하는지 확인하는 것이다.

특히 single outlier와 persistent break를 구분할 수 있는 feature가 있는지 본다.

======================================================================
15. Model ladder V1
===================

Feature bank가 안정화되면 supervised models를 비교한다.

무거운 deep model은 이번 첫 10시간의 우선순위가 아니다.

우선:

M0
manual statistical ensemble

M1
logistic regression

M2
regularized linear / calibrated linear model

M3
small HistGradientBoosting 또는 equivalent

M4
small LightGBM

M5
small shallow XGBoost

패키지 설치와 cloud compatibility를 확인한다.

모델 선택에서 CV 점수 하나만 보지 않는다.

평가 tuple:

(
median_TS_AUC,
mean_TS_AUC,
worst_fold,
std,
latency_us_per_point,
model_size,
state_memory,
determinism
)

복잡한 모델이 0.0005 좋아지면서 inference가 20배 느려지는 경우 자동 champion으로 승격하지 마라.

======================================================================
16. Supervised training sampling
================================

모든 time point가 동일한 정보량을 가진다고 가정하지 않는다.

하지만 label을 보고 편리하게 cherry-pick한 sampling도 피한다.

몇 가지 사전 정의된 sampling policy를 비교할 수 있다.

S0:
all development rows

S1:
deterministic time-stratified subsample

S2:
per-series capped deterministic sampling

S3:
time-step/class weighting으로 TS-AUC 구조를 어느 정도 반영

sampling policy 또한 config와 seed를 저장한다.

validation series는 절대로 subsampling 후 training으로 섞지 않는다.

======================================================================
17. 평가를 score 하나로 축약하지 마라
=========================

각 candidate에 대해 TS-AUC 외에도 diagnostic profile을 만든다.

time bucket:

0–9
10–24
25–49
50–99
100–199
200–399
400+

가능한 범위에서.

그리고:

* early break
* middle break
* late break
* no-break
* short historical
* long historical
* short online
* long online

performance slice를 만든다.

주의:
online total length는 diagnostic에서만 사용할 수 있고 inference feature로는 금지.

또 detector family에 따라 break type label이 공식적으로 제공되지 않을 수 있다.

break type이 없는 경우 true type을 지어내지 마라.

대신 관측 가능한 post-hoc statistical signature를 diagnostic용으로 분류할 수는 있지만
그것이 true DGP label인 것처럼 주장하면 안 된다.

======================================================================
18. Feature-family ablation
===========================

다음 비교를 반드시 수행한다.

A only
B only
C only
D only

A+B
A+D
B+D
A+B+C+D

필요하면 E/F 추가.

목적:
“feature 100개 넣으니 올라갔다”가 아니라 어느 break information family가 실제 일반화되는지 확인.

각 family에 대해:

* median gain
* worst-fold effect
* latency cost
* redundancy

기록.

성능이 약하거나 불안정한 family는 제거 후보.

======================================================================
19. 시간축 fracture 연구
===================

기존 금융 프로젝트에서 가져올 핵심 연구 습관 중 하나다.

각 model이:

* very early online
* early
* middle
* late

에서 성능이 어떻게 달라지는지 본다.

또 evidence age 기준:

break 직후 0~4 points
5~9
10~19
20~49
50+

가능한 경우 diagnostic 계산.

목표:

“break 이후 시간이 충분히 지난 경우만 잘 맞히는 모델”인지,
“실제 early detection이 가능한 모델”인지 분리.

TS-AUC 자체를 최종 authority로 유지하되 이 fracture 분석을 모델 해석에 사용한다.

======================================================================
20. Inference latency를 연구 단계부터 측정
=================================

공식 환경에서는 최대 매우 많은 online point에 prediction해야 하므로 latency가 중요하다.

각 candidate의 실제 streaming `update + prediction` 시간을 benchmark한다.

warm-up 수행 후:

* median us/point
* p95
* p99
* 1M point extrapolation
* 10M point extrapolation

기록.

Python allocation을 줄인다.

매 point마다 금지:

* pandas DataFrame 생성
* sklearn DataFrame 변환
* model reload
* AR refit
* historical rescan
* list 전체 재계산
* expensive rolling slicing
* disk write

가능하면 numpy scalar / fixed-size array / deque / simple Python scalar state를 사용한다.

LightGBM/XGBoost single-row prediction overhead가 심하면 별도로 측정한다.

필요하면:

* compact model
* fewer trees
* linear surrogate
* manual score layer

등을 비교한다.

성능과 latency의 Pareto table을 만든다.

======================================================================
21. Cloud-style streaming contract
==================================

최종 inference implementation은 공식 형태를 그대로 따른다.

`train(datasets, model_directory_path)`

`infer(datasets, model_directory_path)`

infer는 generator.

초기 ready yield.
각 x_online point마다 정확히 prediction 1회 yield.

online iterator를 list()로 변환하지 마라.
online 전체를 미리 읽지 마라.
다음 point를 요청하기 전에 현재 prediction을 내보낸다.

`infer()` 안에서 file write 금지.

model은 train에서 저장.
infer 시작 시 load.

multiprocessing을 사용한다면 model/state isolation과 memory multiplication을 고려한다.

INFER_PARALLELISM은 무작정 최대치로 설정하지 않는다.

1 / 2 / 4 정도를 latency와 RAM 기준으로 비교할 수 있다.

Windows local에서는 official runner의 parallelism behavior가 cloud와 다를 수 있으므로 차이를 문서화한다.

======================================================================
22. Determinism hard gate
=========================

같은 model artifact와 같은 input으로 infer를 두 번 수행했을 때 동일 prediction을 내야 한다.

seed 고정.

random training library:

* Python
* NumPy
* sklearn
* XGBoost
* LightGBM

가능한 deterministic option 설정.

determinism smoke를 최소 2회.

prediction max abs diff 기록.

official local tester의 determinism check도 가능한 경우 사용한다.

determinism FAIL 모델은 champion 불가.

======================================================================
23. 실패를 버리지 말고 ledger에 남겨라
==========================

모든 실험은:

`experiments/results/EXPERIMENT_LEDGER.csv`

에 append.

필드 예:

experiment_id
timestamp
git_commit
feature_version
feature_families
model
params
sampling_policy
fold
ts_auc
worst_fold
latency
memory
status
notes

실패 실험도 기록.

예:
OOM
too_slow
no_gain
fold_instability
leakage_fail
determinism_fail

동일 실패를 반복하지 않게 한다.

======================================================================
24. 자동 연구 루프
============

기본 파이프라인이 완성된 뒤 남은 시간을 idle하지 마라.

아래 순서로 반복한다.

STEP A
현재 champion의 weakest fold / weakest time slice 확인.

STEP B
원인을 feature family 수준으로 가설화.

STEP C
변경사항을 1~2개 정도로 제한한 candidate 생성.

STEP D
Development CV.

STEP E
성능 + worst fold + latency 비교.

STEP F
개선이 재현되면 promote.
아니면 reject.

STEP G
ledger 기록.

그리고 반복.

한 번에 hyperparameter 100개 brute force하지 마라.

Random search를 무작정 크게 돌리지 마라.

초기 프로젝트에서는 feature understanding > massive tuning.

======================================================================
25. 과적합 방지 규칙
=============

금지:

* seal 보고 feature 수정
* reduced test 반복 조회
* fold seed shopping
* 수십 개 CV split 중 좋은 split만 보고
* label leakage
* suffix leakage
* online total length feature
* time-only score trick
* dataset_id target encoding
* validation 결과 본 뒤 gate 완화
* validation series row를 train에 섞기
* test reduced target을 training에 사용
* leaderboard 점수에 맞춘 무한 튜닝

dataset_id는 join/split에만 사용.
feature로 입력 금지.

time_online 자체는 실험적으로 사용할 수 있지만:
TS-AUC는 같은 t끼리 비교하므로 순수 time feature는 유효 신호가 아니라는 점을 명시한다.

time과 evidence interaction이 필요한 경우에도 결과를 조심스럽게 검증한다.

======================================================================
26. LOCAL_FINAL_SEAL 정책
=======================

이번 10시간 run 동안 LOCAL_FINAL_SEAL은 열지 않는다.

목표는 첫날에 최고 점수를 인증하는 것이 아니다.

목표는 앞으로 연구할 수 있는 신뢰 가능한 개발 체계를 만드는 것이다.

따라서 최종 champion candidate는:

DEVELOPMENT CV 기준 champion

으로만 표기.

`FINAL_CHAMPION`이라는 이름을 쓰지 마라.

대신:

`DEV_CHAMPION_CANDIDATE`

사용.

Seal 개봉은 다음 별도 세션에서 사용자와 전략을 정한 후 한다.

======================================================================
27. 10시간 운용 스케줄
===============

정확한 시간보다 작업 완료를 우선하되 wall-clock을 관리한다.

대략:

0:00–1:00
bootstrap / official contract / data acquisition

1:00–2:00
data audit / immutable manifest / split lock

2:00–3:00
official scorer parity / leakage tests / baseline reproduction

3:00–4:30
streaming feature bank A/B/C/D

4:30–6:00
baseline ladder + feature-family tests

6:00–7:30
supervised model ladder

7:30–8:30
ablation / fracture / latency optimization

8:30–9:15
focused iterative improvements

9:15–10:00
full reproducibility check / cloud-style smoke / documentation / handoff

작업이 빨라지면 연구 루프를 더 돈다.

작업이 늦어지면:
deep model보다
scorer
leakage guard
split
feature engine
baseline
deployment contract
를 우선한다.

======================================================================
28. 실행 중 checkpoint
===================

각 주요 phase 종료 후:

`checkpoints/CHECKPOINT_<NN>.json`

생성.

포함:

* current git commit
* finished tasks
* active candidate
* best dev score
* unresolved bugs
* next action
* elapsed time
* artifact paths

장기 실행이 중간에 죽어도 이어갈 수 있게 한다.

모델 training 중에도 중요한 결과는 즉시 disk에 저장.

단 raw secret이나 credential 저장 금지.

======================================================================
29. 최소 산출물
==========

10시간 종료 시 아래 파일들이 존재하도록 한다.

00_START_HERE.md

01_PROJECT_STATUS.json

02_OFFICIAL_COMPETITION_CONTRACT.md

03_RAW_DATA_MANIFEST.json

04_DATA_AUDIT.md

05_DATA_AUDIT.json

06_LABEL_CONTRACT.md

07_LEAKAGE_CONTRACT.md

08_SPLIT_LOCK.json

09_SCORER_PARITY.json

10_BASELINE_RESULTS.csv

11_FEATURE_CATALOG.csv

12_MODEL_RESULTS.csv

13_CV_RESULTS.csv

14_FEATURE_ABLATIONS.csv

15_TIME_FRACTURE_REPORT.csv

16_LATENCY_BENCHMARK.json

17_DETERMINISM_REPORT.json

18_DEPLOYMENT_SMOKE.json

19_DEV_CHAMPION_CANDIDATE.json

20_EXPERIMENT_LEDGER.csv

21_LIMITATIONS.md

22_NEXT_SESSION_PLAN.md

23_REPRODUCTION_COMMANDS.md

24_SOURCE_MANIFEST.json

그리고:

`SRC_SNAPSHOT_<timestamp>.zip`

생성.

ZIP에는:

* source code
* configs
* tests
* small reports
* manifests
* experiment summaries

포함.

대규모 raw competition data는 ZIP에 넣지 않는다.

개인 credential/token도 절대 포함하지 않는다.

======================================================================
30. 00_START_HERE.md는 다음 질문에 바로 답해야 한다
======================================

사용자가 아침에 이것 하나만 읽어도 상황을 이해할 수 있어야 한다.

한국어로 작성.

반드시:

1. 대회 데이터를 정상 확보했는가?
2. 데이터가 몇 series인가?
3. break / no-break 비율은?
4. 데이터에서 중요한 특징은 무엇이었나?
5. scorer parity는 PASS인가?
6. leakage guard는 PASS인가?
7. split은 어떻게 봉인했나?
8. official baseline은 얼마였나?
9. 우리가 만든 baseline 중 최고는?
10. supervised candidate 중 최고는?
11. DEV CV TS-AUC는?
12. fold별 점수는?
13. worst fold는?
14. 어떤 feature family가 가장 유효했나?
15. 어떤 feature family가 쓸모없었나?
16. inference latency는?
17. determinism PASS인가?
18. LOCAL_FINAL_SEAL을 열었는가? → 반드시 NO가 정상.
19. reduced local test를 몇 번 사용했나?
20. 현재 가장 큰 병목은?
21. 다음 세션에서 가장 먼저 할 3가지.
22. 현재 결과를 Crunch에 제출 가능한 형태까지 만들었는가?

점수를 과장하지 마라.
실패는 실패라고 써라.

======================================================================
31. DEV champion 선택 기준
======================

점수 하나로 정하지 않는다.

우선순위:

1. leakage PASS
2. scorer parity PASS
3. deterministic PASS
4. valid streaming contract
5. median CV TS-AUC
6. worst fold
7. fold stability
8. early-detection diagnostics
9. latency
10. complexity

아주 작은 mean gain 때문에 worst fold가 크게 무너지면 자동 승격하지 않는다.

winner를 애매하게 여러 개 남기지 말고:

PRIMARY_DEV_CANDIDATE
BACKUP_FAST_CANDIDATE
BACKUP_ROBUST_CANDIDATE

최대 3개 정도로 정리한다.

======================================================================
32. 첫 세션에서 하지 말 것
=================

이번에는 다음을 우선순위에서 제외한다.

* Transformer 대규모 연구
* foundation model
* LSTM/GRU 대규모 sweep
* 2025 winning solution 복제
* 외부 synthetic dataset 대규모 수집
* leaderboard chasing
* 수백~수천 hyperparameter search
* ensemble 20개
* 복잡한 neural architecture

단 기본 feature + shallow model이 예상보다 빨리 완전히 포화되고
8시간 이후에도 충분한 시간이 남는 경우에만
작은 temporal neural baseline 하나 정도를 exploratory로 수행할 수 있다.

그 경우도 PRIMARY pipeline을 변경하지 않는다.

======================================================================
33. 공식 cloud tester
===================

가능하면 공식 `crunch_tools.test()` 흐름을 사용해 최종 smoke test를 한다.

다음 확인:

* train 실행
* model artifact 생성
* infer streaming
* prediction row count
* prediction dtype
* [0,1]
* NaN 없음
* Inf 없음
* deterministic
* correct `(id,time)` index
* prediction column
* no iterator misuse

official runner와 최대한 동일한 환경에서 smoke.

문제가 발생하면 그 원인을 고치고 다시 테스트한다.

======================================================================
34. 자동 디버깅 규칙
=============

에러 발생 시:

1. traceback 저장
2. 원인 분류
3. 최소 재현
4. 수정
5. unit test 추가
6. 전체 relevant test 재실행

같은 버그가 두 번 발생하지 않게 한다.

OOM:
chunking / dtype / cache 구조 개선.

속도:
profiler로 hot path 확인.

성능 부진:
무작정 model complexity를 늘리지 말고 feature family와 normalization 분석.

CV instability:
특정 fold를 삭제하지 말고 이유를 분석.

======================================================================
35. 코드 품질
=========

연구용이지만 spaghetti script 하나에 전부 넣지 않는다.

하지만 지나친 abstraction도 피한다.

핵심:

StreamingFeatureState
TS-AUC scorer
dataset loader
split manager
experiment runner
model wrapper

정도로 명확히 분리.

함수와 변수명은 읽기 쉽게.

핵심 statistical formula와 causality contract에는 주석.

모든 config는 코드에 흩어진 magic constant보다 config 파일 또는 dataclass 중심으로 관리한다.

======================================================================
36. 테스트
=======

최소:

* scorer tests
* label contract tests
* prefix invariance
* no future length access
* deterministic split
* deterministic feature
* synthetic mean break
* synthetic variance break
* synthetic dependence break
* no-break
* single outlier
* model save/load parity
* offline replay vs live streaming parity
* prediction range
* no NaN/inf

offline replay와 streaming inference가 동일 prefix에서 동일 feature/prediction을 내야 한다.

이것은 hard gate다.

======================================================================
37. 이번 10시간의 성공 기준
==================

성공은 leaderboard 고득점이 아니다.

다음이 달성되면 성공이다.

PASS:

* 공식 데이터 확보 또는 인증 blocker 명확화
* 데이터 audit
* raw manifest
* 정확한 TS-AUC
* series-level CV
* untouched local seal
* leak-proof streaming feature engine
* official baseline reproduction
* 최소 4개 break feature family
* baseline ladder
* 최소 2~3 supervised candidates
* ablation
* temporal fracture
* latency benchmark
* deterministic inference
* Crunch-compatible smoke
* reproducible handoff

그리고 가능하면:

official EWMA보다 명확히 좋은 DEVELOPMENT CV candidate를 만든다.

하지만 개선이 안 되더라도 결과를 조작하거나 seal을 열지 않는다.

======================================================================
38. 마지막 독립 audit
================

종료 전 스스로 연구자 역할에서 auditor 역할로 바꿔 검토한다.

질문:

* 미래를 본 feature가 하나라도 있는가?
* online total length를 사용했는가?
* target/tau가 feature code에 전달되는가?
* 같은 series가 train/val 양쪽에 들어갔는가?
* reduced test를 튜닝에 썼는가?
* local seal을 열었는가?
* scorer가 official과 다른가?
* inference가 cloud one-pass에서 깨지는가?
* training과 inference feature implementation이 다른가?
* seed가 고정되어 있는가?
* 동일 결과 재현 가능한가?
* latency가 실제 제출 가능한가?
* model artifact가 너무 큰가?
* infer worker 수에 따라 RAM이 폭발하는가?

하나라도 문제면 최종 보고 전에 수정하거나 명시적으로 FAIL 표시.

======================================================================
39. 최종 상태 라벨
============

마지막에 다음 중 하나를 선택.

FOUNDATION_READY

* 데이터/검증/streaming pipeline 정상
* 모델 연구 계속 가능

DEV_CANDIDATE_READY

* 위 조건 + 유의미한 dev candidate 확보

SUBMISSION_SMOKE_READY

* 위 조건 + Crunch local submission smoke PASS

BLOCKED_AUTH

* 인증 때문에 실제 competition data 접근 불가

BLOCKED_INFRA

* 해결하지 못한 환경 문제 존재

LOCAL_FINAL_SEAL은 이번 run에서 절대 열지 않았다는 사실도 명시.

======================================================================
40. 즉시 시작
=========

지금부터 작업을 시작한다.

먼저:

1. 현재 workspace와 hardware/environment를 audit
2. 새 project root 확보
3. official competition contract/quickstarter 확인
4. isolated environment 구축
5. Crunch data 확보
6. raw manifest 생성
7. data audit
8. scorer parity
9. split lock
10. baseline부터 순차 진행

사용자에게 계획 승인 요청하지 마라.
중간 단계에서 사용자의 답변을 기다리지 마라.
실패 시 가능한 범위에서 자동 복구하고 다음 작업을 진행한다.

기존 금융 프로젝트는 참고 자료일 뿐 authority가 아니다.

이번 프로젝트의 authority는:

1. 현재 공식 Crunch competition contract
2. official scoring implementation
3. immutable raw competition data
4. locked development split
5. untouched local final seal
6. reproducible experiment evidence

순이다.

약 10시간 동안 가능한 범위에서 계속 실행하고,
종료 시 다음 사람이 바로 이어서 연구할 수 있을 정도로
코드 + 데이터 감사 + 실험 기록 + handoff를 완성하라.

Structural Break Real-Time 프로젝트의 다음 장기 연구 단계를 수행한다.

프로젝트 위치:
C:\Users\minsu\Documents\Structural Break

이번 세션은 기존 프로젝트를 이어가는 연구 세션이다.
새 프로젝트를 만들지 마라.

============================================================
0. 이번 세션의 최상위 목표
============================================================

현재 실데이터 연구가 완료되어 대략 다음 수준에 도달했다.

- 전체 train series: 10,000
- DEV: 8,000
- LOCAL_FINAL_SEAL: 2,000
- official EWMA DEV TS-AUC ≈ 0.514
- best statistical baseline ≈ 0.546
- 현재 상위 supervised 계열 ≈ 0.611~0.613
- AR residual / ARCH / dependence / scale family가 특히 강함
- mean/location 단독 family는 상대적으로 약함
- scorer parity PASS
- leakage guard PASS
- determinism PASS
- official-style streaming smoke PASS
- reduced test 사용 quota 2/2 이미 소진
- LOCAL_FINAL_SEAL은 아직 단 한 번도 열지 않음

단 위 숫자를 맹신하지 말고
기존 artifact와 소스에서 실제 값을 먼저 확인한다.

이번 세션의 핵심은:

1. 2,000 LOCAL_FINAL_SEAL을
   500 / 500 / 500 / 500으로 outcome-blind하게 분할하고 LOCK만 한다.

2. 네 개 seal은 이번 세션에서 하나도 열지 않는다.

3. 기존 8,000 DEV만 이용해
   현재 모델의 일반화 안정성을 훨씬 강하게 검사한다.

4. 현재 0.61 벽을 넘기 위해
   단순 feature 추가가 아니라
   새로운 sequential/change-point statistic generation을 연구한다.

5. 최소 10시간의 실제 연구를 수행한다.

6. 최소 10시간 연구가 끝난 뒤
   추가 45~60분을 final audit / reproducibility / handoff에 사용한다.

============================================================
1. 매우 중요한 시간 규칙
============================================================

이전처럼 `--hours 10`이 단순 최대 timeout이 되어
정해진 후보를 다 돌리고 1~2시간 만에 종료하면 안 된다.

이번에는 실제 wall-clock 기반 연구 scheduler를 구현한다.

예:

research_start = monotonic()
minimum_research_end = research_start + 10 hours

while monotonic() < minimum_research_end:
    계속 의미 있는 연구를 진행한다.

최소 연구시간 10시간이 지나기 전에는
최종 handoff를 작성하고 종료하지 마라.

단순 sleep으로 시간을 채우는 것은 절대 금지.

후보 목록이 소진되면:
- weakest slice forensic
- 새로운 statistic family
- robustness
- bootstrap
- alternate validation
- latency optimization
- ablation
- calibration
- ensemble
등 다음 연구 backlog로 자동 이동한다.

최소 연구시간:

10시간

그 이후:

45~60분
final validation
reproduction
artifact audit
handoff

따라서 정상 목표 총 wall-clock은
약 10시간 45분~11시간이다.

환경/Codex 자체 제한으로 프로세스가 중단될 위험에 대비하여
모든 phase마다 checkpoint를 남긴다.

============================================================
2. 기존 결과를 먼저 읽어라
============================================================

다음을 전부 읽고 현재 상태를 복원한다.

특히:

00_GPT_READ_FIRST.md
01_GPT_HANDOFF.md
BASELINE_RESULTS_REAL.csv
MODEL_RESULTS_REAL.csv
CV_RESULTS_REAL.csv
FEATURE_ABLATIONS_REAL.csv
FEATURE_STANDALONE_RANKING.csv
FEATURE_MARGINAL_CONTRIBUTIONS.csv
SAMPLING_WEIGHTING_RESULTS.csv
TIME_FRACTURE_REAL.csv
BREAK_AGE_DIAGNOSTICS.csv
NEXT_SESSION_PLAN.md
LIMITATIONS.md

그리고:
src/
scripts/
configs/
tests/
experiments/
artifacts/

를 확인한다.

현재 champion 및 상위 candidate의 정확한 이름과 score를
artifact에서 직접 읽어온다.

특히 다음 세 유형을 복원한다.

PRIMARY_DEV_CANDIDATE
HIGHEST_MEAN_CANDIDATE
ROBUST_BACKUP_CANDIDATE

기존 champion configuration을 수정하지 말고
frozen reference로 보존한다.

============================================================
3. LOCAL SEAL 2000 → 500 × 4 분할
============================================================

가장 먼저 수행한다.

현재 LOCAL_FINAL_SEAL 2,000개 ID만 대상으로
target/tau를 절대 읽지 않고 deterministic hash를 이용해:

SEAL_A = 500
SEAL_B = 500
SEAL_C = 500
SEAL_D = 500

으로 나눈다.

중요:

- label 확인 금지
- tau 확인 금지
- prediction 금지
- TS-AUC 계산 금지
- feature diagnostic 금지
- metadata 기반 성능 추론 금지

dataset_id만 사용한다.

기존 LOCAL_FINAL_SEAL membership을 변경하면 안 된다.

즉:

기존 2,000 set
    ↓
그 내부에서만
500 / 500 / 500 / 500

이다.

deterministic cryptographic hash를 이용한다.

예:
SHA256(project_frozen_salt + dataset_id)

salt 역시 고정해 기록.

결과:

SEAL_PARTITION_500x4_LOCK.json
SEAL_A_IDS.txt
SEAL_B_IDS.txt
SEAL_C_IDS.txt
SEAL_D_IDS.txt

생성.

각 파일 SHA256 저장.

그리고:

SEAL_A_OPENED = false
SEAL_B_OPENED = false
SEAL_C_OPENED = false
SEAL_D_OPENED = false

명시.

============================================================
4. Seal 접근 자체를 코드로 막아라
============================================================

이번 연구에서 실수로라도 seal label을 읽지 못하게 guard를 만든다.

예:

SealAccessError

개발용 loader는 기본적으로:

DEV IDs only

를 반환.

seal ID가 training/evaluation/prediction 대상으로 들어가면
즉시 hard fail.

테스트:

- seal ID가 training rows에 없음
- seal ID가 validation rows에 없음
- seal ID가 OOF prediction에 없음
- seal ID가 experiment ledger에 없음
- seal ID label access attempt → fail

최종 audit에서 자동 확인.

결과:

SEAL_GUARD_REPORT.json

============================================================
5. reduced test도 완전히 봉인
============================================================

이전 세션에서 reduced test 사용 횟수 2/2가 이미 끝났다.

이번 세션에서는:

X_test.reduced
y_test.reduced

성능 평가에 사용 금지.

이미 확보한 이전 reduced 결과를 보고
candidate를 선택하거나 수정하지 않는다.

REDUCED_TEST_NEW_USAGE = 0

이어야 한다.

============================================================
6. Public leaderboard 제출도 하지 마라
============================================================

이번 세션에서는 실제 Crunch submission을 자동으로 하지 않는다.

submission artifact 생성 및 local smoke는 가능.

하지만:

- upload
- submit
- leaderboard score 조회

금지.

Public LB도 unseen 정보이므로
별도 전략 세션에서 사용할 것이다.

이번 연구는 DEV 8,000 안에서 끝낸다.

============================================================
7. 현재 8,000 DEV의 검증력을 극대화
============================================================

기존 locked 5-fold CV를 primary development authority로 유지한다.

하지만 같은 split 하나에만 의존하지 않는다.

새로운 candidate 연구를 시작하기 전에
다음 robustness validation suite를 미리 freeze한다.

A. PRIMARY LOCKED 5-FOLD
기존 split 그대로.

B. ALT HASH SPLIT #1

C. ALT HASH SPLIT #2

D. ALT HASH SPLIT #3

중요:

alternate split의 seed/hash rule은
새 모델 결과를 보기 전에 지금 고정한다.

좋은 split을 선택하면 안 된다.

alternate split은 champion selection authority가 아니라
split sensitivity/stability diagnostic이다.

모든 모델은 같은 alternate splits를 사용한다.

결과:

ROBUSTNESS_SPLIT_LOCK.json

============================================================
8. paired series bootstrap 구축
============================================================

현재 후보들 사이의 차이는
0.001~0.002 수준이므로
단순 mean 차이만 보고 우열을 정하면 안 된다.

OOF predictions를 이용해서
series 단위 paired bootstrap evaluator를 만든다.

row bootstrap 금지.

resample unit:

dataset_id / series

이다.

초기 screening:
500 bootstrap replicates

finalists:
2,000 bootstrap replicates

각 candidate pair:

delta_TS_AUC
median_delta
mean_delta
2.5 percentile
97.5 percentile
P(delta > 0)

계산.

최소 비교:

현재 PRIMARY
vs
현재 highest-mean

현재 PRIMARY
vs
현재 robust backup

새 generation
vs
현재 PRIMARY

bootstrap 코드 자체가 너무 느리면
TS-AUC의 time-stratified 구조를 보존하는
optimized implementation을 만든다.

통계 구조를 바꾸는 approximation은 금지.

============================================================
9. historical-only regime stress test
============================================================

Seal 대신 강한 외삽 stress를 만든다.

DEV 8,000 series의 historical clean segment만 이용한다.

target/tau를 절대 clustering input으로 사용하지 않는다.

historical에서:

location
scale
MAD
skew
kurtosis
tail rate
ACF1
ACF2
ACF4
ACF8
small AR coefficients
innovation variance
spectral/coarse frequency signature
historical length

등 outcome-blind summary를 만든다.

standardize 후
작은 K의 deterministic clustering.

예:
K=5

K 값은 model score를 보고 변경하지 않는다.

그 후:

Leave-One-Historical-Regime-Out

stress evaluation.

즉 cluster 하나를 완전히 validation domain으로 두고
나머지에서 학습.

이 결과는 primary CV score 대신
generalization stress로 사용한다.

각 후보:

regime 1
regime 2
...
regime K

TS-AUC

그리고 worst regime 기록.

============================================================
10. mechanism fracture diagnostic
============================================================

training에서는 tau가 있으므로
evaluation-only로 break 후 변화를 분석할 수 있다.

주의:

이 정보는 모델 feature/training input에 절대 넣지 않는다.

오직 diagnostic.

각 break series에 대해
pre/post statistical signature를 분석하여
post-hoc mechanism category를 만든다.

예:

MEAN_DOMINANT
SCALE_DOMINANT
DEPENDENCE_DOMINANT
TAIL_DOMINANT
MIXED
WEAK_OR_AMBIGUOUS

classification rule은
모델 score 보기 전에 freeze한다.

이건 true DGP label이라고 주장하지 않는다.

오직 statistical signature slice다.

각 candidate에 대해
mechanism별 TS-AUC를 기록.

어떤 candidate가
특정 mechanism 하나만 과도하게 먹고
다른 family를 파괴하는지 검사한다.

============================================================
11. 현재 0.61 모델에 대한 forensic
============================================================

새 feature를 만들기 전에
현재 상위 3개 모델이 왜 다른지 분석한다.

다음 비교:

PRIMARY
HIGHEST_MEAN
ROBUST_BACKUP

각각:

- fold별 OOF prediction correlation
- disagreement series
- disagreement time points
- mechanism별 차이
- historical regime별 차이
- break age별 차이
- no-break false positive 차이
- AR order sensitivity
- ARCH evidence sensitivity

상위 모델들이 거의 동일한 prediction을 내면
ensemble로 얻을 수 있는 이득이 제한적임을 기록.

반대로 disagreement가 의미 있게 complementary하면
후반 ensemble 연구 후보로 유지.

============================================================
12. 이번 세션의 핵심 신규 연구 방향
============================================================

단순 feature 100개 추가가 목표가 아니다.

새로운 sequential evidence generation을 연구한다.

우선순위:

1. historical whitening
2. innovation normalization
3. sequential likelihood evidence
4. tau/change-age integration
5. dependence change evidence
6. scale/change evidence
7. robust distribution evidence
8. lightweight supervised combination

============================================================
13. Generation W1 — Historical whitening
============================================================

현재 AR residual이 강하므로 더 정교하게 일반화한다.

historical segment에서만 모델을 fit.

AR orders 후보를 사전 고정:

AR1
AR2
AR4
AR8
AR12

필요한 경우 stable fitting guard.

각 series마다 historical에서 한 번 fit.

online에서는 recursive prediction.

innovation:

e_t = x_t - xhat_t

그리고 historical innovation 기준으로:

- standard deviation normalization
- MAD normalization
- ARCH conditional scale normalization
- EWMA conditional scale normalization

후보 생성.

online에서 AR refit 금지.

historical을 반복 scan 금지.

각 point O(p) 이하.

============================================================
14. Generation W2 — innovation empirical CDF / Gaussianization
============================================================

historical innovation distribution을 이용해
online residual을 rank-normalize한다.

historical innovation ECDF:

u_t = F_hist(e_t)

clip:
epsilon <= u <= 1-epsilon

z_t = Phi^-1(u_t)

이렇게 하면 series마다
tail/scale 차이를 어느 정도 공통 null space로 옮길 수 있다.

비교:

raw innovation
standardized innovation
robust standardized innovation
ECDF uniform score
Gaussianized innovation z

historical ECDF lookup이 매 point O(H)가 되면 안 된다.

sorted historical innovations + binary search 사용 가능.

============================================================
15. Generation S1 — mean sequential evidence
============================================================

whitened/gaussianized innovation에서:

two-sided CUSUM
Page-Hinkley
EWMA shift
signed accumulated evidence

개발.

여러 scale/hazard를 무작정 폭발시키지 않는다.

사전 정의 작은 grid.

예:

fast
medium
slow

정도.

============================================================
16. Generation S2 — scale sequential evidence
============================================================

현재 B/ARCH 계열이 강했으므로 중요.

innovation z에 대해:

z² - 1
|z|
log(z² + epsilon)

등을 이용한:

variance CUSUM
scale Page-Hinkley
EWMA log-scale shift
conditional scale mismatch

개발.

variance increase뿐 아니라
variance decrease에도 대응.

two-sided evidence.

============================================================
17. Generation S3 — dependence sequential evidence
============================================================

dependence가 강한 family이므로 집중.

whitened innovation에서:

z_t * z_(t-1)
z_t * z_(t-2)
z_t * z_(t-4)
z_t * z_(t-8)

historical expected value 대비
recursive deviation.

또:

prediction error correlation
absolute innovation correlation
squared innovation correlation

등도 탐색.

각 lag family를 별도로 만들고
ablation.

============================================================
18. Generation S4 — distribution/tail evidence
============================================================

매 point full KS/AD 계산 금지.

incremental approximation 사용.

historical ECDF space에서
under null이면 대략 Uniform(0,1)이어야 한다.

이를 이용해:

tail exceedance evidence
central-vs-tail mass imbalance
signed tail imbalance
coarse CDF bin deviation

예:
8-bin / 16-bin 정도.

running EWMA bin counts.

CvM/AD-like approximate statistic을
O(number_of_bins) 또는 O(1) incremental로 계산.

============================================================
19. Generation G — sequential GLR
============================================================

중요 신규 방향.

현재 시점까지
“어느 과거 시점에서 break가 발생했을 가능성”을
한 개 고정 window statistic보다 더 직접적으로 평가한다.

하지만 exact O(T²) GLR 금지.

candidate change age grid를 제한한다.

예:

1
2
4
8
16
32
64
128

또는 exponentially spaced ages.

각 candidate age에 대해
pre/post sufficient statistic을 streaming으로 유지하거나
ring buffer / cumulative sum으로 저비용 계산.

대상:

mean shift GLR
variance shift GLR

가능하면 whitened z space에서 계산.

최종 evidence는:

max over candidate tau
logsumexp over candidate tau

둘 다 비교.

특히 `max`는 multiple-testing inflation 가능성이 있으므로
null normalization 필요 여부 분석.

============================================================
20. Generation BAYES-LITE — tau-integrated evidence
============================================================

완전한 무거운 BOCPD를 무작정 구현하지 않는다.

먼저 lightweight approximation 연구.

목표:

P(break has happened by t | observations)

와 논리적으로 가까운 sequential evidence를 만든다.

candidate change ages를
exponential grid로 제한하고,

각 hypothetical change time에 대해
likelihood ratio를 유지.

그리고:

logsumexp(
    log prior_tau
    +
    log likelihood ratio
)

형태의 evidence.

mean shift amplitude도
몇 개 grid로 integrate하거나
analytic conjugate approximation이 가능하면 사용.

variance shift도 별도.

중요:

true tau는 절대 inference에 사용하지 않는다.

training tau는 label 생성/평가만.

============================================================
21. BOCPD는 exploratory only
============================================================

시간이 충분하면 truncated Bayesian Online Change Point Detection을
작은 experimental candidate로 구현할 수 있다.

단:

- max run length 제한
- pruning
- vectorized update
- latency benchmark

필수.

10M-point inference에 부적합하면
연구 결과만 남기고 primary candidate에서 제외.

복잡한 BOCPD 때문에 몇 시간 날리지 마라.

============================================================
22. ARCH 계열 확장
============================================================

현재 ARCH1이 강했다면
작은 범위에서만 확장.

ARCH1
ARCH2
EWMA conditional variance

비교.

full GARCH optimization을 series마다 무겁게 돌리는 방향은
초기 우선순위가 아니다.

historical fit cost와 online latency 측정.

============================================================
23. Cross-order AR mismatch
============================================================

AR2 / AR4 / AR8 / AR12 등이
서로 다른 scale의 dependence structure를 잡을 수 있다.

다음 evidence 후보:

AR4 error vs AR8 error
AR8 error vs AR12 error
short-order prediction advantage
long-order prediction collapse

하지만 pairwise explosion 금지.

가장 논리적인 2~3개만.

============================================================
24. metric-aligned supervised objective
============================================================

기존 binary classification baseline을 유지하되
TS-AUC와 더 맞는 objective를 연구한다.

TS-AUC는 동일 online time t에서
series들의 ranking을 평가한다.

따라서 다음을 비교한다.

A. ordinary binary LightGBM

B. existing weighting

C. time-pair contribution weighting

D. pairwise ranking approximation

E. LightGBM ranking objective exploratory

ranking group 후보:

same absolute online t

각 t가 query/group 역할.

단 구현 전에
TS-AUC와 논리적으로 일치하는지 확인.

NDCG 최적화 자체가 목표가 아니다.

pairwise ordering 개선이 실제 TS-AUC에 도움이 되는지만
DEV CV로 확인.

ranking model이 복잡하거나 성능이 떨어지면 폐기.

============================================================
25. time contribution weighting
============================================================

각 t에서 TS-AUC의 weight는
roughly positive-negative pair count와 관련된다.

training weighting 후보를
사전에 소수만 정의.

예:

W0 uniform
W1 inverse series density correction
W2 n_pos(t)*n_neg(t) proportional
W3 class-balanced within t + time contribution

주의:

validation target distribution을 보고
weight를 최적화하지 않는다.

DEV train fold에서 계산 가능한 label statistics만 사용.

============================================================
26. evidence memory 연구
============================================================

target은 break 발생 이후 계속 positive이므로
instantaneous detector만으로는 부족할 수 있다.

비교:

raw evidence
cumulative max
decayed max
leaky accumulator
posterior-like evidence memory
hysteresis

false alarm persistence에 주의.

no-break series false-positive fracture를 반드시 확인.

============================================================
27. candidate 생성 규칙
============================================================

한 실험에서 10가지 요소를 동시에 바꾸지 마라.

각 experiment는 명시적으로:

HYPOTHESIS
CHANGE
EXPECTED_EFFECT
RISKS

를 기록.

예:

HYPOTHESIS:
AR-whitened ECDF Gaussianization은
series-specific heavy-tail variation을 줄여
dependence detector generalization을 개선한다.

CHANGE:
AR8 residual → historical ECDF → Gaussianized z

EXPECTED:
regime robustness 개선

RISK:
historical ECDF sampling noise

형태.

============================================================
28. 2-stage screening
============================================================

모든 후보를 바로 full 5-fold로 돌려
10시간을 낭비하지 않는다.

STAGE SCREEN:

기존 DEV의 사전 고정 subset/folds를 이용한
cheap screening.

단 screening subset 역시 지금 미리 고정하고
candidate score를 본 뒤 변경 금지.

유망 후보만:

FULL PRIMARY 5-FOLD

진입.

하지만 screening 결과가
최종 성능 authority는 아니다.

============================================================
29. promotion gate를 사전 고정
============================================================

기존 PRIMARY 대비 새 candidate promotion 조건을
결과 보기 전에 고정.

추천:

FULL primary CV mean delta >= +0.0015

AND

median non-worse

AND

worst fold degradation > -0.0015 이내

AND

paired bootstrap P(delta > 0) >= 0.85

AND

3개 alternate split/stress 평가 중
최소 2개에서 non-negative

AND

historical-regime worst collapse 없음

AND

major mechanism slice catastrophic collapse 없음

AND

latency acceptable

이 조건을 만족해야
NEW_DEV_CANDIDATE로 승격.

============================================================
30. NEW GENERATION gate
============================================================

단순 candidate 승격과
“seal을 열 가치가 있는 새로운 generation”은 다르다.

Seal opening recommendation은 이번 세션에서도
실제 실행하지 않고 문서에 recommendation만 한다.

NEW_GENERATION_SEAL_WORTHY 조건:

기존 PRIMARY 대비

FULL DEV mean delta >= +0.004

또는

mean delta >= +0.003
AND
bootstrap probability >= 0.95
AND
robustness suite 전반 명확한 개선
AND
새 mechanism/statistic에서 독립적 gain

정도.

이 gate를 결과 본 뒤 낮추지 마라.

============================================================
31. Seal은 이번 세션에서 절대 열지 않는다
============================================================

아무리 좋은 candidate가 나와도:

SEAL_A
SEAL_B
SEAL_C
SEAL_D

성능을 계산하지 마라.

대신:

SEAL_OPEN_RECOMMENDATION.md

만 작성.

내용:

- 지금 seal을 열 필요 있는가?
- YES/NO
- 어떤 candidate generation 때문에 그런가?
- DEV improvement
- bootstrap
- stress evidence
- seal을 열 경우 어떤 gate만 볼 것인가?

이번 정상 결과:

ALL_SEALS_UNOPENED = TRUE

============================================================
32. ensemble 연구
============================================================

후반부에 current top models/new statistics가
실제로 complementary한 경우에만 ensemble 연구.

prediction correlation / disagreement 먼저 확인.

후보:

simple mean
rank average
fixed weighted average

weight를 continuous optimize해서 DEV에 맞추지 마라.

예:

50/50
67/33
33/67

정도 사전 grid.

ensemble gain이 +0.0002 같은 수준이면
복잡성 대비 가치 낮다고 기록.

============================================================
33. calibration
============================================================

TS-AUC는 ranking metric이므로
calibration 자체는 primary objective가 아니다.

monotonic transform은 AUC를 바꾸지 않는다.

따라서 Platt/isotonic calibration에
시간을 낭비하지 않는다.

단 multiple detector를 합치는 경우
scale alignment 목적으로
fold-train-only normalization은 가능.

============================================================
34. early detection은 secondary
============================================================

기존 결과에서 break age 0~4 성능이 약했다.

하지만 전체 metric weight에서
early 구간의 비중이 작을 수 있다.

따라서 early detection만 올리려고
전체 TS-AUC를 희생하지 마라.

Primary authority:
TS-AUC.

Secondary diagnostics:
early break age
false alarm
delay.

============================================================
35. latency hard gate
============================================================

모든 finalist에 대해:

feature state update
model prediction
combined latency

측정.

median
p95
p99

그리고:

1M point
10M point

예상 시간.

historical initialization cost도 별도.

현재 model보다
0.001 좋아지지만
latency 50배 증가하는 candidate는
primary promotion 시 조심.

============================================================
36. RAM / hardware
============================================================

실제 하드웨어를 자동 탐지한다.

사용 가능한 환경이 대략:

- Core Ultra 7 258V
- RAM 32GB
- RTX 3080 Ti eGPU

일 가능성이 있으나
실제 탐지값을 authority로 한다.

feature generation:
CPU 우선

LightGBM/XGB:
GPU가 실제 이득이면 사용 가능.

GPU 때문에 data transfer overhead가 더 크면 CPU 사용.

RAM 32GB 기준:
- float32
- memmap
- parquet shards
- chunking

사용.

전체 4M+ rows를 중복 DataFrame으로 여러 개 만들지 마라.

============================================================
37. 기존 cache 재사용
============================================================

기존 frozen feature가 동일하면
검증된 cache 재사용 가능.

새 feature generation은 version/hash를 변경.

cache key:

raw data SHA256
split hash
feature config hash
code version

포함.

stale cache 자동 오사용 방지.

============================================================
38. adaptive research loop
============================================================

기본 연구 후보를 모두 실행한 뒤
최소 10시간이 안 지났으면 반드시 계속 연구.

반복:

1. current leader 확인

2. weakest dimension 확인
   - fold
   - regime
   - mechanism
   - break age
   - false positive
   - latency

3. 원인 hypothesis 작성

4. 변경 1~2개

5. screening

6. full CV if justified

7. bootstrap

8. robustness

9. promote/reject

10. ledger/checkpoint

그리고 반복.

============================================================
39. adaptive priority
============================================================

결과에 따라 방향 변경.

CASE A
whitening이 크게 개선
→ innovation normalization + sequential evidence 집중

CASE B
GLR이 크게 개선
→ tau grid / null normalization 개선

CASE C
Bayes-lite가 크게 개선
→ hazard/amplitude prior small predefined grid

CASE D
dependence가 계속 strongest
→ AR innovation/dependency evidence 집중

CASE E
scale strongest
→ conditional variance / squared innovation evidence 집중

CASE F
new statistics standalone는 강하지만 boosting에서 gain 없음
→ redundancy 분석

CASE G
split instability
→ new model tuning보다 generalization 원인 조사

============================================================
40. 금지 사항
============================================================

절대 금지:

SEAL_A open
SEAL_B open
SEAL_C open
SEAL_D open

reduced test 재평가

public LB 제출

dataset_id feature

tau feature

online total future length

future suffix

row-random split

seed shopping

결과 보고 promotion gate 완화

mechanism diagnostic label을 training feature로 사용

post-break info를 online feature에 사용

DEV score를 높이기 위한 무제한 hyperparameter sweep

대형 Transformer

대규모 LSTM/GRU

외부 winning solution 복사

포럼/Reddit high-score 전략을 계속 검색하면서 따라하기

이번 세션은 independent research를 유지한다.

============================================================
41. 최소 10시간 research backlog
============================================================

다음 queue를 모두 만들어 놓는다.

PRIORITY 1
seal partition/guard
robust split lock
bootstrap evaluator

PRIORITY 2
current top forensic

PRIORITY 3
AR whitening

PRIORITY 4
ECDF/Gaussian innovation

PRIORITY 5
mean/scale/dependence sequential CUSUM

PRIORITY 6
GLR

PRIORITY 7
Bayes-lite tau integration

PRIORITY 8
ARCH extension

PRIORITY 9
time-aligned ranking objective

PRIORITY 10
historical-regime stress

PRIORITY 11
mechanism fracture

PRIORITY 12
ensemble

PRIORITY 13
latency optimization

PRIORITY 14
bootstrap finalists

PRIORITY 15
independent rerun/reproduction

따라서 특정 연구가 실패해도
작업할 것이 없다는 이유로 일찍 종료하면 안 된다.

============================================================
42. 대략적인 시간 계획
============================================================

0:00–0:45
기존 결과 복원
seal 500x4 partition
guard
robust split freeze

0:45–1:45
bootstrap evaluator
current champion forensic

1:45–3:15
AR whitening
innovation normalization
ECDF Gaussianization

3:15–5:00
sequential CUSUM / Page-Hinkley
scale/dependence evidence

5:00–6:45
GLR
tau-grid
Bayes-lite

6:45–8:00
supervised combination
ranking/weight objective

8:00–9:00
historical regime stress
mechanism fracture
bootstrap

9:00–10:00
adaptive improvement loop
final candidate refinement

10시간 연구 완료 후:

10:00–10:45/11:00
reproduction
tests
latency
determinism
source manifest
handoff
GPT ZIP

실험 속도에 따라 조정 가능하지만
순수 연구 10시간 미만 종료 금지.

============================================================
43. checkpoint
============================================================

30~60분마다 또는 major experiment 후:

checkpoints/
STAGE_NEXT_CHECKPOINT_<NN>.json

기록:

elapsed_research_seconds
current_candidate
current_best_score
completed_experiments
next_queue
failures
artifact hashes
git commit

resume script 작성.

프로세스가 죽어도
이미 완료한 full CV를 다시 처음부터 반복하지 않도록 한다.

============================================================
44. experiment ledger 강화
============================================================

각 experiment:

experiment_id
parent_id
generation
hypothesis
change
feature_version
model
params
primary_cv_mean
median
worst
delta_vs_primary
bootstrap_p
bootstrap_ci_low
bootstrap_ci_high
alt_split_results
regime_worst
mechanism_worst
latency
status
reason
elapsed_time

기록.

status:

SCREEN_FAIL
FULL_FAIL
ROBUSTNESS_FAIL
PROMOTED
NEW_GENERATION
LATENCY_FAIL
BUG
REDUNDANT

============================================================
45. 최종 후보
============================================================

종료 시 최대:

PRIMARY_DEV_CANDIDATE
ROBUST_DEV_CANDIDATE
FAST_DEV_CANDIDATE

3개.

기존 champion이 여전히 최고라면
억지로 새 모델을 승격하지 않는다.

============================================================
46. 최종 산출물
============================================================

다음 생성/업데이트.

00_GPT_READ_FIRST_NEXT.md
01_GPT_HANDOFF_NEXT.md

SEAL_PARTITION_500x4_LOCK.json
SEAL_GUARD_REPORT.json

ROBUSTNESS_SPLIT_LOCK.json

CURRENT_TOP_FORENSIC.md
CURRENT_TOP_DISAGREEMENT.csv

PAIRED_BOOTSTRAP_RESULTS.csv

HISTORICAL_REGIME_POLICY.md
HISTORICAL_REGIME_RESULTS.csv

MECHANISM_SIGNATURE_POLICY.md
MECHANISM_FRACTURE_RESULTS.csv

WHITENING_RESULTS.csv
INNOVATION_NORMALIZATION_RESULTS.csv

SEQUENTIAL_STAT_RESULTS.csv
GLR_RESULTS.csv
BAYES_LITE_RESULTS.csv

RANKING_OBJECTIVE_RESULTS.csv
WEIGHTING_RESULTS_NEXT.csv

FEATURE_ABLATION_NEXT.csv
MODEL_RESULTS_NEXT.csv
CV_RESULTS_NEXT.csv

TIME_FRACTURE_NEXT.csv
BREAK_AGE_NEXT.csv

LATENCY_NEXT.json
DETERMINISM_NEXT.json
REPRODUCTION_NEXT.json

EXPERIMENT_LEDGER.csv

DEV_CANDIDATES_NEXT.json

SEAL_OPEN_RECOMMENDATION.md

LIMITATIONS_NEXT.md
NEXT_SESSION_PLAN.md
SOURCE_MANIFEST_NEXT.json

그리고:

STRUCTURAL_BREAK_GPT_HANDOFF_NEXT_<timestamp>.zip

생성.

raw data와 credential은 ZIP에 넣지 않는다.

============================================================
47. 00_GPT_READ_FIRST_NEXT.md
============================================================

한국어로 아주 이해하기 쉽게 작성.

반드시:

1. 실제 연구시간
2. 총 experiment 수
3. 기존 Primary score
4. 새 best score
5. delta
6. 5-fold score
7. median
8. worst fold
9. bootstrap CI
10. P(delta > 0)
11. alternate split 결과
12. historical regime worst
13. mechanism별 성능
14. whitening 효과
15. ECDF Gaussianization 효과
16. GLR 효과
17. Bayes-lite 효과
18. ranking objective 효과
19. strongest new idea
20. 실패한 주요 아이디어
21. latency
22. seal A/B/C/D opened 여부
23. reduced new usage
24. public LB submission 여부
25. seal을 지금 열 필요가 있는가?
26. 다음 우선순위 3개

를 숫자로 기록.

============================================================
48. 종료 audit
============================================================

최종적으로 auditor 모드.

확인:

SEAL_A_OPENED == false
SEAL_B_OPENED == false
SEAL_C_OPENED == false
SEAL_D_OPENED == false

REDUCED_NEW_USAGE == 0

PUBLIC_SUBMISSIONS_NEW == 0

DEV/seal contamination == 0

future leakage == 0

scorer parity PASS

offline/stream parity PASS

determinism PASS

selected candidate reproducible PASS

모든 결과 artifact SHA256.

============================================================
49. 핵심 연구 철학
============================================================

현재 0.611~0.613에서
0.0003짜리 gain을 쫓아
DEV를 더 과적합시키는 것이 목표가 아니다.

목표는:

새로운 structural-break evidence engine을 만들어
0.004 이상의 generation-level improvement 가능성을 찾는 것.

특히:

AR whitening
innovation normalization
sequential likelihood
tau integration
dependence change

가 이번 세션의 핵심이다.

seal은 비싼 자산이다.

검증 데이터가 필요하다는 이유만으로
500개를 쉽게 소비하지 마라.

DEV에서 가능한 검증을 최대한 수행하고도
정말 새로운 generation을 인증해야 할 때만
다음 세션에서 seal 사용을 논의한다.

============================================================
50. 지금 실행
============================================================

계획 설명만 출력하고 멈추지 마라.

즉시:

1. 기존 상태 복원
2. seal 500x4 LOCK
3. seal access guard
4. robustness split lock
5. bootstrap evaluator
6. current candidate forensic
7. whitening 연구
8. sequential statistic 연구
9. GLR/Bayes-lite
10. supervised combination
11. robustness
12. adaptive loop

순서로 실제 코드를 작성하고 실행한다.

최소 10시간의 실질적인 연구가 완료되기 전에는
최종 handoff를 작성하고 종료하지 마라.

중간 실패가 발생하면 자동 복구한다.

이번 세션에서도
LOCAL SEAL은 하나도 열지 않는다.
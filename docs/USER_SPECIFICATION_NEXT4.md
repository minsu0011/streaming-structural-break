Structural Break Real-Time 프로젝트 NEXT4 장기 모델 연구를 수행한다.

프로젝트 위치:
C:\Users\minsu\Documents\Structural Break

기존 프로젝트 전체를 이어서 사용한다.
새 프로젝트를 만들지 마라.

이번 세션은 모델 점수를 실제로 높이기 위한 연구 세션이다.

하지만 단순 DEV score mining이 아니라:

- 서로 다른 구조의 모델/통계량 탐색
- strict causal streaming
- group-level validation
- cross-fitting
- pre-registration
- alternate split
- historical-regime stress
- whole-series bootstrap
- synthetic counterexample
- latency/reproducibility

를 동시에 사용해
일반화 가능성이 높은 개선만 남기는 것이 목표다.

============================================================
0. 최종 목표
============================================================

목표:

CrunchDAO Structural Break Real-Time competition의
hidden test에서 높은 TS-AUC를 얻을 모델을 만든다.

모델은 반드시:

- causal
- streaming
- 미래누수 없음
- online 현재 point까지만 사용
- dataset_id target encoding 없음
- tau feature 없음
- final online length 없음
- deterministic
- computationally feasible
- reproducible

이어야 한다.

Primary metric:

official TS-AUC

이다.

Synthetic metric,
first-alarm delay,
fixed threshold FPR

등은 secondary diagnostic이다.

============================================================
1. 현재 상태를 source/artifact에서 직접 복원
============================================================

먼저 최신 NEXT2/NEXT3 결과를 읽는다.

특히:

00_GPT_READ_FIRST_NEXT3_STOPPED.md
01_GPT_HANDOFF_NEXT3_STOPPED.md
02_PROCESS_AND_NEXT_STEPS.md

MODEL_RESULTS_NEXT3.csv
CV_RESULTS_NEXT3.csv
PAIRED_BOOTSTRAP_NEXT3.csv
ALT_SPLIT_RESULTS_NEXT3.csv
REGIME_RESULTS_NEXT3.csv
MECHANISM_RESULTS_NEXT3.csv

ROUTER_RESULTS.csv
SCALE_REPAIR_RESULTS.csv
C205_ABLATION_RESULTS.csv
ORTHOGONAL_COMPLEMENT_RESULTS.csv

COUNTEREXAMPLE_RESULTS_NEXT3.csv
LONG_NULL_RESULTS_NEXT3.csv

BOTTLENECK_BOARD_STOPPED_NEXT3.json
ENGINEERING_AND_VALIDATION_NEXT3.json
LIMITATIONS_NEXT3_STOPPED.md

그리고 이전 NEXT2:

DEV_CANDIDATES_NEXT2.json
CURRENT_GENERATION_COMPONENT_GRAPH.md

관련 source/config/test를 모두 확인한다.

프롬프트의 숫자를 authority로 삼지 마라.

현재 frozen scientific reference들을 실제 artifact hash로 복원한다.

최소:

FROZEN_J102
FROZEN_PERFORMANCE_BL80
FROZEN_DISTILLED

보존.

탐색상 최고였지만 승격되지 않은 I2 등은
EXPLORATORY_REFERENCE로만 둔다.

기존 canonical model을 덮어쓰지 않는다.

============================================================
2. 현재 알려진 출발점
============================================================

artifact 확인 전제에서 현재 대략:

FROZEN PERFORMANCE:

BL2_C205_DEPENDENCE_80
= 0.8 × J102
+ 0.2 × C205

DEV mean TS-AUC ≈ 0.6232684

이전 J102:
≈ 0.6216318

NEXT3 탐색 최고 I2:
≈ 0.6238383

하지만 고정 promotion gate 미달.

따라서 이번 NEXT4 기준 champion은
I2가 아니라 기존 FROZEN PERFORMANCE다.

============================================================
3. 절대 보호 자산
============================================================

LOCAL SEAL:

SEAL_A = 500
SEAL_B = 500
SEAL_C = 500
SEAL_D = 500

이번 NEXT4에서도:

A opened = false
B opened = false
C opened = false
D opened = false

유지.

절대 금지:

- seal x 값 분석
- seal y/tau 접근
- seal prediction
- seal TS-AUC
- seal mechanism diagnostic

Reduced:
새 사용 0.

Public LB:
새 submission 0.

Cloud submission:
0.

이번 세션에서는
DEV 8,000과 이미 존재하는 locked validation machinery만 사용한다.

============================================================
4. 시간 계약
============================================================

이번 세션은:

실질적인 모델 연구 최소 6시간

이다.

최종 audit/documentation은
그 이후 별도 45~60분.

따라서 목표 총 wall-clock:

약 6시간 45분 ~ 7시간+

최소 연구 6시간 이전에:

"실험 목록을 다 돌렸다"
"더 할 게 없다"

는 이유로 종료하지 마라.

단 sleep으로 시간 채우기 금지.

실험 queue가 끝나면
bottleneck-driven adaptive research loop로 이동한다.

============================================================
5. NEXT3 중단 상태 처리
============================================================

NEXT3는 사용자 요청으로 중단됐으며
마지막 fresh raw reproduction / final 30M API / final full audit 일부가
완료되지 않았다.

하지만 기존 NEXT2 frozen scientific models는
이미 독립 재현 evidence가 있으므로
NEXT4 scientific research의 reference로 사용 가능하다.

NEXT3 E003 runtime을
최종 canonical이라고 가정하지 마라.

NEXT4 시작 시 필요한 최소 preflight만 수행한다.

새 연구와 관계없는
NEXT3 전체 deployment finalization에
첫 몇 시간을 소비하지 않는다.

============================================================
6. Preflight
============================================================

연구 전에 빠르게 확인:

- raw SHA256
- split SHA
- seal partition SHA
- scorer parity
- FROZEN PERFORMANCE prediction fingerprint
- environment
- git/source status
- test smoke

문제 없으면 진행.

문제가 있으면 scientific experiment 중단 후 해결.

결과:

artifacts/next4/preflight/NEXT4_PREFLIGHT.json

============================================================
7. 이번 연구의 가장 중요한 원칙
============================================================

현재 DEV는 이미 여러 세션에서 많이 사용됐다.

따라서:

"새로운 모델이 DEV에서 +0.0003 올랐다"

는 것 자체는 거의 가치가 없다.

이번에는 새 candidate가 생기기 전에
무엇을 시험할지 제한한다.

각 hypothesis family마다:

1. finite candidate menu를 먼저 작성
2. menu hash freeze
3. stop rule freeze
4. promotion gate freeze
5. 그 뒤 score 계산

한다.

결과를 본 뒤:

threshold 추가
weight 미세조정
lag 하나 더 추가
seed 변경

하는 방식 금지.

============================================================
8. Candidate budget
============================================================

이번 NEXT4 전체에서
scientific candidate explosion을 막는다.

권장 최대:

major hypothesis families <= 8

family당 full CV candidate:
기본 1~4개

전체 새 FULL candidate:
최대 약 24개

특별한 이유 없이 30개를 넘지 마라.

screen candidate는 더 많을 수 있지만
동일 family의 hyperparameter micro-search는 금지.

목표:

후보 개수보다
아이디어 다양성.

============================================================
9. 실험 단계
============================================================

모든 새 아이디어는:

STAGE 0
사전등록

STAGE 1
unit/synthetic sanity

STAGE 2
cheap locked screen

STAGE 3
full frozen 5-fold

STAGE 4
paired whole-series bootstrap

STAGE 5
ALT_1 / ALT_2 / ALT_3

STAGE 6
H-regime holdout

STAGE 7
mechanism fracture

순서.

screen 탈락을 감정적으로 full CV에 올리지 마라.

============================================================
10. Pre-registration schema
============================================================

각 experiment마다 실행 전에 저장:

experiment_id
parent
family
hypothesis
causal_mechanism
exact_changes
parameters
candidate_menu_id
expected_gain
risk
screen_gate
full_gate
stop_rule
created_before_score = true

결과:

artifacts/next4/registrations/

에 저장.

============================================================
11. 모델 연구 방향을 다양화한다
============================================================

이번에는 variance branch만 계속 변형하지 않는다.

서로 다른 원리를 가진 다음 방향을
독립 hypothesis family로 다룬다.

FAMILY A
dependence / whiteness

FAMILY B
robust scale / direction asymmetry

FAMILY C
distribution change

FAMILY D
sequential likelihood / change-age mixture

FAMILY E
within-time ranking objective

FAMILY F
multi-expert ensemble

FAMILY G
historical regime adaptation

FAMILY H
compact temporal state model

단 neural network 대규모 연구는 금지.

============================================================
12. FAMILY A — dependence / whiteness V2
============================================================

현재 C205는 유효한 complement다.

하지만 단순 lag product만 사용한다.

새로운 dependence family를
C205의 미세변형이 아닌 다른 관점으로 연구.

후보 아이디어:

A1. whiteness residual score

Historical AR8 또는 existing whitening 이후
online innovation의 serial dependence가
H null 대비 얼마나 달라졌는지 측정.

lag 1/2/4/8을 개별 raw product로만 주지 말고:

- aggregate portmanteau-like statistic
- signed vs unsigned dependence
- short vs long dependence energy

를 streaming으로 계산.

예:

Qshort =
sum_lag w_l * standardized_corr_lag²

lag:
1,2,4,8

또는 작은 고정 set.

A2. partial-correlation style approximation

AR1/AR2 residual relationship을 이용한
partial dependence proxy.

full repeated matrix inversion 금지.

Historical에서 coefficients fit,
online update는 O(p).

A3. magnitude dependence

z_t²,
|z_t|

의 autocorrelation 변화.

volatility clustering structure 변화를
단순 variance level과 분리.

A4. sign dependence

sign(z_t) * sign(z_t-k)

로 amplitude에 둔감한 구조변화.

각 후보는 C205와
prediction correlation 및 incremental blend gain을 측정.

============================================================
13. Dependence candidate 성공 조건
============================================================

Standalone score보다 중요한 것:

FROZEN PERFORMANCE에 추가했을 때
incremental gain.

후보가:

standalone > 0.54

AND

J102/C205와 충분히 다른 prediction

AND

fixed blend에서 +0.001 이상

이면 full 연구 가치 높음.

하지만 weight micro-search 금지.

고정 blend 후보 예:

90/10
80/20

정도만.

============================================================
14. FAMILY B — robust scale asymmetry V2
============================================================

기존 S1/S2/S3를 그대로 반복하지 마라.

과거 실패를 먼저 읽는다.

새 아이디어는
"negative scale information이 H heavy-tail calibration에서 압축된다"
는 forensic을 활용해야 한다.

후보:

B1. asymmetric historical normalization

positive raw-energy와
negative raw-energy를
같은 SD로 정규화하지 않는다.

H에서:

positive deviation scale
negative deviation scale

을 별도로 계산.

Online:

energy > center
→ upper scale normalization

energy < center
→ lower scale normalization

B2. semi-variance reference

H residual의:

upper semivariance
lower semivariance

별도.

B3. quantile-calibrated direction evidence

현재 conditional variance ratio가
historical conditional variance distribution에서
몇 percentile인지 이용.

upper-tail percentile
lower-tail percentile

분리.

B4. signed robust log-scale

log conditional variance ratio의
positive/negative branch를
별도 robust H calibration.

중요:

기존 winsor/Huber/ECDF exact duplicate가 되지 않도록
ledger를 먼저 검색.

============================================================
15. Scale repair 성공 조건
============================================================

단순 synthetic variance-down improvement만으로 채택 금지.

최소:

overall DEV >= reference - 0.0005

AND

SCALE supported slice 개선

AND

variance-increase slice catastrophic loss 없음

AND

ARCH/tail synthetic decrease improvement

이면 ROBUSTNESS candidate.

성능 승격은 별도 gate.

============================================================
16. FAMILY C — distribution shift
============================================================

현재 tail/distribution family가
variance/dependence보다 약했지만
독립 complement 가능성은 있다.

새로운 low-cost streaming two-sample statistic 연구.

Historical reference는 immutable.

후보:

C1. histogram divergence

Historical normalized innovation을
고정 bins로 나눈 distribution.

Online EWMA bin mass와 비교.

Metrics finite menu:

- chi-square-like
- Hellinger-like
- Jensen-Shannon-like approximation

bins:
8 또는 16

정도.

C2. quantile imbalance

Historical quantiles:

5
10
25
50
75
90
95

고정.

Online exceedance EWMA deviation.

C3. skewness-like signed tail imbalance

left extreme rate
vs
right extreme rate.

C4. center-to-tail redistribution

variance level만 변한 것과
shape/tail이 변한 것을 구분하도록:

center mass
moderate tail
extreme tail

비율.

full KS / AD 매 point 금지.

============================================================
17. FAMILY D — sequential likelihood V2
============================================================

현재 Bayes-lite / age 256/512가 중요했다.

이번에는 age grid를 미세조정하지 말고
likelihood model 자체를 바꾼다.

D1. mixture mean-shift likelihood

whitened innovation z에서
post-change mean amplitude:

±small
±medium
±large

고정 finite grid.

각 candidate change age와 amplitude의
log-likelihood ratio를 logsumexp.

D2. scale-mixture likelihood

variance multiplier 후보를:

decrease
mild increase
strong increase

양 방향으로 사전 고정.

예:
0.5
0.7
1.5
2.0

실제 값은 등록 시 freeze.

D3. combined mean+scale mixture

너무 많은 조합 금지.

작은 grid만.

D4. hazard mixture

한 개 hazard를 가정하지 말고
short/medium/long hazard 3개 정도의 고정 mixture.

주의:

true tau 사용 금지.

future length 금지.

============================================================
18. Sequential likelihood 계산량
============================================================

O(T²) 금지.

age grid는 exponential/fixed sparse.

기존 prefix/ring buffer infrastructure 재사용.

O(num_ages × num_amplitudes)

가 bounded small constant가 되게.

10M latency 예상 같이 측정.

============================================================
19. FAMILY E — TS-AUC aligned ranking learner
============================================================

NEXT3 router에서
logistic/proper loss 개선이 TS-AUC 상승으로 연결되지 않았다.

따라서 모델 결합 단계에서
metric alignment를 직접 연구.

핵심:

TS-AUC는 같은 online time t에서
series 간 ranking이다.

후보:

E1. pairwise logistic ranking

train fold 내부에서
같은 t의 positive/negative pair를
deterministic subsample.

objective:

score(pos) > score(neg)

E2. LightGBM ranking objective

group = online t

형태가 실제 TS-AUC 구조와 맞는지 먼저 검토.

NDCG 자체를 목표로 하지 말고
pair ordering 관점.

E3. weighted binary approximation

각 t의 effective pair mass:

n_pos(t) × n_neg(t)

반영.

기존 W3와 정확히 무엇이 다른지 명시.

============================================================
20. Pairwise ranking 누수 방지
============================================================

validation t의 label distribution을
training weight에 사용 금지.

각 training fold 내부 통계만 사용.

outer validation은 완전 untouched.

pair sampling seed freeze.

pair count가 너무 크면
deterministic capped sampling.

same-series pair 금지 여부도 논리적으로 검토.

============================================================
21. FAMILY F — multi-expert model
============================================================

현재 J102와 C205는:

variance expert
dependence expert

라고 볼 수 있다.

다음 expert 후보가 실제 독립 신호를 가지면
3-expert 구조를 검토.

예:

Expert 1:
J102 variance/general

Expert 2:
C205 dependence

Expert 3:
distribution or sequential likelihood

단 expert가 standalone 약하고
correlation 높으면 추가하지 않는다.

============================================================
22. Expert 결합 방식
============================================================

무제한 router 금지.

우선:

fixed weights

예:

70/20/10
75/15/10
80/10/10

같은 아주 작은 사전 grid.

또는 rank-average.

Dynamic router는 NEXT3에서 이미 약했으므로
새 정보 없이 다시 복잡화하지 않는다.

============================================================
23. FAMILY G — historical regime adaptation
============================================================

Historical H는 clean이므로
합법적인 context다.

현재 model은 각 series H에서 normalization은 하지만
"H가 어떤 regime인지"를 learner가 충분히 활용하지 않을 수 있다.

Outcome-blind historical context:

- H volatility
- tail index proxy
- autocorrelation
- AR coefficients
- ARCH persistence
- spectral low/high ratio
- H length

를 아주 작은 context vector로 만든다.

중요:

dataset identity 학습 금지.

context는 H-only.

============================================================
24. Historical context 활용 방식
============================================================

후보:

G1.
context feature를 base learner에 직접 추가.

G2.
H regime별 normalization parameters 변경.

G3.
small mixture-of-experts with
H-only deterministic gate.

단 H cluster를 target/tau로 만들지 않는다.

existing H-regime stress와 contamination 없는지 확인.

============================================================
25. FAMILY H — compact temporal state model
============================================================

대형 LSTM/Transformer 금지.

하지만 순수 hand-engineered statistic 외의
작은 temporal model 하나 정도는 탐색 가능.

예:

H1.
작은 online state-space linear filter.

H2.
tiny GRU는 이번 세션 기본적으로 제외.

H3.
recursive logistic state:

s_t =
a*s_(t-1)
+ b*f_t

그 위에 fixed nonlinear transform.

parameters는 train fold에서 학습.

목표:

tree가 여러 evidence memory를
static하게 조합하는 것보다
더 나은 persistent change evidence가 가능한지 확인.

복잡한 deep sequence model 금지.

============================================================
26. 현재 FROZEN model을 계속 재학습하지 마라
============================================================

이미 frozen OOF prediction이 있으면
candidate blend 분석에서 재사용.

새 candidate 때문에 필요한 경우에만
base model refit.

동일 J102/C205 fold model을
매 실험마다 다시 fit하지 마라.

compute 낭비 방지.

============================================================
27. Nested structure
============================================================

모든 meta-model,
router,
stacking,
residual learner,
context gate

는 strict nested structure.

outer fold validation은
candidate creation에서 어떤 방식으로도 보지 않는다.

outer train 내부 OOF를 이용해
meta component를 학습.

============================================================
28. Cross-fitting hard gate
============================================================

다음 검사 자동화:

meta training row의 base prediction이
동일 row를 학습한 base model에서 나오지 않았는가?

TRUE이면:

LEAKAGE_FAIL

즉시 candidate 폐기.

============================================================
29. Prefix invariance
============================================================

새 streaming feature마다:

future suffix를 바꿔도
현재 prefix까지 feature/prediction 동일

테스트.

최소 random 100 cases.

finalists:
더 확대.

============================================================
30. No future-length hard guard
============================================================

Feature state / model API에는:

total online length

전달 자체를 하지 않는다.

dataset iterator를 list로 변환 금지.

next point read-ahead 금지.

============================================================
31. Negative controls
============================================================

새 feature/model infrastructure가
가짜 signal을 생성하지 않는지 제한적 검사.

예:

- label permutation
- invalid future-shift control
- constant score
- shuffled series ID

future-shift invalid control이
높은 점수를 내면
pipeline leakage 가능성 조사.

============================================================
32. Promotion gate — PERFORMANCE
============================================================

이번 세션 시작 전에 freeze.

현재 FROZEN PERFORMANCE 대비
새 candidate 승격:

mean Δ >= +0.0013

AND

median >= reference median - 0.0005

AND

worst fold >= reference worst - 0.001

AND

whole-series paired bootstrap:
P(Δ>0) >= 0.90

AND

95% CI lower > 0
가능하면 필수

AND

ALT 3개 중 최소 2개 positive

AND

H-regime catastrophic failure 없음

AND

supported mechanism 하나의 gain 때문에
다른 major supported mechanism이 크게 무너지지 않음

AND

leakage PASS

AND

runtime feasible.

결과 보고 완화 금지.

============================================================
33. NEW GENERATION gate
============================================================

진짜 다음 generation:

FROZEN PERFORMANCE 대비

mean Δ >= +0.0025

AND

bootstrap P > 0.95

AND

ALT 3/3 positive

AND

worst fold non-worse 또는 매우 작은 손실

AND

새 독립 signal 또는 명확한 robustness 개선

AND

nested/generalization diagnostic 양호.

+0.0035 이상이면 strong generation.

그래도 seal은 이번 세션에 열지 않는다.

============================================================
34. Candidate statistical comparison
============================================================

full candidate끼리 비교 시:

paired whole-series bootstrap

최소:
2,000 draws.

generation finalist:
5,000 draws 가능.

단 bootstrap은 fitted DEV conditional evidence임을 명시.

adaptive selection 보정이 아님.

============================================================
35. Family-wise multiplicity 관리
============================================================

각 family에서 후보 여러 개를 시험할 경우:

"best-of-N" 효과를 기록.

최종 report에는:

family candidate count

를 반드시 포함.

가능하면
family-level bootstrap max-T 또는
simple Holm-like correction을 exploratory diagnostic으로 수행.

단 공식 score를 수정하는 게 아니다.

목표는:

후보 20개 중 우연히 1개가 +0.0005 뜬 것을
과신하지 않는 것.

============================================================
36. Winner's curse audit
============================================================

새 최고 candidate가 생기면:

selection pool 내
1위와 2위 차이

family candidate count

bootstrap uncertainty

alternate split dispersion

을 이용해
winner's curse risk를 기록.

작은 gain이면 conservative하게 처리.

============================================================
37. Split stability
============================================================

기존:

PRIMARY locked 5-fold
ALT_1
ALT_2
ALT_3

를 그대로 사용.

새 seed 금지.

새 candidate마다 split 새로 만들기 금지.

============================================================
38. H-regime stress
============================================================

기존 outcome-blind H regime partition 유지.

새 candidate가:

primary CV +0.002
but
특정 H regime -0.01

이면 경고.

reason 분석.

============================================================
39. Mechanism diagnostics
============================================================

기존 post-hoc mechanism:

DEPENDENCE
SCALE
MIXED
WEAK/AMBIGUOUS
MEAN
TAIL

등을 evaluation-only로 사용.

support가 작은 slice는
과도하게 해석하지 않는다.

mechanism을 feature/router target으로 넣지 않는다.

============================================================
40. Time fracture
============================================================

모든 finalist:

online time:
0-9
10-24
25-49
50-99
100-199
200-399
400+

break age:
0-4
5-9
10-19
20-49
50+

기록.

전체 score 개선이
400+만 올려 만든 것인지 확인.

============================================================
41. False positive / outlier
============================================================

기존 약점:

single outlier persistence.

새 evidence model이
outlier를 더 악화시키는지 반드시 검사.

하지만 synthetic outlier 하나 때문에
real DEV +0.003 candidate를 자동 reject하지 않는다.

diagnostic로 사용.

============================================================
42. Long-null
============================================================

기존 long-null FPR growth가 있다.

새 candidate가
memory/sequential evidence를 추가하면:

1024
4096
16384

최소 검사.

FPR 폭증 시 기록.

그러나 competition TS-AUC가 primary.

============================================================
43. Model diversity 측정
============================================================

새 standalone model마다:

Pearson
Spearman
within-time Spearman
top-score overlap
disagreement

vs:

J102
C205
BL80

측정.

Standalone 0.58인데 correlation 0.99인 모델보다
Standalone 0.56인데 correlation 0.3이고 blend +0.002면
후자가 더 가치 있을 수 있다.

============================================================
44. Blend gain이 authority
============================================================

Complement 연구에서는:

standalone score

보다

FROZEN PERFORMANCE에 추가한
incremental TS-AUC

를 주요 판단으로 사용.

단 weight search는 finite.

============================================================
45. Learner 다양성
============================================================

동일 feature에서 learner가 병목일 가능성도 제한적으로 확인.

후보:

LightGBM 100-tree existing

small HistGradientBoosting

regularized logistic

small XGBoost / compact tree

pairwise rank learner

다만 learner hyperparameter sweep 금지.

각 learner는 사전 fixed config 1~2개만.

============================================================
46. Tree hyperparameter 미세튜닝 금지
============================================================

num_leaves 15/16/17...
depth 2/3/4...
learning rate 0.04/0.05...

식 미세 search 금지.

새 score는 구조적 정보에서 얻어야 한다.

============================================================
47. Feature importance 사용 규칙
============================================================

SHAP/importance는
candidate 생성 힌트일 뿐.

importance가 낮다고
feature 삭제 후 점수가 그대로일 것이라 가정 금지.

matched ablation이 authority.

============================================================
48. Bottleneck board
============================================================

NEXT4에서도 계속 유지.

초기:

B4 adaptive DEV uncertainty
B2 scale-down downstream
B6 outlier persistence
B3 complement redundancy
B5 long-null
B7 early evidence

에 더해:

B9 missing independent signal
B10 metric-objective mismatch
B11 H-regime adaptation

등 필요시 추가.

약 45~60분마다 갱신.

============================================================
49. Bottleneck priority
============================================================

우선순위는:

expected TS-AUC impact
× evidence strength
× solvability

대비

overfit risk
× compute
× complexity

로 판단.

한 번에 ACTIVE <= 2.

============================================================
50. Adaptive research loop
============================================================

기본 family 실험이 끝나도
6시간 미만이면 종료 금지.

반복:

1. 현재 best 확인
2. bottleneck board 확인
3. high-value bottleneck 선택
4. 기존 실패 검색
5. 새 가설 등록
6. finite menu freeze
7. implement
8. unit tests
9. screen
10. full CV
11. bootstrap
12. ALT/H/mechanism
13. latency
14. promote/reject
15. checkpoint
16. 다시 bottleneck 분석

============================================================
51. Adaptive rule
============================================================

CASE A

새 dependence family가
C205보다 더 좋은 complement:
→ C205 대체/3-expert 비교.


CASE B

scale asymmetry가 real DEV에서도 성공:
→ J102 branch와 blend 양쪽에서 검증.


CASE C

distribution model standalone는 약하지만
blend gain 큼:
→ complement로 유지.


CASE D

sequential likelihood +0.002 이상:
→ evidence family 중심으로 확장.


CASE E

ranking learner가
같은 feature로 기존 binary learner를 명확히 이김:
→ metric-aligned learner 연구 집중.


CASE F

H context가 ALT/H-regime robustness 개선:
→ context model 발전.


CASE G

10개 이상 새 구조적 가설에서
best gain < +0.0007:
→ score mining stop.
DEV saturation risk HIGH.


CASE H

+0.0025 generation 후보:
→ freeze.
새 micro-tuning 중단.
robustness/reproduction 집중.

============================================================
52. DEV saturation protection
============================================================

다음 조건 중 2개 이상이면:

DEV_SATURATION_RISK = HIGH

- 10+ independent hypotheses no > +0.0007
- top gains fluctuate across folds
- bootstrap CI repeatedly crosses zero
- ALT splits inconsistent
- best gains all same family microvariants

이후 후보 생성 속도를 크게 줄인다.

남은 시간을:

- generalization diagnostics
- independent validation planning
- simplification
- deployment
- audit

에 사용.

============================================================
53. Runtime constraints
============================================================

새 candidate는
10M points 기준 현실적인 inference인지 확인.

FROZEN PERFORMANCE 약 90초대 direct runtime을
reference로 사용.

새 model이:

+0.0015 gain
but
10× 느림

이면 별도 판단.

3× 이상 느려지면 strong justification 필요.

============================================================
54. Shared state
============================================================

새 experts가:

AR8
ARCH1
innovation
conditional variance

등을 중복 계산하지 않게
공유 가능성을 찾는다.

단 scientific prediction을 바꾸지 않는
runtime optimization은
별도 engineering stage에서만.

============================================================
55. Research scheduler
============================================================

최소 6시간 실연구를 보장하는
NEXT4 orchestrator를 사용.

예:

scripts/run_next4_autonomous.py

기능:

- wall clock
- productive clock
- task queue
- checkpoint
- crash retry
- resource monitor
- candidate budget
- final audit transition

============================================================
56. Hang recovery
============================================================

child experiment가:

30분 이상 progress 없음
AND
CPU/activity/output 없음

이면:

HANG_SUSPECTED

1회 재시도.

다시 hang:
QUARANTINE.

전체 run 계속.

============================================================
57. Checkpoints
============================================================

최소 30~45분마다:

NEXT4_CHECKPOINT_xxx.json

내용:

wall_seconds
productive_seconds
current_family
current_hypothesis
best_score
best_delta
full_candidate_count
screen_count
failure_count
current_bottlenecks
seal_status
next_task

============================================================
58. Resource monitor
============================================================

RAM 32GB 환경 고려.

RAM > 85%:

concurrency 감소
large dataframe 해제
memmap/chunk 사용

global environment 변경 금지.

GPU 사용은
실제 이득 있을 때만.

============================================================
59. Cache policy
============================================================

cache key에:

raw hash
feature config hash
source hash
split hash
candidate id

포함.

stale cache 재사용 금지.

============================================================
60. Existing failure deduplication
============================================================

NEXT2/NEXT3 ledger에서
동일한 아이디어가 이미 실패했는지 확인.

다음과 같은 이름만 바꾼 반복 금지:

- same router
- same lower semiscale
- same winsorization
- same C205 lag subset
- same residual learner

새 가설은 기존 실패와 차이를 명시.

============================================================
61. 최소 시간 계획
============================================================

0:00–0:30
preflight
reference reconstruction
candidate family menu freeze

0:30–1:30
FAMILY A dependence/whiteness

1:30–2:30
FAMILY B robust asymmetric scale

2:30–3:20
FAMILY C distribution shift

3:20–4:10
FAMILY D sequential likelihood

4:10–5:00
FAMILY E ranking objective

5:00–5:30
FAMILY F/G complement + H context

5:30–6:00+
adaptive bottleneck loop
best finalists robustness

6시간 productive research 완료 후:

+45~60분
final validation/audit/handoff.

실험 시간에 따라 재배치 가능.

============================================================
62. Finalist requirements
============================================================

새 finalist는 최소:

- full 5-fold
- exact TS-AUC
- paired bootstrap
- ALT
- H-regime
- mechanism
- prefix invariance
- deterministic
- latency

완료.

미완료 finalist를 champion이라고 부르지 않는다.

============================================================
63. 최종 후보 역할
============================================================

최대:

FROZEN_PERFORMANCE
NEXT4_PERFORMANCE
NEXT4_ROBUSTNESS
NEXT4_EXPERIMENTAL_GENERATION

4개.

후보 20개 남기지 않는다.

============================================================
64. Seal policy
============================================================

이번에도 실제 seal은 열지 않는다.

종료 시:

SEAL_OPEN_RECOMMENDATION_NEXT4.md

에만 기록.

판정:

NO
MAYBE
YES

generation candidate가 있어도
실제 개봉하지 마라.

============================================================
65. Public submission
============================================================

실제 submission 금지.

PUBLIC_SUBMISSION_RECOMMENDATION_NEXT4.md

만 작성.

============================================================
66. 최종 audit
============================================================

6시간 실질 연구가 끝난 후
45~60분.

다음 수행:

- raw hash
- seal guard
- scorer parity
- selected candidate tests
- prefix invariance
- offline/stream parity
- determinism
- save/load parity
- full suite 가능하면
- model source hash
- split hash
- reduced usage 0
- public usage 0
- seal usage 0

============================================================
67. Fresh reproduction
============================================================

새 NEXT4 candidate가 실제 promotion되었으면
fresh output directory에서:

raw DEV
→ feature
→ training
→ inference

최소 selected model 1개 재현.

기존 cached prediction과 exact 또는 strict tolerance 비교.

새 candidate가 없으면
FROZEN model을 또 의미 없이 재학습할 필요 없다.

============================================================
68. 최종 산출물
============================================================

00_GPT_READ_FIRST_NEXT4.md
01_GPT_HANDOFF_NEXT4.md

NEXT4_PREFLIGHT.json

HYPOTHESIS_FAMILY_LOCK.json
CANDIDATE_BUDGET.json

DEPENDENCE_V2_RESULTS.csv
SCALE_ASYMMETRY_RESULTS.csv
DISTRIBUTION_SHIFT_RESULTS.csv
SEQUENTIAL_LIKELIHOOD_RESULTS.csv
RANKING_OBJECTIVE_NEXT4.csv
H_CONTEXT_RESULTS.csv

COMPLEMENT_RESULTS_NEXT4.csv
MODEL_RESULTS_NEXT4.csv
CV_RESULTS_NEXT4.csv
PAIRED_BOOTSTRAP_NEXT4.csv
ALT_RESULTS_NEXT4.csv
REGIME_RESULTS_NEXT4.csv
MECHANISM_RESULTS_NEXT4.csv
TIME_FRACTURE_NEXT4.csv

COUNTEREXAMPLE_RESULTS_NEXT4.csv
LONG_NULL_RESULTS_NEXT4.csv

BOTTLENECK_BOARD_INITIAL_NEXT4.json
BOTTLENECK_BOARD_FINAL_NEXT4.json
BOTTLENECK_HISTORY_NEXT4.csv

LEAKAGE_AUDIT_NEXT4.json
PREFIX_INVARIANCE_NEXT4.json
DETERMINISM_NEXT4.json
LATENCY_NEXT4.json

DEV_CANDIDATES_NEXT4.json

SEAL_GUARD_NEXT4.json
SEAL_OPEN_RECOMMENDATION_NEXT4.md
PUBLIC_SUBMISSION_RECOMMENDATION_NEXT4.md

FINAL_AUDIT_NEXT4.json
REPRODUCTION_NEXT4.json

LIMITATIONS_NEXT4.md
NEXT_SESSION_PLAN_NEXT4.md

EXPERIMENT_LEDGER.csv

SOURCE_MANIFEST_NEXT4.json

그리고:

STRUCTURAL_BREAK_GPT_HANDOFF_NEXT4_<timestamp>.zip

생성.

raw competition data
credentials
seal labels
real OOF full matrix

ZIP 제외.

============================================================
69. 00_GPT_READ_FIRST_NEXT4.md
============================================================

반드시 한국어로 아래를 명확히 답한다.

1. 실제 productive research 시간
2. wall-clock
3. 새 hypothesis family 수
4. screen candidate 수
5. full candidate 수
6. learner fit 수
7. 시작 champion
8. 시작 exact score
9. 탐색 최고 score
10. 최종 선택 score
11. 최종 delta
12. 새 generation 여부
13. strongest new idea
14. dependence V2 결과
15. scale asymmetry 결과
16. distribution result
17. sequential likelihood result
18. ranking objective result
19. H-context result
20. 새 complement result
21. 5-fold scores
22. median
23. worst fold
24. bootstrap CI
25. P(delta>0)
26. ALT_1/2/3
27. H-regime worst
28. mechanism trade-offs
29. early/late fracture
30. long-null
31. outlier
32. latency
33. state size
34. leakage tests
35. determinism
36. fresh reproduction
37. DEV saturation risk
38. SEAL_A/B/C/D opened?
39. reduced usage?
40. public submission?
41. seal open recommendation
42. 다음 연구 병목 3개

============================================================
70. 성공 정의
============================================================

이번 연구 성공은 다음 중 하나.

A.
FROZEN PERFORMANCE보다
+0.0025 이상 새로운 generation.

B.
+0.0013 이상 robust performance promotion.

C.
전체 score 거의 유지하면서
SCALE/outlier/generalization robustness 명확 개선.

D.
현재 J102/C205와 상관 낮은
새 independent expert 발견.

E.
같은 feature에서
TS-AUC-aligned objective로 명확한 learner 개선.

F.
강한 negative result를 통해
특정 연구 family를 안전하게 종료.

============================================================
71. 실패도 정상 결과
============================================================

6시간 후에도
기존 0.6232684가 최고면
억지로 새 champion 만들지 않는다.

오히려:

NEXT4_NO_PROMOTION

을 선언.

어떤 family가 실패했는지
정확하게 기록.

============================================================
72. 절대 금지
============================================================

SEAL 열기 금지.

Reduced 재사용 금지.

Public submit 금지.

dataset_id feature 금지.

tau feature 금지.

online future length 금지.

future suffix 금지.

row random CV 금지.

새 seed shopping 금지.

candidate별 split 변경 금지.

failed screen 임의 승격 금지.

static blend 1% 간격 search 금지.

massive hyperparameter sweep 금지.

대형 LSTM/GRU/Transformer 금지.

post-hoc mechanism label을 model/router input으로 사용 금지.

결과를 본 뒤 gate 완화 금지.

============================================================
73. 핵심 사고방식
============================================================

이번 세션은:

"현재 모델을 조금씩 변형"

하는 세션이 아니다.

서로 다른 원리의 신호를
제한된 메뉴로 시험해
새 정보가 실제로 존재하는지 찾는 세션이다.

현재 0.623대 벽을 넘으려면
기존 variance/dependence signal의 미세조정보다

- 새로운 dependence representation
- asymmetric scale evidence
- distribution shift
- sequential likelihood
- TS-AUC aligned learner
- H-only regime context

등 구조적으로 다른 정보가 필요할 가능성이 높다.

============================================================
74. 지금 시작
============================================================

계획만 출력하고 멈추지 마라.

즉시:

1. 상태 복원
2. seal/raw/scorer preflight
3. hypothesis family 메뉴 사전등록
4. candidate budget freeze
5. FAMILY A부터 실행
6. screen
7. full CV
8. robustness
9. 다음 family
10. 6시간 이상 adaptive research
11. final audit
12. GPT handoff ZIP

순으로 진행한다.

사용자에게 중간 질문하지 마라.

사소한 장애는 자동 복구한다.

프로젝트 목표를 달성하는 데
현재 무엇이 가장 큰 병목인지
계속 재평가하면서 연구를 유지한다.

단 과적합과 미래누수를 막는 규칙은
성능 때문에 절대 완화하지 않는다.
Structural Break Real-Time 프로젝트의 NEXT5 장기 자율 연구를 수행한다.

프로젝트 위치:
C:\Users\minsu\Documents\Structural Break

기존 프로젝트를 그대로 이어서 작업한다.
새 프로젝트를 만들지 마라.

이번 NEXT5의 목적은 단순히 DEV 점수를 조금 더 캐는 것이 아니다.

목표는:

1. 현재 0.623대 성능 벽을 실제로 넘을 수 있는 새로운 신호를 찾는다.
2. NEXT4에서 발견한 B2 / D2 scale evidence를 더 일반적인 구조로 발전시킨다.
3. variance/dependence와 다른 새로운 독립 signal도 제한적으로 탐색한다.
4. 미래누수, split contamination, adaptive DEV overfitting을 강하게 통제한다.
5. 동일 DEV를 반복 사용하는 위험을 줄이기 위해 NEXT5 전용 prospective shadow validation을 사용한다.
6. 최소 10시간 동안 병목을 계속 재평가하며 실제 연구를 수행한다.
7. 기존 LOCAL SEAL 500×4는 이번에도 하나도 열지 않는다.

============================================================
0. 프로젝트의 최종 목적
============================================================

CrunchDAO Structural Break Real-Time competition에서
hidden test TS-AUC를 최대화할 수 있는 모델을 만든다.

모델은 반드시:

- causal
- streaming
- current observation까지만 사용
- future suffix 접근 불가
- final online length 접근 불가
- tau를 inference feature로 사용하지 않음
- dataset_id target encoding 없음
- deterministic
- reproducible
- 현실적인 latency
- hidden distribution에서 일반화 가능성 높음

이어야 한다.

PRIMARY METRIC:

official TS-AUC

이다.

다음은 secondary diagnostic:

- synthetic DGP score
- fixed-threshold alarm FPR
- detection delay
- outlier persistence
- mechanism slice

이다.

Secondary metric 때문에 primary TS-AUC를 희생하지 마라.

============================================================
1. 가장 먼저 최신 NEXT4 상태를 복원
============================================================

다음 파일과 관련 source를 모두 읽는다.

00_GPT_READ_FIRST_NEXT4.md
01_GPT_HANDOFF_NEXT4.md

MODEL_RESULTS_NEXT4.csv
CV_RESULTS_NEXT4.csv
PAIRED_BOOTSTRAP_NEXT4.csv
ALT_RESULTS_NEXT4.csv
REGIME_RESULTS_NEXT4.csv
MECHANISM_RESULTS_NEXT4.csv
TIME_FRACTURE_NEXT4.csv

COUNTEREXAMPLE_RESULTS_NEXT4.csv
LONG_NULL_RESULTS_NEXT4.csv

DEV_CANDIDATES_NEXT4.json
FINAL_AUDIT_NEXT4.json
LEAKAGE_AUDIT_NEXT4.json
PREFIX_INVARIANCE_NEXT4.json
DETERMINISM_NEXT4.json
REPRODUCTION_NEXT4.json
LATENCY_NEXT4.json

LIMITATIONS_NEXT4.md
NEXT_SESSION_PLAN_NEXT4.md

그리고:

NEXT3 관련 결과
NEXT2 frozen reference
EXPERIMENT_LEDGER.csv
src/
scripts/
configs/
tests/
artifacts/

를 확인한다.

프롬프트에 적힌 숫자를 authority로 사용하지 마라.
artifact/source가 authority다.

============================================================
2. FROZEN reference를 먼저 선언
============================================================

최소 다음 모델을 immutable reference로 고정.

FROZEN_PRIMARY
= BL2_C205_DEPENDENCE_80

NEXT4_ROBUSTNESS
= D2_DISCRETE_SCALE_MIXTURE

EXPLORATORY_NEAR_MISS
= B2_LOG_ENERGY_SEMITAIL

FROZEN_DISTILLED
= 기존 J102 tree-used fallback

정확한 모델 SHA / score / entry를 artifact에서 읽는다.

기존 model artifact를 덮어쓰지 마라.

NEXT5는:

src/next5/
artifacts/next5/

namespace에서 작업.

============================================================
3. 현재 과학적 출발점
============================================================

source 확인 전제에서 현재 대략:

FROZEN PRIMARY:
BL80
≈ 0.6232684

NEXT4 near-miss:
B2_LOG_ENERGY_SEMITAIL
≈ 0.6245721
Δ ≈ +0.001304

B2는:
- original 5-fold 5/5 개선
- ALT 3/3 개선
- P(delta>0) 높음
- 그러나 primary bootstrap CI lower < 0
- late time에서 대부분 gain
- heavy-tail decrease / repeated outlier에서 약점

NEXT4 robustness:
D2_DISCRETE_SCALE_MIXTURE
≈ 0.6235962
전체 gain은 작음

그러나 supported SCALE:
Δ ≈ +0.00894
CI 양수

따라서 NEXT5의 가장 중요한 질문은:

"B2의 전체 score 상승 신호와
D2의 robust scale information을
과적합 없이 하나의 더 일반적인 evidence representation으로
만들 수 있는가?"

이다.

============================================================
4. 절대 금지 — SEAL
============================================================

LOCAL SEAL 2000:

A = 500
B = 500
C = 500
D = 500

이번 NEXT5에서도:

SEAL_A_OPENED = false
SEAL_B_OPENED = false
SEAL_C_OPENED = false
SEAL_D_OPENED = false

절대:

seal label
seal tau
seal prediction
seal score
seal mechanism
seal feature analysis

하지 않는다.

raw file SHA 계산은 가능하지만
seal observation decoding/analysis와 구분한다.

============================================================
5. Reduced / public / cloud
============================================================

이번 세션:

REDUCED_NEW_USAGE = 0
PUBLIC_SUBMISSION_NEW = 0
CLOUD_SUBMISSION_NEW = 0

유지.

기존 reduced quota 2/2 상태를 변경하지 않는다.

실제 leaderboard를 model selection에 사용하지 않는다.

============================================================
6. 최소 연구 시간
============================================================

ACTIVE PRODUCTIVE RESEARCH:

최소 10시간
= 36,000 seconds

이다.

그 이후:

FINAL AUDIT:
약 50~60분

따라서 정상 총 wall-clock 목표:

약 10시간 50분 ~ 11시간+

10시간 productive research가 끝나기 전에
최종 handoff를 작성하고 종료하지 마라.

sleep으로 시간 채우기 금지.

실험 queue가 소진되면
adaptive bottleneck loop를 계속 실행한다.

============================================================
7. NEXT5 전용 prospective shadow validation
============================================================

매우 중요.

DEV 8,000 전체가 이미 과거 세션에서 반복 사용됐다.

따라서 완전히 fresh한 데이터는 아니지만,
NEXT5 내부 selection bias를 줄이기 위해
"prospective shadow"를 만든다.

기존 DEV 8,000 IDs만 대상으로:

SHA256(frozen_next5_shadow_salt + dataset_id)

을 사용하여 deterministic하게:

NEXT5_RESEARCH_POOL ≈ 80%
NEXT5_SHADOW ≈ 20%

로 나눈다.

예:
6,400 / 1,600

정확한 개수는 group constraints에 맞게 결정.

중요:

- SEAL 2,000과 완전히 별도
- NEXT5_SHADOW는 true seal이라고 부르지 않는다.
- 과거 training/DEV history 때문에 완전히 independent data도 아니다.
- 오직 NEXT5 내부 adaptive selection을 줄이는 prospective guard다.

============================================================
8. Shadow 사용 계약
============================================================

NEXT5_SHADOW labels는
candidate family 설계 중 열지 않는다.

Research phase 동안:

NEXT5_RESEARCH_POOL만 이용해
candidate generation / screen / model selection.

SHADOW는:

후보 generation과 major family가 거의 완료된 뒤
사전등록 finalist 최대 3개에 대해
단 한 번 평가.

중요:

SHADOW score를 본 뒤:

- feature 수정 금지
- weight 수정 금지
- hyperparameter 수정 금지
- 새로운 candidate 생성 금지

Shadow 결과는:

confirmation / rejection

용도.

그 뒤 다시 research를 해야 한다면
새 NEXT session에서 별도 계획을 세운다.

============================================================
9. Shadow contamination guard
============================================================

Research phase loader는 기본:

NEXT5_RESEARCH_POOL ONLY

로 동작.

SHADOW ID가:

training
validation
feature cache
OOF
candidate selection
bootstrap selection

에 들어가면 hard fail.

NEXT5_SHADOW_GUARD.json 생성.

============================================================
10. 기존 PRIMARY 평가도 ResearchPool 기준으로 재구성
============================================================

새 candidate와 공정 비교를 위해
ResearchPool 내에서:

기존 BL80
B2
D2

reference prediction / retraining protocol을
미리 고정.

기존 full-DEV 점수는 그대로 보존하되
NEXT5 selection authority는
NEXT5_RESEARCH_POOL CV가 된다.

단 기존 모델을 의미 없이 수백 번 refit하지 마라.

필요한 frozen prediction/cache 재사용.

============================================================
11. ResearchPool split
============================================================

NEXT5_RESEARCH_POOL에서:

group-aware 5-fold

freeze.

기존 duplicate/family group integrity 유지.

그리고:

ALT-A
ALT-B

정도만 ResearchPool 내부 deterministic validation으로 추가 가능.

새 seed shopping 금지.

split hash 저장.

NEXT5_SPLIT_LOCK.json

============================================================
12. 후보 개수 제한
============================================================

과적합 방지를 위해:

major hypothesis family:
최대 7개

screen candidate:
최대 약 35개

full candidate:
최대 20개

SHADOW finalist:
최대 3개

를 기본 hard budget으로 한다.

예외는 severe bug rerun 정도만.

random search 금지.

============================================================
13. 후보 등록
============================================================

모든 hypothesis는 DEV score 전에:

experiment_id
family
parent
scientific_question
mechanism
exact_feature_change
exact_model_change
parameter_menu
screen_gate
full_gate
promotion_gate
stop_rule
expected_risk
source_hash

기록.

등록 후 candidate config hash freeze.

============================================================
14. 이번 세션의 핵심 FAMILY S — Scale Evidence Unification
============================================================

가장 높은 우선순위.

B2와 D2를 단순 weight blend하지 않는다.

목표는:

B2의 log-energy semitail information

+

D2의 explicit discrete scale-likelihood information

를 하나의 일관된 scale evidence representation으로 만드는 것.

============================================================
15. S1 — directional scale likelihood
============================================================

Historical innovation 기준으로
variance ratio candidate를:

decrease
null
increase

방향별 likelihood로 만든다.

예:

scale multipliers:
0.5
0.7
1.0
1.5
2.0

정확한 grid는 score 보기 전에 freeze.

online residual z에서:

log p(z | scale=s)
-
log p(z | null)

계산.

그리고:

decrease evidence
increase evidence
absolute change evidence
signed scale evidence

를 별도 유지.

============================================================
16. S2 — robust scale likelihood
============================================================

Gaussian likelihood만 신뢰하지 않는다.

Historical residual tail behavior를
H-only로 추정해:

Gaussian-like
heavy-tail-like

null calibration을 다르게 할 수 있는
작은 robust likelihood family를 연구.

후보:

Student-t likelihood
또는
Huberized log-likelihood

중 1~2개.

df/threshold micro search 금지.

H-only estimate 또는 고정 사전값.

============================================================
17. S3 — B2/D2 information decomposition
============================================================

B2와 D2 각 feature를 분석.

다음 correlation/conditional information 측정:

B2 feature ↔ D2 feature
B2 score ↔ D2 score
B2 residual gain conditional on D2
D2 residual gain conditional on B2

within-time ranking 기준.

목표:

같은 scale information을 중복 표현하는지,
서로 다른 부분이 있는지 확인.

============================================================
18. S4 — scale evidence fusion model
============================================================

raw BL80 + B2 + D2 prediction을
그냥 0.01 단위 weight sweep하는 건 금지.

대신 unified scale features를 만들고
기존 learner에 직접 추가하거나
작은 scale specialist learner를 학습.

후보 최대:

S4a:
BL80 feature + unified scale

S4b:
J102/C205 base + unified scale specialist fixed blend

S4c:
scale specialist만 교체

정도.

============================================================
19. FAMILY P — Persistent vs Transient Shock
============================================================

현재 반복 outlier와 long memory가 약점.

목표:

"persistent structural break"

와

"temporary shock/outlier"

를 구별하는 causal evidence.

============================================================
20. P1 — two-timescale confirmation
============================================================

fast evidence:
짧은 EWMA

slow evidence:
긴 EWMA

에서:

fast large
but slow not confirming

이면 transient.

fast + slow both large

이면 persistent.

feature 예:

fast
slow
fast-slow
min(fast, slow)
confirmation_ratio

finite menu만.

============================================================
21. P2 — confirmation counter
============================================================

단일 extreme point로 evidence가 오래 남지 않도록:

recent K observations 중
same-direction evidence count

또는
soft persistence mass

를 유지.

예:

evidence > threshold

같은 hard threshold는 H-only calibration에서 freeze.

candidate score 보고 변경 금지.

============================================================
22. P3 — shock-and-recovery state
============================================================

reset heuristic을 직접 쓰기 전에
shock state와 persistent state를 분리.

예:

shock_t =
large instantaneous residual evidence

persistent_t =
decay * persistent_(t-1)
+
confirmed_evidence

shock는 빠르게 감소.
persistent는 confirmation 때만 증가.

small fixed decay menu.

============================================================
23. Outlier test
============================================================

P family는:

O1 single outlier
OB16 repeated outlier
long null
late true break

를 동시에 본다.

single outlier만 좋아지고
late true break를 죽이면 실패.

============================================================
24. FAMILY W — Windowed Change Evidence
============================================================

기존 evidence는 누적 memory가 길다.

새 방향:

historical null vs
최근 causal window

를 직접 비교.

window lengths:
예:
16
32
64
128

정도 작은 grid.

미래 길이 필요 없음.

============================================================
25. W1 — recent scale ratio
============================================================

recent window innovation energy
vs
historical/reference scale.

log variance ratio:

log(V_recent / V_ref)

signed.

============================================================
26. W2 — robust recent scale
============================================================

variance 대신:

MAD-like
median absolute residual
trimmed energy

등 1개 정도.

매 point full sort 금지.

streaming approximation / small ring buffer 허용.

============================================================
27. W3 — multiwindow disagreement
============================================================

short vs long window:

scale_16 - scale_128

등으로:

change onset
vs
persistent regime

구별.

pair explosion 금지.

2~3개 사전 조합만.

============================================================
28. FAMILY E — E-process / sequential evidence
============================================================

long-null accumulation과 scale change를
동시에 더 principled하게 처리하기 위한 방향.

완전한 이론적 anytime-valid 구현을 억지로 주장하지 않는다.

그러나 likelihood-ratio based
nonnegative evidence process를 탐색.

============================================================
29. E1 — scale mixture e-value
============================================================

Under null:

historical calibrated innovation distribution.

Alternatives:

scale down
scale up

mixture likelihood ratio.

e_t =
p_alt(z_t) / p_null(z_t)

sequential log evidence:

L_t = L_(t-1) + log(e_t)

단 무한 누적 문제를 막기 위해
candidate-age mixture 또는 windowed evidence와 결합.

============================================================
30. E2 — resettable mixture without future
============================================================

change age 후보:

1
2
4
8
16
32
64
128
256

등 sparse grid.

각 hypothetical recent start에 대해
log likelihood ratio 유지.

max보다는
fixed prior logsumexp 우선.

O(T²) 금지.

============================================================
31. FAMILY R — Robust Null Modeling
============================================================

B2가 heavy-tail 환경에서 약한 점을 직접 겨냥.

Historical H에서:

tail heaviness
kurtosis proxy
ARCH persistence

를 계산.

하지만 이것을 target/model selection shortcut으로 사용하지 않는다.

Null distribution parameterization에만 사용.

============================================================
32. R1 — H-adaptive tail model
============================================================

H innovation을:

Gaussian
Student-t-like

중 어떤 null calibration으로 처리할지
deterministic하게 정하거나
continuous parameter를 계산.

Outcome label 사용 금지.

============================================================
33. R2 — conditional scale PIT
============================================================

ARCH forecast를 거친 standardized innovation energy가
H distribution에서 어느 quantile인지 계산.

현재 squared residual value 자체보다
H-relative surprise를 사용.

upper surprise
lower surprise

분리.

============================================================
34. R3 — self-normalized score
============================================================

H extreme tail 때문에 denominator가 폭발하지 않도록:

median
IQR
MAD
quantile spread

등 robust scale을 사용하되,
NEXT2/NEXT3/NEXT4에서 이미 실패한 exact 방법과 중복하지 않는다.

새 방식이 기존 실패와 무엇이 다른지 명시.

============================================================
35. FAMILY M — Mean / Location Specialist V2
============================================================

mean 계열은 과거 약했으므로
낮은 budget만 배정.

하지만 hidden DGP diversity를 위해
완전히 포기하지는 않는다.

최대 2개 candidate.

============================================================
36. M1 — robust signed location likelihood
============================================================

whitened innovation에서:

positive mean shift
negative mean shift

양쪽 fixed likelihood mixture.

기존 D1과 다른 점을 명확히 한다.

단순 D1 amplitude grid 변경 금지.

예:
rank/sign 기반 likelihood
또는
Student-t location likelihood.

============================================================
37. M2 — sign-run evidence
============================================================

magnitude에 덜 민감한:

sign imbalance
run persistence

를 streaming state로 사용.

outlier robustness 기대.

standalone가 약하면 바로 family 종료.

============================================================
38. FAMILY X — Cross-Family Interaction
============================================================

마지막 단계에서만 사용.

S / P / W / E / R family 중
실제로 standalone 또는 base-increment gain이 확인된 신호만 결합.

처음부터 interaction 100개 생성 금지.

============================================================
39. X1 — scale × persistence
============================================================

scale evidence가 강한데
persistent confirmation도 강한 경우에만
high confidence.

예:

scaleScore * persistenceScore

또는 shallow tree interaction.

============================================================
40. X2 — scale × H-tail context
============================================================

H가 heavy-tail regime인지에 따라
같은 scale score를 다르게 해석.

H-only information이므로 causal.

그러나 NEXT4 G1 같은
context-only domination이 재발하는지 검사.

============================================================
41. X3 — dependence × scale
============================================================

C205 dependence signal과
new scale signal이 동시에 강할 때
mixed break 가능성을 잡는 interaction.

소수만.

============================================================
42. 기존 closed families 재탐색 금지
============================================================

NEXT4에서 사실상 닫힌 다음 메뉴를
이름만 바꿔 다시 실행하지 마라.

- A1/A2/A3 portmanteau exact variants
- C1/C2/C3 histogram distribution exact variants
- E1 pairwise linear exact recipe
- E2 pairwise interactions exact recipe
- G1/G2 H-context exact recipe
- H1 fixed linear state exact recipe
- NEXT3 router R1/R2/R3 exact variants
- NEXT3 S1/S2/S3 exact variants
- C205 lag subset exact variants

정말 다른 가설일 때만 새 family로 허용.

============================================================
43. Learner policy
============================================================

feature engineering이 primary 연구 대상.

learner는:

기존 LightGBM reference

+

선택적:
small HistGB
regularized linear
compact XGB

정도.

한 feature candidate마다
learner 5개 돌리지 마라.

초기에는 기존 learner 1개.

feature가 유망할 때
learner sensitivity 1개 정도.

============================================================
44. Hyperparameter search 금지
============================================================

다음 금지:

learning_rate grid
num_leaves brute force
depth brute force
min_child grid
subsample grid
random seed sweep

새 signal discovery가 목표.

============================================================
45. Nested meta model
============================================================

meta model / ensemble learner를 사용하면
strict nested cross-fitting 필수.

outer validation prediction은
meta training에서 절대 사용 금지.

inner OOF only.

automated leakage assertion 작성.

============================================================
46. NEXT5 prospective flow
============================================================

추천 구조:

RESEARCH_POOL
    ↓
5-fold research CV
    ↓
screen/full
    ↓
candidate shortlist

최대 3개 finalist freeze
    ↓
FINALIST_MANIFEST hash lock

그 뒤 처음으로:
NEXT5_SHADOW 평가

SHADOW 이후
candidate 변경 금지.

============================================================
47. Finalist 선택 규칙
============================================================

SHADOW 열기 전 최대 3개:

1. PERFORMANCE candidate
2. ROBUSTNESS candidate
3. ORTHOGONAL / NEW-SIGNAL candidate

선정.

같은 family의 거의 동일한 3개 모델을
shadow에 올리지 마라.

============================================================
48. ResearchPool promotion gate
============================================================

FROZEN PRIMARY 대비:

mean Δ >= +0.0012

AND

median non-worse

AND

worst fold >= -0.001 degradation

AND

whole-series bootstrap
P(delta>0) >= 0.90

AND

CI lower > -0.0003

AND

ALT research splits 일관성

AND

leakage PASS

면 SHADOW finalist 자격.

주의:

ResearchPool 표본이 작아졌으므로
기존 full-DEV threshold를 그대로 기계 적용하지 말고
이 gate를 score 보기 전에 lock.

============================================================
49. Strong finalist
============================================================

ResearchPool에서:

Δ >= +0.0025
AND
CI lower > 0
AND
fold 4/5 이상 positive
AND
major robustness collapse 없음

이면 STRONG.

============================================================
50. Shadow confirmation rule
============================================================

SHADOW 평가 전에 rule freeze.

예:

SHADOW delta > 0

AND

no catastrophic mechanism/regime collapse

이면:
CONFIRMED_DIRECTIONAL

SHADOW delta >= +0.001
이면:
CONFIRMED_STRONG

SHADOW delta <= -0.001
이면:
REJECT_GENERALIZATION

사후 변경 금지.

============================================================
51. Shadow는 model selection에 반복 사용하지 않는다
============================================================

SHADOW 결과를 본 뒤:

"weight 조금 바꾸자"
"feature 하나 제거하자"
"threshold 조금 조정하자"

금지.

그 순간 shadow도 adaptive DEV가 된다.

NEXT5 종료.

후속 변화는 NEXT6에서
새 prospective protocol 필요.

============================================================
52. 기존 8k full DEV score도 마지막에 계산 가능
============================================================

SHADOW 결과를 본 뒤가 아니라,
final role 결정 후
최종 reporting 목적으로:

ResearchPool + Shadow = 기존 DEV 8,000

전체 OOF/reporting score를 재구성 가능.

하지만 full DEV 점수 보고
NEXT5 role을 다시 변경하지 않는다.

============================================================
53. Multiplicity 관리
============================================================

family별 후보 수 기록.

각 family에서:
best-of-N selection risk 계산.

가능하면:

bootstrap max statistic
또는
family-wise adjusted diagnostic

제공.

공식 score를 수정하지 않는다.

============================================================
54. Winner-gap audit
============================================================

top candidate와 2등 차이가:

< 0.0007

이면:
TIE / UNCERTAIN

으로 표시.

차이 0.0002로
"압도적 winner"라고 쓰지 마라.

============================================================
55. Negative control
============================================================

이번에도 최소:

label permutation 1회

future feature trap

constant score

prefix mutation

검증.

negative control을 candidate selection에 쓰지 않는다.

============================================================
56. Causality hard gate
============================================================

새 feature API는:

historical
+
current online point
+
past state

만 사용.

다음 argument 자체를 feature class에 주지 마라:

tau
online_total_length
future_iterator
future_suffix

============================================================
57. Prefix invariance
============================================================

모든 새로운 feature family:

최소 random 100 prefix tests.

finalist:
300 이상.

검사:

future suffix 변경
future length 변경
reordered series
reset

현재 prefix prediction exact.

============================================================
58. Same-time metric alignment
============================================================

TS-AUC는 같은 online t에서 ranking.

따라서 evaluation에서:

within-time Spearman
pair ordering accuracy
time contribution

계산.

단 time index 자체만으로 score 만드는 건 의미 없음.

============================================================
59. Time fracture
============================================================

필수:

0-9
10-24
25-49
50-99
100-199
200-399
400+

B2처럼 late gain만으로
전체 점수가 올라가는지 반드시 확인.

============================================================
60. Break-age fracture
============================================================

0-4
5-9
10-19
20-49
50+

보고.

하지만 early 구간만 위해
overall TS-AUC 희생 금지.

============================================================
61. Mechanism fracture
============================================================

기존:

SCALE
DEPENDENCE
MIXED
WEAK
MEAN
TAIL

평가.

support 부족 subset은
"supported"라고 주장 금지.

post-hoc mechanism은
training feature로 사용 금지.

============================================================
62. H-regime stress
============================================================

Historical H-only regime partition 유지.

finalist가:

Research CV positive
Shadow positive
하지만 H leave-one-regime catastrophic negative

이면 warning.

============================================================
63. Synthetic suite
============================================================

고정 suite 유지:

Gaussian
AR
ARCH
heavy-tail
scale up/down
dependence
outlier
repeated outlier
mixed

새 candidate 보고 generator parameter 변경 금지.

============================================================
64. Long-null
============================================================

1024
4096
16384

검사.

새 persistent state가
eventual false alarm을 심각하게 악화시키는지 확인.

============================================================
65. Outlier impulse response
============================================================

finalist마다:

single outlier t0 이후
score/evidence trajectory

0
1
2
4
8
16
32
64
128
256
512

추적.

반복 outlier도 별도.

============================================================
66. Latency gate
============================================================

현재 BL80 runtime을 reference.

finalist:

point median
p95
p99
historical initialization
state bytes
10M direct

측정.

3x 이하 기본 허용.

+0.003급 강한 성능이면
조금 더 느려도 별도 검토 가능.

============================================================
67. Shared computation
============================================================

B2/D2/new scale model에서:

AR8
ARCH
innovation
conditional variance

중복 계산 최소화.

하지만 runtime optimization은
scientific correctness 확정 후.

============================================================
68. Candidate budget monitor
============================================================

orchestrator가 실시간 기록:

screen_count
full_count
family_count
shadow_candidate_count

budget 초과 시 hard stop.

============================================================
69. Adaptive bottleneck manager
============================================================

45~60분마다:

BOTTLENECK_BOARD

갱신.

초기 예상:

B1 scale generalization
B2 persistent-vs-transient shock
B3 heavy-tail null calibration
B4 long-null accumulation
B5 independent signal discovery
B6 NEXT5 selection uncertainty

최대 ACTIVE 2개.

============================================================
70. 기대가치 계산
============================================================

각 bottleneck:

estimated TS-AUC impact
× evidence
× generalization relevance

대비

overfit risk
× implementation complexity
× runtime

로 우선순위 설정.

============================================================
71. Adaptive loop
============================================================

기본 메뉴 완료 후:

while productive_time < 10h:

    1. bottleneck board 갱신
    2. 실패 ledger 검색
    3. 가장 가치 높은 병목 선택
    4. 새 hypothesis 작성
    5. finite menu lock
    6. code
    7. tests
    8. screen
    9. full if pass
    10. bootstrap
    11. fractures
    12. promote/reject
    13. checkpoint

계속.

============================================================
72. Adaptive rules
============================================================

CASE A:
Unified scale model이 +0.002 이상
→ 해당 family 집중.

CASE B:
B2/D2 fusion이 중복이라 gain 없음
→ scale family 확장 중지.

CASE C:
Persistent-vs-transient가 outlier와 real DEV 모두 개선
→ 새 generation 후보.

CASE D:
robust null은 synthetic만 개선
→ 폐기.

CASE E:
새 independent expert가 standalone 약하지만 blend +0.0015
→ finalist.

CASE F:
ResearchPool에서 좋은데 Shadow negative
→ generalization rejection.
다시 튜닝 금지.

CASE G:
Shadow positive strong
→ candidate freeze.
추가 DEV mining 중지.

CASE H:
약 12개 구조적 가설 모두 < +0.0007
→ NEXT5_DEV_SATURATION_HIGH.
남은 시간은 robustness/reproduction/analysis.

============================================================
73. 같은 family micro-search 금지
============================================================

예:

scale multiplier
0.6
0.65
0.7
0.75

무한 grid 금지.

고정 small grid만.

============================================================
74. 새로운 seed 금지
============================================================

random seed가 필요한 모든 것:

NEXT5 시작 때 freeze.

모델 seed
pair sampling
synthetic
bootstrap

전부.

결과 보고 seed 변경 금지.

============================================================
75. Bootstrap policy
============================================================

ResearchPool finalist:

2000 whole-series bootstrap.

Shadow:
가능하면 analytic/paired point estimate 우선.

Shadow에서 bootstrap을 수십 seed 돌려
유의성을 억지로 찾지 않는다.

============================================================
76. FINAL SEAL policy
============================================================

A/B/C/D는 이번에도 절대 열지 않는다.

NEXT5_SHADOW와 LOCAL SEAL을
같은 것으로 취급하지 마라.

LOCAL SEAL은 훨씬 더 귀한
실제 independent reserve다.

============================================================
77. 언제 LOCAL SEAL을 추천할지
============================================================

이번 세션 실제 open은 금지.

다만 종료 보고서에서:

SEAL_OPEN_RECOMMENDATION_NEXT5.md

작성.

YES recommendation 조건 예:

NEXT5_SHADOW에서 새 generation positive
AND
ResearchPool + Shadow 모두 안정적
AND
구조적으로 새로운 signal
AND
기존 DEV saturation 때문에 추가 내부 검증 효용 낮음

그 경우에만
다음 세션에서 SEAL-A 500 논의.

============================================================
78. Research time plan
============================================================

0:00–0:40
state restore
preflight
NEXT5 shadow lock
research splits
family menus

0:40–2:30
Scale Evidence Unification
S family

2:30–4:00
Persistent vs Transient
P family

4:00–5:15
Windowed evidence
W family

5:15–6:30
E-process / robust sequential likelihood
E family

6:30–7:30
Robust Null
R family

7:30–8:15
Mean/independent low-budget family

8:15–9:15
Cross-family interaction
full validation finalists

9:15–10:00+
adaptive highest-value research
finalist lock

10h productive research 완료 후:

SHADOW finalist evaluation

그 이후:
50~60분 final audit/handoff.

============================================================
79. 중요한 순서
============================================================

SHADOW는
productive 10h 연구가 반드시 끝난 뒤에만
열어야 하는 것은 아니다.

하지만:

candidate family menus 완료
finalist 최대 3개 freeze
FINALIST_MANIFEST hash 생성

이 세 조건이 충족된 이후에만 가능.

가능하면 연구 후반부까지 미룬다.

============================================================
80. Checkpoint
============================================================

30~45분마다:

NEXT5_CHECKPOINT_XXX.json

필드:

wall_seconds
productive_seconds
current_family
current_candidate
screen_count
full_count
research_best
shadow_opened
shadow_finalists
seal_status
resource_usage
failures
next_task

============================================================
81. Crash recovery
============================================================

completed experiment는 재실행하지 않는다.

hash + receipt 확인.

RUNNING에서 crash:

safe restart 최대 1회.

동일 failure 반복:
QUARANTINE.

전체 run 계속.

============================================================
82. Research process resource rules
============================================================

RAM 32GB 가정하되
실제 시스템 탐지값 authority.

RAM >85%:
concurrency 감소.

feature matrix:
float32 / memmap / chunk.

기존 Python 환경/global package 파괴 금지.

============================================================
83. Failure logging
============================================================

각 실패:

BUG
NEGATIVE_RESULT
SCREEN_FAIL
FULL_FAIL
SHADOW_FAIL
LEAKAGE_FAIL
LATENCY_FAIL
QUARANTINED

구분.

============================================================
84. New generation gate
============================================================

NEXT5 ResearchPool + Shadow 종합 판단.

NEW_GENERATION 후보는:

ResearchPool strong
AND
Shadow positive
AND
primary/ALT stability
AND
no leakage
AND
no major catastrophic weakness

필수.

단 기존 full-DEV exact threshold +0.0025와
직접 숫자를 기계적으로 비교하지 않는다.
ResearchPool/Shadow 구조가 달라졌기 때문.

============================================================
85. 최종 모델 역할
============================================================

최대:

FROZEN_PRIMARY
NEXT4_ROBUSTNESS
NEXT5_PERFORMANCE
NEXT5_ROBUSTNESS

4개.

B2 near-miss는 research reference로만 유지 가능.

============================================================
86. Final audit
============================================================

약 50~60분.

반드시:

raw SHA
DEV/shadow/seal membership
seal access 0
reduced usage 0
public usage 0

scorer parity
prefix invariance
future suffix traps
future length traps

selected model offline/stream parity
save/load parity
determinism

complete test suite

selected new candidate fresh raw reproduction

latency/state

source manifest

수행.

============================================================
87. Fresh reproduction
============================================================

새 NEXT5 role이 생겼을 때만:

raw ResearchPool/full DEV
→ feature
→ fit
→ inference

fresh output dir에서 재현.

selection score는 fresh all-DEV in-sample에서 다시 계산하지 않는다.

quality authority는 heldout OOF / shadow.

============================================================
88. Final artifacts
============================================================

00_GPT_READ_FIRST_NEXT5.md
01_GPT_HANDOFF_NEXT5.md

NEXT5_PREFLIGHT.json

NEXT5_SHADOW_LOCK.json
NEXT5_SHADOW_GUARD.json
NEXT5_SPLIT_LOCK.json
NEXT5_RANDOMNESS_LOCK.json

HYPOTHESIS_FAMILY_LOCK_NEXT5.json
CANDIDATE_BUDGET_NEXT5.json

SCALE_UNIFICATION_RESULTS.csv
PERSISTENCE_RESULTS.csv
WINDOWED_EVIDENCE_RESULTS.csv
EPROCESS_RESULTS.csv
ROBUST_NULL_RESULTS.csv
MEAN_SPECIALIST_RESULTS.csv
CROSS_FAMILY_RESULTS.csv

MODEL_RESULTS_NEXT5.csv
CV_RESULTS_NEXT5.csv
PAIRED_BOOTSTRAP_NEXT5.csv

ALT_RESULTS_NEXT5.csv
REGIME_RESULTS_NEXT5.csv
MECHANISM_RESULTS_NEXT5.csv
TIME_FRACTURE_NEXT5.csv
BREAK_AGE_NEXT5.csv

COUNTEREXAMPLE_RESULTS_NEXT5.csv
LONG_NULL_RESULTS_NEXT5.csv
OUTLIER_IMPULSE_NEXT5.csv

FINALIST_MANIFEST_NEXT5.json
SHADOW_RESULTS_NEXT5.csv

BOTTLENECK_BOARD_INITIAL_NEXT5.json
BOTTLENECK_BOARD_FINAL_NEXT5.json
BOTTLENECK_HISTORY_NEXT5.csv

LEAKAGE_AUDIT_NEXT5.json
PREFIX_INVARIANCE_NEXT5.json
DETERMINISM_NEXT5.json
LATENCY_NEXT5.json

DEV_CANDIDATES_NEXT5.json

SEAL_GUARD_NEXT5.json
SEAL_OPEN_RECOMMENDATION_NEXT5.md
PUBLIC_SUBMISSION_RECOMMENDATION_NEXT5.md

REPRODUCTION_NEXT5.json
FINAL_AUDIT_NEXT5.json

LIMITATIONS_NEXT5.md
NEXT_SESSION_PLAN_NEXT5.md

EXPERIMENT_LEDGER.csv
SOURCE_MANIFEST_NEXT5.json

그리고:

STRUCTURAL_BREAK_GPT_HANDOFF_NEXT5_<timestamp>.zip

생성.

ZIP에서 제외:

raw competition data
seal values/labels
credentials
full real point OOF
feature cache
old archives

============================================================
89. 00_GPT_READ_FIRST_NEXT5.md
============================================================

한국어로 다음을 정확히 답한다.

1. productive research time
2. wall-clock
3. family 수
4. screen 수
5. full 수
6. learner fit 수

7. 시작 primary
8. 시작 score

9. ResearchPool best
10. Shadow finalist 목록
11. Shadow scores
12. 최종 selected model
13. 최종 estimated gain

14. B2/D2 fusion 결과
15. scale likelihood 결과
16. robust null 결과
17. persistence 결과
18. windowed evidence 결과
19. e-process 결과
20. mean specialist 결과
21. new independent expert 여부

22. 5-fold ResearchPool 결과
23. ALT
24. regime
25. mechanism
26. time fracture
27. break age

28. repeated outlier
29. single outlier
30. long-null

31. bootstrap
32. winner gap
33. family multiplicity
34. shadow confirmation

35. latency
36. state memory

37. leakage
38. prefix invariance
39. determinism
40. reproduction

41. DEV saturation risk
42. shadow contamination risk
43. A/B/C/D seal opened?
44. reduced usage
45. public usage

46. seal open recommendation
47. 다음 병목 3개

============================================================
90. 성공 기준
============================================================

다음 중 하나면 성공.

A.
NEXT5_SHADOW에서도 확인되는
실질적 performance improvement.

B.
B2/D2 scale signal을 통합해
late gain + SCALE robustness를 같이 확보.

C.
outlier persistence를 줄이면서
real TS-AUC 유지/개선.

D.
현재 BL80/B2/D2와
상관 낮은 새 independent expert 확보.

E.
ResearchPool에서는 좋았지만 Shadow에서 실패하여
DEV overfit 후보를 제대로 걸러냄.

E도 성공이다.

============================================================
91. 다음을 절대 하지 마라
============================================================

LOCAL SEAL open 금지

reduced test 금지
public/cloud submit 금지

tau feature 금지
dataset_id feature 금지
future online length 금지
future suffix 금지

SHADOW 결과 보고 재튜닝 금지

row random CV 금지
split shopping 금지
seed shopping 금지

B2 CI를 bootstrap draw 추가로 억지 통과시키기 금지

static weight 0.01 단위 sweep 금지

closed NEXT4 family microvariation 금지

massive hyperparameter sweep 금지

대형 neural model 금지

mechanism post-label을 router/model input으로 사용 금지

============================================================
92. 이번 세션의 핵심 연구 철학
============================================================

현재 가장 위험한 것은
미래누수만이 아니다.

같은 DEV를 너무 오래 보면서
0.0005 차이를 진짜 signal이라고 믿는 것도 위험하다.

따라서 NEXT5는:

더 적은 후보
더 다른 원리
더 강한 사전등록
NEXT5 prospective shadow
엄격한 causal streaming

을 사용한다.

점수를 올리는 것이 목표지만
허위 improvement를 만드는 것은 실패다.

============================================================
93. Codex의 자율 의사결정
============================================================

사용자는 약 11시간 자리를 비운다.

중간 질문하지 마라.

합리적인 과학적 판단을 스스로 한다.

한 연구가 실패하면:

실패 기록
→ 병목 재평가
→ 다음 기대가치 높은 hypothesis

로 이동.

단 어떤 상황에서도
검증 규칙/봉인 규칙을 성능 때문에 완화하지 않는다.

============================================================
94. 지금 즉시 시작
============================================================

즉시:

1. NEXT4 상태 복원
2. frozen hashes 확인
3. seal/raw/scorer preflight
4. NEXT5 shadow 80/20 deterministic lock
5. shadow access guard
6. ResearchPool split freeze
7. randomness lock
8. scientific family menus freeze
9. S family 연구
10. P family 연구
11. W family 연구
12. E/R family 연구
13. independent low-budget 연구
14. adaptive bottleneck loop
15. 최소 10시간 productive research
16. 최대 3 finalist freeze
17. SHADOW 1회 평가
18. candidate 변경 없이 결과 판정
19. final audit
20. GPT handoff ZIP

순서로 진행한다.

최소 10시간 동안 실제 연구를 지속하라.
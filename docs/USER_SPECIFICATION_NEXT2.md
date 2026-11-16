Structural Break Real-Time 프로젝트의 다음 장기 연구 세션을 수행한다.

프로젝트 위치:
C:\Users\minsu\Documents\Structural Break

이전 프로젝트를 그대로 이어서 작업한다.
새 프로젝트를 만들지 마라.

이번 세션의 핵심은
현재 확보된 0.6216대 generation을 무작정 더 튜닝하는 것이 아니다.

목표는:

1. 현재 성능 상승의 실제 원인을 최소 구성으로 분해
2. 불필요한 GLR / serial correction / memory 계산 제거 가능성 확인
3. variance decrease / heavy-tail / ARCH 등 발견된 counterexample 보강
4. long-horizon null behavior 개선
5. 현재 variance generation과 상관이 낮은 complementary signal 탐색
6. 가능하면 새로운 generation-level improvement 확보
7. LOCAL SEAL A/B/C/D는 전부 계속 미개봉

이다.

============================================================
0. 현재 상태
============================================================

기존 artifact 기준으로 대략 다음 상태다.

- 전체 train series: 10,000
- DEV: 8,000
- LOCAL FINAL RESERVE: 2,000
- reserve는 500 / 500 / 500 / 500으로 deterministic partition 완료
- SEAL_A = 500
- SEAL_B = 500
- SEAL_C = 500
- SEAL_D = 500

네 seal 모두 아직 unopened.

이전 official EWMA:
약 0.514

이전 generation:
약 0.611

현재 formal PRIMARY:
J102 계열
약 0.62163

현재 scientific reference:
Y001 long-variance + Bayes-lite 계열
약 0.62163

현재 simplification / robustness 후보:
B104 계열
약 0.62149
worst fold는 J102보다 오히려 조금 강함

이전 분석상 실제 큰 상승은
J102의 serial correction 자체가 아니라:

- raw innovation energy
- conditional variance / ARCH
- variance-change evidence
- long change-age evidence
- Bayes-lite

에서 발생한 것으로 보인다.

J102 vs Y001 차이는 사실상 0.

explicit GLR 제거도 거의 손실이 없었던 것으로 보인다.

단 위 숫자는 프롬프트를 authority로 사용하지 말고
반드시 기존 artifact와 source를 직접 읽어서 복원한다.

============================================================
1. 이번 세션의 시간 규칙
============================================================

이번 세션은:

순수 연구 최소 6시간

이다.

최소 6시간이 끝난 이후에만
final audit / reproduction / handoff 단계로 이동한다.

Final audit:
45~60분

따라서 목표 총 wall-clock:

약 6시간 45분 ~ 7시간

중요:

`--hours 6`이 최대 timeout일 뿐이라
candidate list가 빨리 끝나고 2시간 만에 종료하는 상황을 금지한다.

실제 scheduler는:

research_start = monotonic()
minimum_end = research_start + 6 hours

while monotonic() < minimum_end:
    의미 있는 연구를 계속한다.

sleep으로 시간 채우기 금지.

queue가 끝나면 adaptive backlog로 이동.

중간 재부팅/오류에 대비해 checkpoint와 resume 지원.

============================================================
2. 가장 먼저 이전 상태를 복원
============================================================

다음과 관련된 최신 artifact를 모두 읽는다.

- 00_GPT_READ_FIRST_NEXT.md
- 01_GPT_HANDOFF_NEXT.md
- MODEL_RESULTS_NEXT.csv
- CV_RESULTS_NEXT.csv
- PAIRED_BOOTSTRAP_RESULTS.csv
- SEQUENTIAL_STAT_RESULTS.csv
- GLR_RESULTS.csv
- BAYES_LITE_RESULTS.csv
- WHITENING_RESULTS.csv
- INNOVATION_NORMALIZATION_RESULTS.csv
- HISTORICAL_REGIME_RESULTS.csv
- MECHANISM_FRACTURE_RESULTS.csv
- BREAK_AGE_NEXT.csv
- TIME_FRACTURE_NEXT.csv
- CURRENT_TOP_FORENSIC.md
- LIMITATIONS_NEXT.md
- NEXT_SESSION_PLAN.md
- EXPERIMENT_LEDGER.csv
- SEAL_OPEN_RECOMMENDATION.md

그리고:

src/
scripts/
configs/
tests/
experiments/
artifacts/

를 직접 확인한다.

정확한 candidate 이름과 score를 source/artifact에서 복구한다.

최소 세 모델을 immutable reference로 freeze:

FROZEN_PRIMARY
FROZEN_SCIENTIFIC_REFERENCE
FROZEN_ROBUST_REFERENCE

기존 모델 파일/feature config를 수정하지 마라.

새 실험은 새 generation/version으로 만든다.

============================================================
3. Seal / reduced / leaderboard 정책
============================================================

이번 세션에서:

SEAL_A_OPENED = false
SEAL_B_OPENED = false
SEAL_C_OPENED = false
SEAL_D_OPENED = false

유지.

Seal label/tau/prediction 접근 금지.

기존 seal access guard를 다시 테스트한다.

Reduced test:
신규 사용 0회.

Public leaderboard:
신규 제출 0회.

Crunch submission:
하지 마라.

이번 연구는 DEV 8,000으로만 진행.

============================================================
4. 현재 generation을 먼저 분해한다
============================================================

이번 세션의 첫 핵심 연구.

현재 0.6216 모델에서
실제 필요한 component가 무엇인지 정확히 파악한다.

현재 candidate genealogy를 복원하고
component graph를 만든다.

예상 주요 component:

BASE
+
AR residual / innovation
+
raw innovation energy
+
conditional variance
+
ARCH1
+
long-age scale evidence
+
Bayes-lite
+
explicit GLR
+
GLR memory
+
serial correction
+
other memory features

정확한 구성은 source를 authority로 한다.

결과:

CURRENT_GENERATION_COMPONENT_GRAPH.md

작성.

============================================================
5. Distillation / 최소구성 연구
============================================================

목표:

현재 약 0.6216 성능을
더 단순한 feature/state/runtime으로 유지할 수 있는지 확인.

한 번에 component 하나 또는 논리적 family 하나만 제거.

우선 ablation:

D0
full current reference

D1
serial correction 제거

D2
explicit GLR 제거

D3
GLR memory 제거

D4
GLR + GLR memory 제거

D5
불필요한 evidence memory 제거

D6
short-age redundant evidence 제거

D7
Bayes-lite 제거

D8
ARCH 제거

D9
raw innovation energy 제거

D10
conditional variance 제거

D11
long-age evidence 제거

이 중 이미 이전 세션에서 정확히 동일 실험이 존재하면
재실행하지 말고 cache/evidence를 재사용.

단 configuration이 달라졌다면 새로 검증.

목표는 단순히 importance를 보는 것이 아니라:

- DEV mean
- median
- worst fold
- bootstrap delta
- latency
- state size
- historical init cost

를 비교.

============================================================
6. Simplification promotion rule
============================================================

복잡한 기존 모델보다 단순 모델을 채택하려면
다음 non-inferiority gate를 사전 freeze.

Simplified candidate가:

mean >= frozen_primary_mean - 0.0005

AND

median >= frozen_primary_median - 0.001

AND

worst_fold >= frozen_primary_worst - 0.001

AND

paired bootstrap
P(delta > -0.0005) >= 0.90

AND

alternate validation에서 catastrophic degradation 없음

AND

runtime 또는 state complexity가 실제로 감소

해야 한다.

점수를 본 뒤 gate 수정 금지.

============================================================
7. 진짜 runtime simplification을 확인
============================================================

중요:

feature column만 제거했는데
StreamingState 내부에서는 GLR이나 기타 statistic을 계속 계산하면
진짜 simplification이 아니다.

따라서:

- feature column 사용 여부
- state update 계산 여부
- historical init 계산 여부

를 모두 분리해서 profile.

B104 같은 후보가
GLR feature는 제거했지만 runtime에서는 계속 계산한다면
`B104_RUNTIME_PRUNED` 버전을 만든다.

불필요한 state 계산 자체를 제거.

정확한 latency gain 측정.

============================================================
8. Counterexample suite를 고정
============================================================

이전 세션에서 variance generation에 다음 약점 가능성이 발견됐다.

- variance decrease
- 일부 ARCH 변화
- heavy-tail 변화
- isolated extreme outlier
- 특정 scale-change 방향
- long null에서 eventual false alarm 증가

이걸 score를 보고 계속 generator를 바꾸지 않도록
이번 세션 시작 시 synthetic stress suite를 freeze.

최소 DGP:

N0
stationary Gaussian null

N1
stationary AR null

N2
stationary ARCH null

N3
heavy-tail stationary null

M1
mean increase

M2
mean decrease

V1
variance ×1.5

V2
variance ×2

V3
variance ×0.7

V4
variance ×0.5

A1
ARCH persistence increase

A2
ARCH persistence decrease

T1
Gaussian → Student-t heavy-tail

T2
heavy-tail → Gaussian

D1
positive autocorrelation change

D2
autocorrelation sign reversal

O1
single giant outlier but no persistent break

X1
mixed scale + dependence

random seeds와 parameters를 먼저 lock.

synthetic 결과를 competition truth라고 주장하지 마라.

오직 mechanistic diagnostic.

============================================================
9. Scale-down 반례 연구
============================================================

현재 variance evidence가 variance increase에는 강하고
decrease에서 약할 가능성을 집중 확인.

후보 statistic:

- signed log variance ratio
- two-sided absolute log variance ratio
- upper CUSUM / lower CUSUM 분리
- conditional variance forecast ratio
- short/long log-scale difference
- lower-scale accumulated evidence
- symmetric variance likelihood ratio

예:

r_t = log((short_var + eps)/(reference_var + eps))

positive evidence:
max(r_t, 0)

negative evidence:
max(-r_t, 0)

absolute:
abs(r_t)

각 방향을 따로 feature로 제공하는 방식 비교.

단 feature 폭발 금지.

============================================================
10. Heavy-tail robustness
============================================================

squared residual은 extreme observation 하나에 너무 민감할 수 있다.

다음을 비교:

raw z²
winsorized z²
Huberized energy
log1p(z²)
abs(z)
clipped abs(z)
robust scale score
historical ECDF tail probability

clip threshold는
candidate score를 보고 계속 수정하지 않는다.

사전 작은 grid:

|z| cap:
3
4
6

정도.

목표:

heavy-tail null false positive 감소
+
variance break 성능 유지.

============================================================
11. ARCH robustness
============================================================

현재 ARCH1 evidence가 중요한 것으로 보인다.

검토:

ARCH1
ARCH2
EWMA conditional variance

비교.

하지만 무거운 full GARCH fitting은 우선순위 낮음.

historical-only fitting.

online refit 금지.

조건부 분산 forecast error:

q_t =
log((innovation² + eps)/(forecast_variance + eps))

를 이용한 two-sided accumulated evidence도 검토.

============================================================
12. Long-horizon null 문제
============================================================

이전 long-null stress에서
fixed threshold eventual alarm rate가 horizon과 함께 증가했다.

이번 세션에서는 이것을
competition metric을 해치지 않는 범위에서 분석.

중요:

TS-AUC가 primary metric.

fixed-threshold alarm calibration을 위해
TS-AUC를 희생하지 마라.

연구 후보:

- evidence / sqrt(effective_age)
- null expected maximum correction
- log(time) correction
- age-dependent evidence normalization
- running standardized evidence
- decayed cumulative evidence

단 순수 time-based score를 만들어서는 안 된다.

모든 correction은 detector evidence의
null growth를 정규화하는 목적으로만 사용.

같은 t에서 series ranking이 어떻게 변하는지 확인.

============================================================
13. Evidence memory forensic
============================================================

break 이후 target은 계속 1이기 때문에
memory가 도움이 될 수 있다.

하지만 memory가 null에서도 계속 최대값을 누적하면
long-horizon false alarm을 만든다.

비교:

cumulative max
decayed max
EWMA evidence
leaky accumulator
resettable accumulator
posterior-like memory

각각:

DEV TS-AUC
late-break
long-null
no-break false alarm
latency

평가.

============================================================
14. Long-age Bayes-lite를 최소화
============================================================

현재 long-age Bayes evidence는 실제로 gain이 있는 것으로 보인다.

하지만 현재 age grid 전체가 필요한지 확인.

현재 실제 age grid를 source에서 확인하고
그에 기반해 pruning.

예시 실험:

all ages

vs

short only

medium only

long only

long sparse grid

예:
32 / 64 / 128 / 256 / 512

또는
64 / 128 / 256 / 512

정확한 grid는 기존 config에 맞춰 설계.

목표:
동일 성능에 더 적은 state / 계산량.

============================================================
15. GLR을 진짜 버릴 수 있는지 확인
============================================================

현재 evidence상 explicit GLR의 marginal gain이 거의 없을 가능성이 있다.

다음 비교:

FULL

NO_GLR_COLUMNS

NO_GLR_RUNTIME

NO_GLR_AND_MEMORY_RUNTIME

Bayes-lite only

GLR only

Bayes + GLR

paired bootstrap으로 확인.

GLR contribution이 없으면
새 clean candidate에서는 완전히 제거.

============================================================
16. J102 serial correction 검증
============================================================

J102와 Y001가 거의 동률이라면
serial correction은 promotion할 근거가 약하다.

이번에는:

- feature-level difference
- runtime difference
- prediction correlation
- disagreement cases
- regime slices
- dependency slice

를 분석.

Serial correction이 특정 dependence mechanism에서는 유효하지만
전체에서 상쇄되는지 확인.

유의미한 complement가 없다면
scientific core에서 제외.

============================================================
17. Complementary signal 탐색
============================================================

현재 variance generation과
동일한 신호를 더 복잡하게 만드는 것만 반복하지 마라.

현재 primary prediction과 상관이 낮으면서
standalone information이 있는 detector를 찾는다.

우선순위:

A. robust mean-change evidence
B. variance-normalized dependence change
C. CDF/tail shape change
D. sign/rank based evidence
E. whiteness departure

각 후보에 대해:

standalone TS-AUC
prediction correlation with primary
incremental blend gain
mechanism-specific gain

측정.

좋은 complementary candidate 조건 예:

standalone TS-AUC > 0.54

AND

primary와 prediction correlation이 지나치게 높지 않음

AND/OR

특정 mechanism에서 명확한 complementary advantage.

============================================================
18. Mean complement
============================================================

현재 mean-dominant slice가 약할 가능성이 있다.

하지만 mean family 전체 성능은 낮았으므로
과도하게 시간 쓰지 마라.

Whitened innovation에서:

two-sided mean CUSUM
signed Bayes mean evidence
rank-sign accumulator
Page-Hinkley

소수만 비교.

목표:

mean slice 개선

단 전체 TS-AUC 악화 시 primary에 넣지 않는다.

============================================================
19. Dependence complement
============================================================

variance level 영향을 먼저 제거한 innovation에서:

z_t * z_(t-1)
z_t * z_(t-2)
z_t * z_(t-4)
z_t * z_(t-8)

및

z² autocorrelation
abs(z) autocorrelation

중 가장 저비용 family 비교.

특히 conditional variance normalization 후에도
dependency change가 남는지 본다.

============================================================
20. Distribution / tail complement
============================================================

historical ECDF uniform space 사용.

u_t = F_hist(e_t)

under null ~ Uniform.

후보:

tail mass
center mass
left/right imbalance
8-bin CDF deviation
16-bin CDF deviation
EWMA bin divergence

매 point full KS / AD 금지.

incremental.

============================================================
21. Complement blending
============================================================

complement signal이 실제로 존재할 때만 blend.

continuous weight optimization 금지.

고정 grid:

primary/complement

90/10
80/20
67/33
50/50

정도.

rank average도 가능.

각 blend:

primary CV
bootstrap
alternate splits
regime
mechanism
worst fold

평가.

============================================================
22. Residual complement model은 cross-fit만 허용
============================================================

현재 primary prediction의 오류를 학습하는
second-stage meta model을 시도하고 싶다면
반드시 strict cross-fitting.

같은 row의 in-sample primary prediction을
meta-model training에 사용 금지.

구조:

inner fold
    ↓
OOF primary prediction
    ↓
meta training

outer fold
    ↓
fully unseen meta evaluation

이 구조가 너무 비싸면
이번 세션에서 생략.

누수 위험이 조금이라도 있으면 하지 마라.

============================================================
23. 새로운 model complexity는 제한
============================================================

이번 세션에서는:

LightGBM
HistGB
linear
fixed statistical blend

정도 사용.

대형 neural model 금지.

Transformer / LSTM / GRU 금지.

massive hyperparameter sweep 금지.

현재 gain은 feature/statistic에서 나오고 있다.

============================================================
24. Candidate screening
============================================================

모든 candidate를 5-fold full CV 돌리지 않는다.

미리 고정된 screening setup 사용.

STAGE 1:
cheap deterministic screen

STAGE 2:
유망 후보 full primary 5-fold

STAGE 3:
finalist bootstrap + alternate split + regime stress

screening subset/split을
candidate 결과 보고 변경 금지.

============================================================
25. New candidate promotion
============================================================

현재 formal PRIMARY 대비:

NEW PERFORMANCE CANDIDATE:

mean delta >= +0.0015

AND

median non-worse

AND

worst fold degradation >= -0.001

AND

bootstrap P(delta > 0) >= 0.85

AND

alternate split 3개 중 2개 이상 non-negative

AND

regime catastrophic failure 없음

이어야 함.

============================================================
26. New generation gate
============================================================

새로운 SEAL-worthy generation이라고 부르려면:

mean delta >= +0.0035

OR

mean delta >= +0.0025
AND
bootstrap P > 0.95
AND
alternate split 모두 양수
AND
new independent mechanism에서 gain

이어야 한다.

이번에도 실제 seal은 열지 않는다.

오직:

SEAL_OPEN_RECOMMENDATION_NEXT2.md

에 recommendation만.

============================================================
27. Simplification candidate와 performance candidate를 분리
============================================================

이번 세션 종료 시 후보를 두 종류로 구분.

A. PERFORMANCE candidate
현재보다 점수 상승.

B. DISTILLED candidate
점수는 거의 동일하지만:
- feature 감소
- runtime 감소
- state 감소
- robustness 유지

가능.

Distilled model이
성능 0.0002 낮더라도
복잡도 30~50% 감소하면 매우 가치 있을 수 있다.

============================================================
28. Latency / state benchmark
============================================================

finalists:

historical initialization
streaming update
model prediction
total

각각 측정.

median
p95
p99

state bytes per series

model file size

1M points
10M points

예상 시간.

특히 runtime pruning 효과를 숫자로 보여라.

============================================================
29. Long-null stress는 formal diagnostic으로 유지
============================================================

최소 horizons:

1024
4096
16384

null generators:

Gaussian
AR
ARCH
heavy-tail

각 candidate에 대해:

score quantiles
max evidence growth
fixed-threshold eventual alarm diagnostic

기록.

단 이걸 primary competition score 대신 사용하지 마라.

============================================================
30. Synthetic counterexample 사용 규칙
============================================================

synthetic stress에서 좋아졌다고
competition candidate를 자동 promote하지 않는다.

synthetic은:

mechanistic safety
diagnostic
counterexample

용도.

최종 authority는
real DEV TS-AUC + robustness.

============================================================
31. Prediction disagreement forensic
============================================================

final candidates끼리:

Pearson
Spearman
rank disagreement
top 1%
top 5%
false positive disagreement
mechanism disagreement

분석.

ensemble은 complement가 있을 때만.

============================================================
32. Adaptive loop
============================================================

기본 backlog가 끝나도
6시간 미만이면 종료 금지.

반복:

1. current best / distilled best 확인
2. 가장 약한 mechanism 확인
3. counterexample 확인
4. one hypothesis 작성
5. statistic 1개 변경
6. screen
7. full CV if justified
8. bootstrap if finalist
9. robustness
10. accept/reject
11. ledger
12. checkpoint

============================================================
33. Adaptive priority rules
============================================================

CASE A
GLR 제거해도 동일
→ GLR runtime 완전 제거

CASE B
Bayes long-age가 핵심
→ age grid 최소화 / normalization 연구

CASE C
scale-down이 약함
→ two-sided variance evidence 집중

CASE D
heavy-tail null에서 FP 증가
→ robust energy 집중

CASE E
long-null만 문제
→ evidence-memory normalization 집중

CASE F
complement signal 발견
→ fixed blend 연구

CASE G
complement 없음
→ 억지 ensemble 금지

CASE H
distilled model이 동일성능
→ 더 단순한 모델을 robust candidate로 유지

============================================================
34. 연구 시간 계획
============================================================

0:00–0:30
상태 복원
frozen candidates 확인
seal guards
research scheduler

0:30–1:45
current generation component graph
distillation / GLR / serial / memory ablation

1:45–3:00
variance decrease
heavy-tail
ARCH
long-null counterexample 연구

3:00–4:15
Bayes-lite / long-age 최소화
runtime pruning
distilled candidate 생성

4:15–5:15
complementary mean/dependence/tail signal 탐색

5:15–6:00
blend / adaptive improvement / finalists robustness

6시간 연구 완료 후:

6:00–6:45 또는 7:00
full reproducibility
tests
latency
determinism
audit
handoff
ZIP

실험이 예상보다 빨라지면
adaptive loop를 더 돈다.

============================================================
35. Checkpoint
============================================================

30~45분마다 또는 major experiment 후:

checkpoints/NEXT2_<NN>.json

내용:

elapsed_research_seconds
experiment_count
current_primary
current_distilled
current_performance_best
next_task
failed_tasks
git commit
artifact hashes

resume 가능하게.

============================================================
36. Experiment ledger
============================================================

필드:

experiment_id
parent
generation
hypothesis
component_change
counterexample_target
feature_count
state_size
model
screen_score
full_mean
median
worst_fold
delta
bootstrap_ci_low
bootstrap_ci_high
bootstrap_p
alt_split_1
alt_split_2
alt_split_3
regime_worst
long_null_status
synthetic_status
latency
status
reason

============================================================
37. 이번 세션에서 절대 하지 말 것
============================================================

SEAL_A open 금지
SEAL_B open 금지
SEAL_C open 금지
SEAL_D open 금지

reduced test 금지

public leaderboard 제출 금지

online future length 금지
future suffix 금지
tau feature 금지
dataset_id feature 금지

row random CV 금지
seed shopping 금지

synthetic score에 맞춘 과적합 금지

promotion gate 사후 완화 금지

J102 이름만 보고 serial correction을 핵심으로 가정 금지

0.0002 gain을 위해 feature 수십 개 추가 금지

대형 neural architecture 금지

============================================================
38. 최종 산출물
============================================================

00_GPT_READ_FIRST_NEXT2.md
01_GPT_HANDOFF_NEXT2.md

CURRENT_GENERATION_COMPONENT_GRAPH.md

DISTILLATION_RESULTS.csv
RUNTIME_PRUNING_RESULTS.csv

SCALE_DOWN_RESULTS.csv
HEAVY_TAIL_ROBUSTNESS_RESULTS.csv
ARCH_ROBUSTNESS_RESULTS.csv

LONG_NULL_RESULTS_NEXT2.csv
EVIDENCE_MEMORY_RESULTS.csv

BAYES_AGE_ABLATION.csv
GLR_NECESSITY_RESULTS.csv
SERIAL_CORRECTION_RESULTS.csv

COMPLEMENT_STANDALONE_RESULTS.csv
COMPLEMENT_CORRELATION.csv
COMPLEMENT_BLEND_RESULTS.csv

COUNTEREXAMPLE_SUITE_LOCK.json
COUNTEREXAMPLE_RESULTS.csv

MODEL_RESULTS_NEXT2.csv
CV_RESULTS_NEXT2.csv
PAIRED_BOOTSTRAP_NEXT2.csv

ALT_SPLIT_RESULTS_NEXT2.csv
REGIME_RESULTS_NEXT2.csv
MECHANISM_RESULTS_NEXT2.csv

LATENCY_NEXT2.json
STATE_MEMORY_NEXT2.json

DEV_CANDIDATES_NEXT2.json

SEAL_GUARD_NEXT2.json
SEAL_OPEN_RECOMMENDATION_NEXT2.md

REPRODUCTION_NEXT2.json
DETERMINISM_NEXT2.json
FINAL_AUDIT_NEXT2.json

LIMITATIONS_NEXT2.md
NEXT_SESSION_PLAN_NEXT2.md

EXPERIMENT_LEDGER.csv

SOURCE_MANIFEST_NEXT2.json

그리고:

STRUCTURAL_BREAK_GPT_HANDOFF_NEXT2_<timestamp>.zip

생성.

raw competition data
credentials
tokens
seal labels

ZIP에 넣지 마라.

============================================================
39. 최종 보고서에 반드시 들어갈 것
============================================================

한국어로 다음을 숫자로 보고.

1. 실제 순수 연구 시간
2. 실험 수
3. 이전 PRIMARY exact score
4. 새 best exact score
5. delta
6. 새 distilled candidate
7. distilled score loss/gain
8. feature 수 감소율
9. runtime 감소율
10. state memory 감소율
11. GLR 실제 필요 여부
12. serial correction 실제 필요 여부
13. Bayes-lite 필요 여부
14. long-age 중 핵심 age
15. scale-down 개선 여부
16. heavy-tail 개선 여부
17. ARCH counterexample 개선 여부
18. long-null behavior
19. strongest complementary detector
20. primary와 complement correlation
21. ensemble gain
22. 5-fold scores
23. worst fold
24. paired bootstrap
25. alternate splits
26. regime worst
27. mechanism weak point
28. SEAL_A opened?
29. SEAL_B opened?
30. SEAL_C opened?
31. SEAL_D opened?
32. reduced new usage
33. public submission new usage
34. 지금 seal을 열 필요가 있는가?
35. 다음 세션 최우선 3가지

============================================================
40. 최종 audit
============================================================

종료 전 별도 auditor mode.

반드시 확인:

ALL_SEALS_UNOPENED = true
REDUCED_NEW_USAGE = 0
PUBLIC_SUBMISSION_NEW = 0

DEV / seal overlap = 0

future leakage = 0

scorer parity PASS

offline / streaming parity PASS

determinism PASS

selected model save/load parity PASS

selected model independent rerun PASS

raw data unchanged

source/artifact hashes recorded

============================================================
41. 이번 세션의 성공 조건
============================================================

성공은 반드시 score가 올라가는 것이 아니다.

다음 중 하나 이상이면 성공:

A.
현재 0.6216보다
+0.002 이상 안정적 성능 상승

B.
0.6216 근처 성능을 유지하면서
feature/state/runtime를 의미 있게 단순화

C.
variance decrease / heavy-tail / ARCH 반례를
real DEV 성능 손실 없이 개선

D.
현재 variance model과 실제로 complementary한
새 signal 발견

E.
현재 generation이 왜 강한지
component-level causal evidence 확보

============================================================
42. 연구 철학
============================================================

현재는 이미 0.611 → 0.6216이라는 큰 jump를 확보했다.

따라서 이제부터:

0.6216 → 0.6219

같은 미세한 점수를 쫓아
DEV를 소모하는 것은 가치가 낮다.

이번 목표는:

"현재 generation의 본질을 이해하고
불필요한 부분을 제거한 뒤
약점을 메우거나
독립적인 새 signal을 찾는 것"

이다.

특히 현재까지의 evidence상:

serial correction ≠ 핵심일 가능성 큼

explicit GLR ≠ 핵심일 가능성 큼

Bayes long-age = 유효 가능성 높음

conditional variance / innovation energy = 핵심 가능성 매우 높음

이라는 가설을
이번 세션에서 정확히 검증한다.

============================================================
43. 지금 즉시 실행
============================================================

계획만 설명하고 멈추지 마라.

즉시:

1. 최신 상태 복원
2. frozen candidate hashes 확인
3. seal guard 확인
4. component graph
5. distillation
6. runtime pruning
7. counterexample suite
8. variance/ARCH/heavy-tail 개선
9. Bayes age ablation
10. complementary signal
11. adaptive loop
12. 최소 6시간 연구
13. final audit
14. GPT handoff ZIP

순서로 실행한다.

최소 6시간의 실질적 연구가 완료되기 전에는
최종 handoff를 작성하고 종료하지 마라.

LOCAL SEAL은 이번에도 하나도 열지 않는다.
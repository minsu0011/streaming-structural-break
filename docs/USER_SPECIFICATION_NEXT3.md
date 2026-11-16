Structural Break Real-Time 프로젝트의 NEXT3 장기 자율 연구/유지보수 런을 수행한다.

프로젝트 위치:
C:\Users\minsu\Documents\Structural Break

이전 프로젝트 전체를 이어서 작업한다.
새 프로젝트로 재시작하지 마라.

이번 작업은 단순히 정해진 실험 목록을 실행하는 세션이 아니다.

너는 앞으로 약 12시간 동안:

- 수석 시계열 연구원
- ML 검증 담당자
- 성능 엔지니어
- 프로젝트 유지보수 담당자
- 독립 감사 담당자

역할을 순환하면서,

"현재 프로젝트에서 목표 달성을 막고 있는 가장 중요한 병목이 무엇인지"

를 계속 다시 판단하고,
그 병목을 해결하기 위한 코드를 만들고,
실험하고,
결과를 검증하고,
필요하면 구현을 고치고,
성능/속도/안정성/재현성을 유지보수한다.

사용자가 자는 동안 중간 질문을 하지 마라.

사소한 오류 때문에 중단하지 마라.

한 접근이 실패하면:
실패 이유를 기록하고,
병목 목록을 다시 평가한 뒤,
다음 기대가치가 높은 작업으로 이동한다.

================================================================
0. 프로젝트의 최종 목표
================================================================

이 프로젝트의 목표는:

CrunchDAO Structural Break Real-Time competition에서

합법적인 streaming inference 제약을 지키면서
hidden test에서 높은 TS-AUC를 얻을 수 있는

- causal
- leak-free
- reproducible
- computationally feasible
- robust

모델을 만드는 것이다.

중요:

우리의 primary objective는 competition TS-AUC다.

fixed-threshold alarm FPR,
first-alarm delay,
synthetic score

등은 중요한 diagnostic이지만
competition TS-AUC보다 우선하지 않는다.

다만 높은 DEV TS-AUC가
명백한 특정 DGP 취약성이나 구현 오류를 이용해 나온 경우는
champion으로 인정하지 않는다.

================================================================
1. 현재 FROZEN 상태를 직접 복원
================================================================

프롬프트에 적힌 숫자를 authority로 사용하지 말고
최신 artifact에서 정확히 복원한다.

우선 다음 최신 파일과 관련 source를 읽는다.

00_GPT_READ_FIRST_NEXT2.md
01_GPT_HANDOFF_NEXT2.md
CURRENT_GENERATION_COMPONENT_GRAPH.md
DEV_CANDIDATES_NEXT2.json

MODEL_RESULTS_NEXT2.csv
CV_RESULTS_NEXT2.csv
PAIRED_BOOTSTRAP_NEXT2.csv
ALT_SPLIT_RESULTS_NEXT2.csv
REGIME_RESULTS_NEXT2.csv
MECHANISM_RESULTS_NEXT2.csv

DISTILLATION_RESULTS.csv
RUNTIME_PRUNING_RESULTS.csv

COMPLEMENT_STANDALONE_RESULTS.csv
COMPLEMENT_CORRELATION.csv
COMPLEMENT_BLEND_RESULTS.csv

COUNTEREXAMPLE_RESULTS.csv
LONG_NULL_RESULTS_NEXT2.csv

LIMITATIONS_NEXT2.md
NEXT_SESSION_PLAN_NEXT2.md

FINAL_AUDIT_NEXT2.json
REPRODUCTION_NEXT2.json

그리고:

src/
scripts/
configs/
tests/
artifacts/
experiments/
checkpoints/

전체 구조를 파악한다.

현재 최소한 다음 역할을 immutable reference로 선언한다.

FROZEN_DISTILLED
= 현재 J102 tree-used implementation

FROZEN_PERFORMANCE
= 현재 J102/C205 80/20 shared implementation

FROZEN_J102_SCIENTIFIC
= 원본 J102 scientific model

기존 canonical artifact를 덮어쓰지 마라.

새 작업은:

NEXT3

namespace / directory / experiment generation에서 수행한다.

================================================================
2. 현재 알려진 핵심 상황
================================================================

artifact로 다시 검증해야 하지만,
현재 연구 가설은 대략 다음과 같다.

현재 performance candidate:

J102 80%
+
C205 dependence specialist 20%

DEV TS-AUC 약 0.62327.

기존 J102 약 0.62163.

즉 약 +0.00164.

C205는 J102와 비교적 낮은 within-time rank correlation을 가진
독립 dependence signal이다.

하지만 improvement는 공짜가 아니다.

현재 mechanism diagnostic에서는:

DEPENDENCE는 개선
SCALE은 악화

라는 trade-off가 존재한다.

또:

- ARCH/tail background의 scale-down
- isolated outlier
- long-null eventual alarm
- early break detection

약점이 남아 있다.

이번 세션에서는
이 중 실제 competition 목표를 제한하는 병목을
다시 우선순위화한다.

================================================================
3. 절대 보존 규칙
================================================================

이번 NEXT3에서도:

SEAL_A_OPENED = false
SEAL_B_OPENED = false
SEAL_C_OPENED = false
SEAL_D_OPENED = false

유지한다.

Seal A/B/C/D:
각 500개.

다음 절대 금지:

- seal label 읽기
- seal tau 읽기
- seal prediction 평가
- seal TS-AUC
- seal mechanism analysis

Reduced:
신규 평가 0회.

Public leaderboard:
신규 제출 0회.

Crunch cloud submission:
0회.

`crunch_tools.test()`가 labeled reduced set을 소비하거나
기존 reduced 평가 정책을 위반할 가능성이 있으면 실행하지 않는다.

submission package 준비와
DEV/synthetic local protocol smoke만 허용.

================================================================
4. Seal guard부터 확인
================================================================

연구 시작 전에 기존 SealGuard를 실행한다.

검증:

DEV training rows에 seal = 0
DEV validation rows에 seal = 0
OOF에 seal = 0
candidate generation에 seal = 0
seal label access = blocked

하나라도 실패하면:

MODELING STOP.

원인을 먼저 해결한다.

Seal protection 실패 상태에서는
모델 실험을 하지 않는다.

================================================================
5. Raw/scorer/reproducibility preflight
================================================================

연구 시작 전 빠른 preflight.

확인:

raw X_train SHA256
raw y_train_index SHA256

이전 frozen hash와 동일.

official scorer parity PASS.

current frozen J102 prediction fingerprint 확인.

current frozen PERFORMANCE fingerprint 확인.

환경 package major mismatch 없는지 확인.

현재 source git status 기록.

결과:

artifacts/next3/preflight/NEXT3_PREFLIGHT.json

생성.

================================================================
6. 12시간 시간 계약
================================================================

이번 run의 목표 총 wall-clock은 약 12시간이다.

구성:

ACTIVE RESEARCH / MAINTENANCE:
약 11시간 10분

FINAL AUDIT / REPRODUCTION / HANDOFF:
약 50분

11시간 10분 이전에
"할 일이 없다"는 이유로 종료하지 마라.

sleep으로 시간을 채우는 것도 금지한다.

의미 있는 연구 queue가 떨어지면
Bottleneck Manager가 새로운 maintenance/research task를 생성한다.

최종 audit은 남은 시간이 약 50분이 되었을 때 시작한다.

환경 장애로 시간이 일부 손실되면
actual productive seconds를 따로 기록한다.

가짜 research time을 만들지 않는다.

================================================================
7. NEXT3 Maintenance Orchestrator를 만든다
================================================================

이번에는 장기 세션 자체가 안정적으로 유지되도록
중앙 orchestrator를 만든다.

예:

scripts/run_next3_autonomous.py

역할:

- total runtime 관리
- current research task 관리
- process heartbeat
- checkpoints
- crash recovery
- experiment ledger
- bottleneck reprioritization
- system resource monitoring
- final audit transition

30~60초마다 가벼운 heartbeat 기록.

예:

artifacts/next3/runtime/HEARTBEAT.json

내용:

timestamp
elapsed
current_phase
current_experiment
pid
last_progress
RAM
disk_free
completed_count
failed_count
current_best
next_task

================================================================
8. 장기런 hung process 방지
================================================================

실험 child process마다 progress heartbeat를 둔다.

다음 상황:

정상적인 장기 학습이 아닌데
30분 이상 progress update 없음

또는

CPU/GPU activity 거의 없고
output/checkpoint 변화 없음

이면:

HANG_SUSPECTED

처리.

1. stack/log 확인
2. 안전하면 child만 종료
3. 실패 receipt 저장
4. 동일 설정 1회만 retry
5. 또 실패하면 QUARANTINE
6. 다음 작업 진행

전체 autonomous run을 죽이지 마라.

================================================================
9. 시스템 유지보수
================================================================

30분마다:

RAM usage
available RAM
disk free
temp/cache size
child process count

확인.

RAM pressure가 85~90% 이상이면:

- concurrency 감소
- batch/chunk 감소
- stale temporary matrix release
- GC
- memmap 사용

으로 대응.

다른 프로젝트/환경 삭제 금지.

global pip uninstall 금지.

raw data 변경 금지.

frozen artifact 변경 금지.

================================================================
10. Atomic artifact 원칙
================================================================

장기 작업 중 중단돼도 artifact corruption이 없도록:

tmp file
→ fsync/close
→ atomic rename

방식 우선.

CSV/JSON ledger도 가능한 범위에서
append-safe 또는 temporary rewrite 방식.

완료 안 된 실험은:

COMPLETE

라고 쓰지 마라.

PENDING
RUNNING
FAILED
QUARANTINED
COMPLETE

를 명확히 구분.

================================================================
11. 병목 관리자 — 이번 세션 핵심
================================================================

이번에는 미리 정한 연구 방향을 맹목적으로 끝까지 따라가지 않는다.

다음 파일을 만든다.

artifacts/next3/bottlenecks/BOTTLENECK_BOARD.json

각 병목에는:

id
name
description
evidence
real_DEV_support
synthetic_support
estimated_metric_impact
confidence
overfit_risk
engineering_cost
runtime_cost
next_test
stop_rule
status

기록.

초기 후보 예:

B1
DEPENDENCE gain / SCALE loss trade-off

B2
ARCH/tail scale-down downstream information loss

B3
current ensemble routing inefficiency

B4
adaptive DEV selection uncertainty

B5
long-horizon evidence accumulation

B6
isolated outlier persistence

B7
early evidence weakness

B8
runtime / dual-model overhead

그러나 기존 결론을 그대로 확정하지 않는다.

실제 artifact를 읽은 뒤 보드를 만든다.

================================================================
12. 병목 우선순위 계산
================================================================

약 45~60분마다 Bottleneck Board를 다시 평가한다.

정성/정량적으로:

expected competition impact
× evidence confidence
× unresolved severity
× achievable improvement

대비

overfit risk
× implementation cost
× runtime cost

를 고려.

현재 가장 중요한 병목 1~2개만 ACTIVE로 둔다.

모든 병목을 동시에 수정하지 마라.

하나의 실험에 변경사항이 너무 많으면
원인 해석이 불가능해진다.

================================================================
13. 연구 task lifecycle
================================================================

모든 새 연구는 실행 전에:

HYPOTHESIS
BOTTLENECK
CHANGE
EXPECTED_EFFECT
RISKS
PRIMARY_METRIC
SECONDARY_DIAGNOSTIC
STOP_RULE

를 저장.

그 다음에만 실행.

결과를 본 뒤 hypothesis를 소급해서 만들지 마라.

================================================================
14. 현재 최우선 연구 후보 — dynamic specialist routing
================================================================

현재 evidence상 가장 직접적인 병목은:

C205를 섞으면
DEPENDENCE는 좋아지지만
SCALE은 악화한다.

따라서 단순한 고정 80/20 대신:

"현재 point에서 dependence specialist를 얼마나 신뢰할지"

를 causal하게 정하는 방법을 연구할 가치가 높다.

하지만 true mechanism label을 router에 넣으면 절대 안 된다.

post-hoc:

DEPENDENCE
SCALE
MEAN
TAIL

diagnostic label은 평가 전용.

router training input 사용 금지.

================================================================
15. Router 후보는 제한된 finite menu
================================================================

무제한 gating architecture 금지.

우선 다음 정도만 고려.

R0
현재 frozen 80/20.

R1
causal dependence-strength vs scale-strength heuristic gate.

R2
작은 logistic gate.

R3
depth-1 또는 depth-2 매우 작은 tree gate.

R4
strict nested cross-fit tiny meta-combiner.

모든 router input은:

현재 point까지 계산 가능한 causal features만.

예:

J102 score
C205 score
current scale evidence
current dependence evidence
abs/current energy
conditional forecast deviation
lag evidence magnitude

정도.

dataset_id 금지.
tau 금지.
future length 금지.

================================================================
16. Router leak 방지
================================================================

특히 meta/router model은
같은 fold의 in-sample base prediction을 사용하면 안 된다.

strict cross-fitting.

Outer validation fold V가 있을 때:

outer train
    ↓
inner folds
    ↓
J102/C205 OOF prediction
    ↓
router training

outer validation
    ↓
base models fit on outer train
    ↓
router predict

형태.

비용이 크더라도
누수 없는 결과만 인정.

shortcut 금지.

================================================================
17. Simple causal gate 우선
================================================================

복잡한 meta-model보다
먼저 해석 가능한 gate를 시험.

예:

depStrength = standardized dependence evidence
scaleStrength = standardized scale evidence

gateSignal =
depStrength - scaleStrength

또는

depStrength /
(abs(scaleStrength)+constant)

같은 아주 소수의 사전 등록식.

최종 score:

(1-w) * J102
+
w * C205

w는:

0
0.1
0.2
0.3

정도의 작은 bounded 범위.

gate가 confidence가 낮으면
J102 쪽으로 fallback.

continuous weight를 DEV에서 무한 최적화하지 마라.

================================================================
18. 핵심 목적
================================================================

Router 연구의 목적은:

DEPENDENCE gain을 유지하면서
SCALE loss를 줄이는 것.

즉 전체 mean만 보지 않는다.

최소:

overall Δ
DEPENDENCE Δ
SCALE Δ
worst fold
alternate splits
H regime

를 같이 본다.

성공 후보 예:

overall +0.001 이상

AND

DEPENDENCE gain 대부분 유지

AND

SCALE loss가 기존 -0.0043보다 의미 있게 감소

이면 과학적으로 매우 가치 있음.

================================================================
19. Negative scale evidence downstream forensic
================================================================

두 번째 핵심 병목.

이전 연구에서:

variance decrease information이
current signed energy/forecast에서 완전히 사라지는 것은 아닌데

historical heavy-tail / ARCH calibration과
downstream evidence / memory / trees를 지나면서
활용이 약해지는 정황이 있다.

이를 component 단위로 분해한다.

순서:

current signed statistic
→ H normalization
→ local EWMA
→ CUSUM
→ Bayes age
→ GLR/memory
→ tree splits
→ final prediction

각 단계에서:

variance decrease pair separation
rank information
dynamic range
saturation
sign preservation

측정.

================================================================
20. H calibration asymmetry를 정확히 조사
================================================================

raw energy는 하한이 있고
heavy-tail H에서는 positive extremes가 scale estimator를 팽창시킨다.

따라서 symmetric SD normalization이
negative/decrease evidence를 압축할 수 있다.

다만 이전에:

winsor
Huber
log
abs
robust
ECDF

등 일부 메뉴가 이미 실패했다.

반드시 기존 experiment ledger를 확인하고
정확히 동일한 실험은 반복하지 마라.

새 실험은 기존 실패와 무엇이 다른지
명시해야 한다.

================================================================
21. 새 scale-down 후보 방향
================================================================

기존과 중복되지 않는 경우에만
다음 아이디어를 검토.

A.
upper/lower semi-scale 별도 calibration

B.
historical lower-tail quantile 기반 decrease evidence

C.
energy distribution의 one-sided PIT

D.
negative evidence 전용 CUSUM

E.
positive/negative Bayes evidence 분리

F.
conditional variance ratio의 direction-specific branch

목표는:

variance increase detector를 망가뜨리지 않고
decrease 정보의 usable range만 회복.

================================================================
22. Pair-matched scale forensic
================================================================

Synthetic에서만 판단하지 않는다.

Real DEV에서 outcome 후 분석용으로
supported subset만 사용하여:

strong scale increase
strong scale decrease

signature를 나눠 볼 수 있다.

주의:

이 post-hoc signature는
evaluation diagnostic 전용이다.

training/router input 금지.

표본이 작으면
confidence interval/support를 함께 보고한다.

================================================================
23. C205 자체를 다시 이해
================================================================

C205가 왜 dependency slice를 올리고
scale slice를 내리는지 조사한다.

확인:

lag 1
lag 2
lag 4
lag 8

각 component contribution.

EWMA8
EWMA32
positive CUSUM
negative CUSUM
Bayes025
Bayes1

각 family ablation.

목표:

C205 전체 20%를 넣는 것보다
실제 complement를 만드는 subset을 찾을 수 있는가?

다만 이전 사용 DEV에 대한 과적합 위험이 있으므로
작은 사전 finite menu로만 수행.

================================================================
24. Dependence specialist pruning
================================================================

C205의 일부 lag/statistic만 실제 complement라면:

C205-lite

후보를 만들 수 있다.

조건:

standalone가 조금 낮아져도
J102 blend gain이 유지되면 성공.

평가:

overall
dependence
scale
latency
state

모두.

================================================================
25. Orthogonal complement 연구
================================================================

C205 score를 J102와 그냥 산술 혼합하는 대신
training fold 안에서만:

C205 dependence signal 중
J102가 이미 설명하는 부분

을 제거하는 orthogonalized complement를
탐색할 수 있다.

단:

OOF / cross-fit 필수.

예:

residual complement

C205_score
-
E[C205_score | J102_score, scaleEvidence]

와 비슷한 저복잡도 접근.

복잡한 residual learner 금지.

실제 incremental ranking gain이 없으면 폐기.

================================================================
26. Blend weight 재탐색 제한
================================================================

이미:

90/10
80/20
67/33
50/50

등을 사용했다.

이번에 단순 static weight를:

0.79
0.81
0.82

식으로 미세조정하면서 DEV를 캐는 행위 금지.

Static blend 80/20은 frozen reference.

새 gain은:

new information
또는
mechanism-aware routing

에서 나와야 한다.

================================================================
27. Long-null은 SECONDARY 병목
================================================================

long-null eventual FPR은 문제지만
competition TS-AUC가 primary다.

이번 12시간의 20% 이상을
alarm calibration에 소비하지 마라.

다만 새 evidence/memory 수정이
TS-AUC와 long-null 둘 다 개선할 경우는 우선순위 상승.

다음은 금지:

단순 age decay로
ranking score를 희생하면서 FPR만 줄이는 것.

================================================================
28. Memory/reset 연구
================================================================

필요할 경우 finite menu:

raw cumulative max
decayed max
leaky accumulator
reset after sustained null evidence
two-timescale memory

정도.

평가:

DEV TS-AUC
late break
long-null
isolated outlier persistence

같이 본다.

long-null 하나만 좋아졌다고 채택 금지.

================================================================
29. Outlier persistence
================================================================

단일 giant outlier 이후
hundreds of points 동안 evidence가 남는 component를 추적.

component impulse response를 만든다.

예:

outlier at t0

t0
t0+1
t0+2
...
t0+1024

동안:

raw energy
EWMA
CUSUM
Bayes
GLR
memory
J102 score
C205 score
blend score

추적.

실제 persistence 원인이 명확해지면
그 component만 최소 수정.

================================================================
30. Early detection은 낮은 우선순위
================================================================

break age 0~4에서 성능이 낮더라도
전체 metric contribution이 작고
현재 late evidence가 overall score를 크게 만든다.

따라서 early-only improvement 때문에
전체 TS-AUC를 희생하지 마라.

단 router/scale repair가
early도 공짜로 올려주면 기록.

================================================================
31. Generalization / DEV overuse 병목
================================================================

매우 중요.

현재 DEV 8,000은
이미 여러 세션에서 상당한 adaptive selection에 사용됐다.

따라서:

+0.0002
+0.0005

수준을 계속 파는 것은 위험하다.

이번 NEXT3에서 후보 promotion threshold를
결과 보기 전에 고정.

============================================================
32. NEXT3 Performance promotion gate
============================================================

현재 FROZEN_PERFORMANCE 대비
새 performance candidate는 최소:

mean Δ >= +0.0015

AND

median non-worse

AND

worst fold degradation >= -0.001

AND

paired whole-series bootstrap
P(Δ>0) >= 0.90

AND

95% CI lower bound > 0
가능하면 충족

AND

3 alternate splits 중
최소 2개 Δ > 0

AND

leave-H-regime catastrophic collapse 없음

AND

supported mechanism 하나를 개선하면서
다른 supported mechanism에서 큰 붕괴 없음

이어야 한다.

결과 보고 완화 금지.

================================================================
33. NEXT3 Generation gate
================================================================

진짜 새 generation이라고 부르려면
FROZEN_PERFORMANCE 대비:

mean Δ >= +0.0025

AND

bootstrap P > 0.95

AND

alternate split 3/3 양수

AND

새로운 독립 mechanism 또는
명확한 robustness 해결

AND

worst fold 유지

를 기본 조건으로 한다.

더 강한:

Δ >= +0.0035

이면 특히 중요.

하지만 이번 세션에서도
seal은 열지 않는다.

================================================================
34. Robustness repair candidate
================================================================

점수가 거의 같더라도 다음은 따로 가치 있음.

ROBUSTNESS_CANDIDATE

조건 예:

mean >= performance - 0.0008

AND

SCALE loss 의미 있게 감소

AND

outlier / ARCH scale-down 개선

AND

worst fold non-worse

이면 별도 후보 역할로 보존.

performance candidate와
robustness candidate를 억지로 하나로 합치지 않는다.

================================================================
35. Distilled candidate도 보존
================================================================

J102 tree-used는:

정확한 예측 동일성
+
낮은 state
+
빠른 initialization

을 갖는 안전한 fallback이다.

새 모델 때문에
이 canonical artifact를 삭제하거나 수정하지 마라.

================================================================
36. 실험 screening
================================================================

모든 아이디어 full 5-fold 금지.

NEXT3 시작 시 screening contract를 고정한다.

Stage S:
cheap deterministic screen.

Stage F:
full original 5-fold.

Stage R:
bootstrap
alternate
H regime
mechanism

유망 후보만 승격.

Failed screen을
마음에 든다는 이유로 full CV로 올리지 않는다.

예외가 필요하다면
새 실험을 보기 전에 protocol reason을 기록.

================================================================
37. Screening reuse
================================================================

이미 과거에 정확히 같은 feature/config가 평가됐다면
새 learner fit하지 마라.

artifact hash가 동일하면 reuse.

단:

code/config/hash가 다르면
동일하다고 가정 금지.

================================================================
38. New hypothesis budget 관리
================================================================

DEV adaptive overuse를 막기 위해
무제한 candidate 생성 금지.

한 병목당:

major hypothesis family 3개 정도

각 family당:

small finite menu

만 허용.

예:

router 방식 3종
scale repair 4종
dependence prune 4종

정도.

무작위 100개 hyperparameter search 금지.

================================================================
39. Bottleneck discovery 자체도 실험
================================================================

예상했던 문제가 실제 primary bottleneck이 아닐 수도 있다.

예:

C205 SCALE loss가
전체 score를 막는 가장 큰 요소가 아닐 수도 있다.

따라서 매 cycle 후:

"이 병목을 해결했을 때
실제 전체 TS-AUC 개선 가능성이 얼마나 남아 있는가?"

를 다시 추정.

작은 subset 문제에
몇 시간을 낭비하지 마라.

================================================================
40. Improvement attribution
================================================================

새 모델이 좋아지면:

무슨 변경이
얼마나 기여했는지

matched ablation을 수행.

예:

BASE
+
router only

BASE
+
scale repair only

BASE
+
router + scale repair

를 비교.

10개 변경을 한 번에 넣고
“좋아졌다” 금지.

================================================================
41. Mechanism labels 사용 규칙
================================================================

post-hoc:

DEPENDENCE
SCALE
MEAN
TAIL
MIXED
WEAK

는 evaluation slice일 뿐이다.

다음 절대 금지:

mechanism category를 feature로 입력.

mechanism category를 router training target으로 사용.

실제 inference 시 알 수 없는 정보를
selector에 이용.

================================================================
42. Metric alignment
================================================================

모델 평가 primary:

official TS-AUC.

동일 t에서 series ranking.

따라서 calibration probability 자체보다
rank improvement가 중요.

monotonic score transform만으로
개선했다고 주장 금지.

================================================================
43. Statistical evidence
================================================================

Finalist는 whole-series paired bootstrap.

최소 2,000 replicates.

최종 NEXT3 champion 후보는
가능하면 5,000 replicates까지 늘릴 수 있다.

단 bootstrap은:

fitted DEV predictions에 conditional.

adaptive model selection까지 보정한
독립 검증이라고 주장하지 않는다.

================================================================
44. Alternate split
================================================================

기존 locked ALT_1
ALT_2
ALT_3

그대로 사용.

새 seed 생성해서 좋은 split 찾기 금지.

================================================================
45. Historical regime holdout
================================================================

기존 outcome-blind H clustering을 그대로 사용.

새 candidate가:

original CV는 올라가는데
특정 H regime에서 크게 무너지면

promotion 보류.

================================================================
46. Learning curve diagnostic
================================================================

새 generation-level 후보가 나오면
필요한 경우:

25%
50%
75%
100%

training fraction curve.

단 full learner fit cost가 크면
finalists만.

목적:

dataset memorization인지
일반 signal인지 확인.

================================================================
47. Negative controls
================================================================

시간이 허용되면:

label permutation
feature time-shift invalid control

등을 제한적으로 사용해

pipeline이
가짜 signal에서도 높은 score를 내는지 확인.

이건 누수 탐지용.

실제 model selection에는 사용하지 않는다.

================================================================
48. Runtime maintenance
================================================================

finalist마다:

historical init median
point median
point p95
point p99
private state
model bytes
10M direct benchmark

측정.

FROZEN_PERFORMANCE 현재 runtime과 비교.

두 모델 ensemble이 너무 느려질 경우
shared calculation을 찾는다.

과학적 prediction을 바꾸지 않고
runtime만 줄이면 engineering promotion 가능.

================================================================
49. Shared computation 우선
================================================================

새 router/scale branch가
이미 계산 중인:

AR8
ARCH1
innovation
conditional variance
lag state

를 재계산하지 않게 한다.

Prepared historical state를 공유.

동일 계산 duplicated 금지.

================================================================
50. Runtime optimization parity
================================================================

최적화는 반드시:

original scientific prediction
vs optimized implementation

전체 DEV 또는 충분한 frozen replay에서
bitwise / strict tolerance parity 확인.

prediction이 바뀌면
runtime optimization으로 분류하지 말고
새 scientific candidate로 분류.

================================================================
51. Model serialization
================================================================

현재 canonical JSON+zlib 형식을 유지할 수 있으면 유지.

pickle/joblib byte alias 문제를 다시 만들지 마라.

새 model artifact:

hash
feature contract
tree structure
training ID hash
training target hash

기록.

================================================================
52. 프로젝트 유지보수
================================================================

새 연구 코드가 계속 쌓이면서
프로젝트가 깨지지 않게 관리한다.

매 major phase 후:

unit tests
targeted integration tests
git status
source manifest

확인.

dead experiment code를 바로 삭제하지 마라.

deprecated/failed로 표시.

과거 evidence를 지우지 않는다.

================================================================
53. Regression tests
================================================================

새 기능 추가 시:

기존 536 tests 전체가 너무 비싸다면
빠른 targeted suite를 자주 실행하고,

중간 major checkpoint마다
expanded suite.

최종 audit에서는
전체 test suite 실행.

기존 PASS를 깨뜨리면서
점수만 좋아지는 candidate는 champion 불가.

================================================================
54. Code quality
================================================================

새 기능을 하나의 5,000줄 script에 넣지 않는다.

권장 구조:

src/next3/
    bottlenecks.py
    router.py
    scale_repair.py
    forensic.py
    maintenance.py
    runtime.py

등 논리적으로 나눈다.

하지만 지나친 abstraction도 금지.

================================================================
55. 실험 실패 처리
================================================================

FAIL도 자산이다.

각 실패에:

왜 실패했는가
어떤 병목을 겨냥했는가
screen 결과
full 결과
synthetic 결과
real DEV 결과
다시 시도할 가치

기록.

같은 실패 아이디어를 이름만 바꿔 반복 금지.

================================================================
56. BUG와 NEGATIVE RESULT 구분
================================================================

코드 버그로 실패한 실험:

BUG

수정 후 같은 hypothesis 재실행 가능.

정상 구현인데 성능이 낮은 실험:

NEGATIVE_RESULT

로 보존.

둘을 섞지 마라.

================================================================
57. Catastrophic finding
================================================================

만약 연구 중 다음이 발견되면:

- leakage
- seal contamination
- scorer mismatch
- frozen prediction 재현 실패
- current champion implementation bug

즉시 performance 연구 중지.

현재 모델 QUARANTINE.

원인 해결부터 한다.

문제를 숨기고 다음 실험 진행 금지.

================================================================
58. Adaptive Research Loop
================================================================

기본 mandatory phase가 끝난 후
남은 대부분 시간은 다음 루프.

while active_research_time_remaining:

    1. BOTTLENECK_BOARD 갱신

    2. 기대가치가 가장 높은 병목 선택

    3. 기존 evidence 읽기

    4. 새로운 hypothesis 작성

    5. 중복 실험 여부 hash 검색

    6. candidate 구현

    7. unit tests

    8. cheap screen

    9. 실패면 기록 후 다음

    10. 통과면 full CV

    11. finalist면 bootstrap/alternate/regime

    12. mechanism trade-off 확인

    13. latency/state 측정

    14. promote/reject

    15. checkpoint

    16. 다시 bottleneck 평가

반복.

================================================================
59. Adaptive 방향 전환 규칙
================================================================

CASE A

Router가 DEPENDENCE gain을 유지하면서
SCALE loss를 줄이고 overall도 개선:

→ router 구조의 최소구성/robustness에 집중.


CASE B

Router가 계속 실패:

→ C205를 route하려 하지 말고
dependence specialist 자체를 scale-invariant하게 재설계.


CASE C

Scale negative evidence repair가 성공:

→ J102와 C205 양쪽에 적용 가능한지 확인.


CASE D

Scale repair는 synthetic만 좋아지고 real DEV 무효:

→ 해당 방향 중단.


CASE E

C205 subset이 동일 blend gain:

→ specialist pruning + runtime 집중.


CASE F

새 독립 complement 등장:

→ fixed finite blend만 비교.


CASE G

모든 성능 아이디어가 +0.001 미만:

→ DEV exhaustion 가능성 증가.
새 미세튜닝을 줄이고
robustness/reproducibility/deployment maintenance로 전환.


CASE H

+0.0025 이상 robust gain:

→ NEW_GENERATION_CANDIDATE로 동결.
seal은 여전히 열지 않음.


CASE I

+0.0035 이상이며 모든 robustness gate 통과:

→ 매우 강한 candidate.
SEAL_OPEN_RECOMMENDATION_NEXT3에 기록.
실제 seal open은 하지 않음.

================================================================
60. Research exhaustion 판단
================================================================

DEV를 무한 소비하면 안 된다.

다음 상태가 약 2시간 이상 지속되면:

- 10개 이상 독립 가설 모두 failed screen
- best full improvement < +0.0007
- 반복 아이디어만 남음

`DEV_RESEARCH_SATURATION = POSSIBLE`

로 표시.

그 뒤 남은 시간을:

- model simplification
- runtime
- reproducibility
- deployment
- counterexample characterization
- next independent validation planning

에 사용.

점수를 억지로 캐지 마라.

================================================================
61. 12시간 기본 시간 배분
================================================================

이 스케줄은 기본값이며
병목 관리자 판단에 따라 변경 가능.

0:00 ~ 0:40
상태 복원
preflight
seal guard
maintenance orchestrator

0:40 ~ 1:30
Bottleneck Board 작성
현재 J102/C205 trade-off forensic

1:30 ~ 3:30
dynamic routing / specialist selection 연구

3:30 ~ 5:30
negative scale evidence / ARCH-tail downstream 연구

5:30 ~ 7:00
C205 pruning / scale-invariant dependence / complement 개선

7:00 ~ 8:30
best candidates full CV / bootstrap / alternate / regime

8:30 ~ 10:00
adaptive highest-value bottleneck research

10:00 ~ 11:10
final scientific iteration
runtime/shared-state optimization
deployment maintenance

11:10 ~ 12:00
FINAL AUDIT
reproduction
documentation
handoff

고정 스케줄이 아니다.

실제 결과에 따라 우선순위를 바꿔라.

================================================================
62. Checkpoint
================================================================

최소 30~45분마다.

artifacts/next3/checkpoints/

NEXT3_CHECKPOINT_001.json
...

기록:

productive_seconds
wall_seconds
current_phase
current_bottleneck
active_hypothesis
current_best_performance
current_best_robust
current_best_distilled
experiments_completed
experiments_failed
tests_status
seal_status
resource_status
next_task
git_commit

================================================================
63. Resume 기능
================================================================

중간에 PC 재부팅이나 process crash가 있어도
완료된 연구를 잃지 않게:

scripts/resume_next3.py

또는 equivalent.

마지막 valid checkpoint를 읽고:

completed experiment는 skip.

incomplete experiment만
안전하게 재개 또는 restart.

================================================================
64. 로그 크기 관리
================================================================

장기런에서 stdout 수 GB 쌓이지 않게 한다.

debug spam 금지.

per-experiment log.

큰 DataFrame 전체 출력 금지.

summary + artifact path만 console.

================================================================
65. 기대 결과 역할
================================================================

최종적으로 최대 다음 4개 역할만 유지.

FROZEN_DISTILLED
기존 안전 fallback.

FROZEN_PERFORMANCE
현재 0.62327 reference.

NEXT3_PERFORMANCE
이번에 확실히 더 좋은 경우.

NEXT3_ROBUST
성능은 비슷하지만 mechanism robustness가 더 좋은 경우.

winner를 10개 남기지 마라.

================================================================
66. Seal recommendation
================================================================

이번에도 실제 seal 개봉 금지.

대신 종료 시:

SEAL_OPEN_RECOMMENDATION_NEXT3.md

작성.

판단:

NO
MAYBE
YES

근거:

performance delta
bootstrap
alternate splits
regime
mechanism
DEV adaptive risk
remaining unresolved bottlenecks

사용자가 다음 세션에서 결정한다.

================================================================
67. Public submission recommendation
================================================================

마찬가지로 실제 제출은 하지 않는다.

PUBLIC_SUBMISSION_RECOMMENDATION_NEXT3.md

작성.

현재 model generation이
public sanity check를 받을 가치가 있는지 판단만 한다.

================================================================
68. Final audit 시작 조건
================================================================

총 wall-clock 약 11시간 10분이 지나면
새 대형 실험 시작 중단.

현재 child experiment를 안전하게 마무리.

FINAL AUDIT MODE 전환.

================================================================
69. Final audit
================================================================

최종 약 50분 동안:

1. all seal guard
2. raw SHA
3. scorer parity
4. targeted full tests
5. 가능하면 complete full test suite
6. current selected models deterministic replay
7. save/load parity
8. offline/live parity
9. canonical serialization parity
10. inference current-point-only
11. future length inaccessible
12. DEV/seal overlap zero
13. reduced new usage zero
14. public submission zero
15. runtime benchmark
16. model artifact hashes
17. source manifest
18. handoff package verification

실행.

================================================================
70. Independent reproduction
================================================================

최종 selected NEXT3 model이 존재하면
가능한 범위에서 fresh pinned environment / clean output directory에서:

raw DEV
→ feature generation
→ training
→ inference

재현.

기존 feature cache를 사용하지 않는 reproduction 하나 이상.

너무 오래 걸려 12시간을 넘길 경우
12시간 종료 전에 최소 essential reproduction을 끝내고
나머지는 handoff에 pending으로 정확히 기록.

================================================================
71. 최종 산출물
================================================================

최소 다음 생성.

00_GPT_READ_FIRST_NEXT3.md
01_GPT_HANDOFF_NEXT3.md

NEXT3_PREFLIGHT.json

BOTTLENECK_BOARD_INITIAL.json
BOTTLENECK_BOARD_FINAL.json
BOTTLENECK_HISTORY.csv

MAINTENANCE_LOG.json
RESOURCE_LOG.csv
HANG_RECOVERY_LOG.csv

ROUTER_PROTOCOL.md
ROUTER_RESULTS.csv

SCALE_DOWN_FORENSIC.md
SCALE_REPAIR_RESULTS.csv

DEPENDENCE_FORENSIC.md
C205_ABLATION_RESULTS.csv
DEPENDENCE_SPECIALIST_RESULTS.csv

ORTHOGONAL_COMPLEMENT_RESULTS.csv

MODEL_RESULTS_NEXT3.csv
CV_RESULTS_NEXT3.csv
PAIRED_BOOTSTRAP_NEXT3.csv
ALT_SPLIT_RESULTS_NEXT3.csv
REGIME_RESULTS_NEXT3.csv
MECHANISM_RESULTS_NEXT3.csv

COUNTEREXAMPLE_RESULTS_NEXT3.csv
LONG_NULL_RESULTS_NEXT3.csv

LATENCY_NEXT3.json
STATE_MEMORY_NEXT3.json

DEV_CANDIDATES_NEXT3.json

SEAL_GUARD_NEXT3.json
SEAL_OPEN_RECOMMENDATION_NEXT3.md
PUBLIC_SUBMISSION_RECOMMENDATION_NEXT3.md

REPRODUCTION_NEXT3.json
DETERMINISM_NEXT3.json
FINAL_AUDIT_NEXT3.json

LIMITATIONS_NEXT3.md
NEXT_SESSION_PLAN_NEXT3.md

EXPERIMENT_LEDGER.csv

SOURCE_MANIFEST_NEXT3.json

그리고:

STRUCTURAL_BREAK_GPT_HANDOFF_NEXT3_<timestamp>.zip

생성.

raw competition data 넣지 마라.
real point OOF 전체 넣지 마라.
credentials/token 넣지 마라.
seal label 넣지 마라.

================================================================
72. 00_GPT_READ_FIRST_NEXT3.md
================================================================

한국어로 사용자가 바로 이해할 수 있게 작성.

반드시 다음 질문에 답한다.

1. 총 wall-clock
2. 실제 productive research time
3. 실험 수
4. 장애/재시작 수
5. 시작 performance score
6. 종료 best performance score
7. delta
8. generation gate 통과 여부
9. 새 performance candidate
10. 새 robustness candidate
11. distilled fallback
12. 현재 가장 큰 병목
13. 처음 예상한 병목과 실제 병목이 달랐는가
14. DEPENDENCE gain
15. SCALE loss
16. router가 trade-off를 개선했는가
17. scale-down downstream 원인은 무엇이었나
18. ARCH/tail 감소 문제를 개선했는가
19. C205에서 실제 필요한 component
20. 새로운 complement 발견 여부
21. 5-fold scores
22. median
23. worst fold
24. bootstrap CI
25. P(delta>0)
26. alternate split 3개
27. regime worst
28. long-null 상태
29. outlier persistence
30. runtime
31. state memory
32. 10M point 예상/실측
33. tests
34. reproducibility
35. SEAL_A opened?
36. SEAL_B opened?
37. SEAL_C opened?
38. SEAL_D opened?
39. reduced 신규 사용?
40. public submission?
41. 지금 seal을 열어야 하는가?
42. public submission을 할 가치가 있는가?
43. DEV research saturation 위험
44. 다음 단계 3개

================================================================
73. Bottleneck final report
================================================================

최종 BOTTLENECK_BOARD_FINAL에는
각 병목을:

RESOLVED
PARTIALLY_RESOLVED
UNCHANGED
WORSE
DEFERRED
NOT_RELEVANT

중 하나로 분류.

그리고 다음 세션의
top 3 bottleneck만 남긴다.

================================================================
74. 실패한 실험도 ZIP에 요약
================================================================

실패 candidate 전체 model binary를
전부 넣을 필요는 없다.

하지만:

hypothesis
config hash
score
failure reason

이 포함된 ledger는 반드시 포함.

그래야 다음 Codex가
같은 실험을 반복하지 않는다.

================================================================
75. 보호할 핵심 프로젝트 원칙
================================================================

절대 하지 말 것:

seal open

reduced 재평가

public submission

future suffix

online total future length

tau feature

dataset_id feature

post-hoc mechanism label routing

row-random CV

new CV seed shopping

candidate마다 split 변경

static blend 0.01 단위 brute-force

massive hyperparameter sweep

Transformer/LSTM/GRU 대규모 연구

synthetic score를 competition score처럼 선택

실패 gate 사후 완화

기존 canonical model overwrite

raw data 수정

실패 기록 삭제

================================================================
76. 중요한 사고방식
================================================================

이번 세션의 목적은:

"12시간 동안 최대한 많은 실험을 한다"

가 아니다.

목적은:

"현재 목표 달성을 가장 크게 막고 있는 병목을
계속 찾아내고,
가장 가치 있는 병목부터 해결하며,
프로젝트를 망가지지 않게 유지하면서
성능과 신뢰도를 앞으로 이동시킨다."

이다.

한 병목이 해결되면
즉시 다음 병목을 찾아라.

병목이 잘못 진단되었다면
그 사실을 인정하고 priority를 바꿔라.

================================================================
77. 결과가 안 좋아도 유지보수 계속
================================================================

새 score improvement가 몇 시간 동안 없더라도
run 종료 금지.

그 경우:

- current model forensic
- test hardening
- runtime optimization
- code simplification
- cache integrity
- reproducibility
- failure deduplication
- documentation
- deployment preflight
- generalization stress

등 프로젝트 전체 가치가 증가하는 작업을 계속한다.

================================================================
78. 현재 성능을 절대 잃지 않는다
================================================================

어떤 NEXT3 실험도
FROZEN_PERFORMANCE와 FROZEN_DISTILLED를 파괴하지 않는다.

새 candidate 실패 시
기존 모델로 즉시 rollback 가능해야 한다.

실험 branch와 canonical branch 분리.

================================================================
79. 최종 상태 라벨
================================================================

종료 시 하나를 선언.

NEXT3_GENERATION_READY

조건:
진짜 generation gate 통과.


NEXT3_PERFORMANCE_IMPROVED

조건:
performance gate만 통과.


NEXT3_ROBUSTNESS_IMPROVED

조건:
점수는 비슷하나 명확한 robustness 개선.


NEXT3_ENGINEERING_IMPROVED

조건:
모델 품질 동일 + runtime/state/reproducibility 개선.


NEXT3_NO_PROMOTION

조건:
새 candidate가 기존을 이기지 못함.

NO_PROMOTION도 정상적인 연구 결과다.

================================================================
80. 지금 즉시 실행
================================================================

계획을 장황하게 출력하고 사용자 승인을 기다리지 마라.

즉시:

1. 최신 NEXT2 상태 읽기
2. frozen hash 확인
3. seal/raw/scorer preflight
4. NEXT3 orchestrator 실행
5. Bottleneck Board 작성
6. 가장 중요한 병목 선택
7. hypothesis 등록
8. 코드 작성
9. test
10. screen
11. full validation
12. 결과에 따라 병목 재평가
13. 11시간 이상 반복
14. 마지막 약 50분 final audit
15. GPT handoff ZIP

순으로 진행한다.

사용자는 약 12시간 동안 자리를 비운다.

중간에 질문하지 말고
합리적인 범위에서 스스로 판단해 계속 진행한다.

단 독립 검증 자산인
SEAL_A/B/C/D는 어떤 상황에서도 열지 않는다.
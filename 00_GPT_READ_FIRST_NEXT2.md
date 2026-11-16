# Structural Break NEXT2 — GPT가 먼저 읽을 문서

기존 프로젝트의 연구를 이어 수행했다. 성능 후보는 **J102/C205 고정 80/20**, 간소화 후보는 **예측이 동일한 J102 tree-used**다. 새 최고 DEV 평균은 **0.6232684128173095**, 이전 PRIMARY 대비 **0.0016366625048295225**다. 성능 후보 기준은 통과했지만 **generation-level 교체 기준은 미달**이다. 네 seal은 모두 미개봉이며, reduced/public 신규 사용은 0이다.

1. **실제 순수 연구 시간:** 21607.243523초 = 6시간 7.243523초. 연구 종료 후 별도 감사 시간을 추가했다. 최종 감사 시간은 `FINAL_AUDIT_NEXT2.json`과 ZIP 외부 검증 기록을 기준으로 한다. 대기나 오프라인 공백을 연구 시간으로 추가하지 않았다.
2. **실험 수:** 새 이름 71개, 기존 정확 재사용 9개, 과학 모델 기록 총 80개. 연구 중 실제 learner fit 394회, 별도 원본 재학습 검증 3회로 총 397회다. 고정 혼합·동일성 재생·H 초기화·합성 진단은 learner fit으로 세지 않았다. 원래 10개와 보조 2개를 포함해 runtime 12개, component 비용/반례 모델 28개를 측정했다.
3. **이전 PRIMARY exact score:** `J102_SERIAL_VARIANCE_512`, 5-fold mean **0.62163175031248**. Y001 **0.621631469960325**, B104 **0.6214890347069302**도 원본 모델/설정을 보존했다.
4. **새 best exact score:** `BL2_C205_DEPENDENCE_80`, mean **0.6232684128173095**, median **0.6216260123520351**. 구현은 `shared_blend_v1`, entry는 `next2_shared_performance_main.py`다.
5. **delta:** 이전 J102 대비 **0.0016366625048295225**. 이 수치는 동일한 원래 5-fold DEV 평가의 차이다.
6. **새 distilled candidate:** `J102_SERIAL_VARIANCE_512_RUNTIME_TREE_USED_V1`; entry `next2_canonical_distilled_main.py`. 학습 모델과 예측을 그대로 두고 실제 초기화/배열/사용하지 않는 tree 입력 계산을 줄였다.
7. **distilled score loss/gain:** mean **0.62163175031248**, Δ **0**. 전체 DEV, 원래 OOF, alternate/H holdout에서 예측이 정확히 같다.
8. **feature 수 감소율:** distilled의 최종 all-DEV tree 입력은 **77→48, 37.662338% 감소**. 과학적 feature 계약은 77개 그대로다. 내부 새 variance 통계 26개를 전부 23개로 줄였다는 뜻은 아니다. 기본 ABCD EWMA cell은 실제 **51→31**로 줄었다. 공유 성능 모델은 총 101개 과학적 입력 중 72개를 tree 입력으로 만든다.
9. **runtime 감소율:** distilled H 초기화 median **7848.30→4639.25 µs, 40.8885% 감소**. point median **5.1→5.1 µs**로 동일, 직접 10M은 **51.6476629→50.7685899초**, 약 **1.702% 감소**다. 성능 모델의 공유 구현은 기존 80/20 compact 대비 **104.597764→93.1921636초**, **10.9042% 감소**했다.
10. **state memory 감소율:** distilled private 수치 배열 **29,173→11,729 bytes, 59.795016% 감소**. 공유 성능 모델 private state **16,441 bytes**, prepared model/shared 수치 배열 **193,472 bytes**다. Python 객체·JIT·allocator·일시적 H peak는 이 수치에 포함하지 않는다. 직접 10M 후 state 증가 0 bytes다.
11. **GLR 실제 필요 여부:** explicit GLR 제거 D202 평균 **0.6212321921035091**, Δ **-0.0003995582089709**, 비열등성 bootstrap 비율 **0.551**로 기준 0.90 미달이다. B104에서는 GLR/GLR-memory 배열과 계산을 실제 제거하고 전체 예측 동일성을 입증했지만, B104 자체를 J102 대신 채택할 통계적 근거는 부족하다. GLR이 꼭 필요하다고 확정할 수도, 제거해도 된다고 확정할 수도 없다.
12. **serial correction 실제 필요 여부:** J102−Y001 평균 차이는 **+0.000000280352155**로 독립적 성능 기여가 입증되지 않았다. 그러나 제거 후보의 median 하락과 비열등성 비율 **0.78** 때문에 현재 간소화 기준은 실패했다.
13. **Bayes-lite 필요 여부:** matched Y001 대비 Bayes 제거 B102 평균 차이 **-0.002774635454164**, conditional paired CI는 약 **[-0.0047041, -0.0008476]**다. 이 recipe에서 유효한 구성이라는 근거가 있다. 일반적인 posterior 확률이나 anytime 보장은 아니다.
14. **long-age 중 핵심 age:** 최대 128인 V007보다 256/512를 함께 추가한 Y001이 평균 **+0.002334211970588** 높다. short/medium/long/sparse ablation으로 256과 512 각각의 독립 기여를 확정하지 못했다. 둘 중 하나만 핵심이라고 말하지 않는다.
15. **scale-down 개선 여부:** 최종 성능 모델의 개선은 아니다. 원래 ARCH variance×0.7에서 J102 **0.473043**, blend **0.462974**; ×0.5에서 **0.588660→0.571729**로 악화했다. 별도 교차 배경 실험에서는 Gaussian/AR 감소는 잘 감지하고 ARCH/tail 감소에서 약해졌다. B104의 개선은 real DEV 비열등성 기준을 통과하지 못했다.
16. **heavy-tail 개선 여부:** real DEV 성능 손실 없이 일관되게 개선한 후보를 확보하지 못했다. 고정 winsor/Huber/log/absolute/robust/ECDF 변형과 t-tail/이상치 반례를 모두 남겼다. 최종 blend의 단일 이상치 전체 false-alarm excess는 **0.744141**로 J102 **0.693359**보다 크다.
17. **ARCH counterexample 개선 여부:** ARCH2/EWMA 등 일부는 합성 감소 반례에서 좋아졌지만 real DEV/holdout 조건에서 탈락했다. 전체 pipeline ARCH 제거 D218 mean **0.6178936567835125**, J102 대비 **-0.0037380935289674**이며 CI가 0 아래다. 새 branch만 ARCH를 제거한 D208과 구분했다.
18. **long-null behavior:** Gaussian 5% 고정 1,024-horizon threshold에서 J102 eventual FPR은 **0.037109→0.128906→0.322266**(1,024/4,096/16,384). blend의 16,384 FPR은 **0.330078**다. odds age-decay는 rank/TS-AUC를 보존하지만 49,152 늦은 변화에서 clean detection을 0으로 만들었다. raw J102 clean detection **0.40625**, blend **0.3203125**와 맞바꾸는 결과이므로 age-decay를 채택하지 않았다. 무한-horizon false-alarm 보장은 없다.
19. **strongest complementary detector:** 실제 성능 혼합에 가장 유용한 것은 `C205_DEPENDENCE`, 단독 DEV mean **0.5755159686249883**. 단독 점수 자체의 최고는 CDF16 **0.585069679508775**지만 혼합 기여는 C205가 가장 컸다.
20. **primary와 complement correlation:** C205와 J102의 admission within-time Spearman **0.42193977197036003**. 최종 blend와 J102는 ordinary Spearman 약 **0.990907**, within-time **0.988041**, Pearson **0.991600**이다. 서로 다른 correlation 정의를 섞지 않는다.
21. **ensemble gain:** 고정 **0.8×J102 + 0.2×C205**가 **0.0016366625048295225**. 90/10, 80/20, 67/33, 50/50만 사용했고 stacking/meta learner는 없다. 공유 연산은 두 100-tree ensemble과 float32 component rounding을 유지한다.
22. **새 best 5-fold scores:** `[0.628814140424981, 0.6371811063970156, 0.6216260123520351, 0.618712391914872, 0.6100084129976436]`.
23. **worst fold:** **0.6100084129976436**. J102 worst **0.6101464788170321** 대비 **-0.00013806581938846652**다.
24. **paired bootstrap:** 2,000 whole-series draws; Δ CI **[0.0006743151264927167, 0.0025479387096717654]**, Δ>0 비율 **0.999**. 고정 OOF 조건부 불확실성이며 model fitting, 적응적 선택, 다중 비교를 보정하지 않는다. posterior 성공 확률이나 외부 일반화 보장으로 해석하지 않는다. 24개 scope를 저장된 fold별 score draw에서 재계산했다.
25. **alternate splits:** 평균 **[0.622572491375891, 0.6211655235665667, 0.6234311259085243]**, J102 대비 Δ **[+0.0016474115222865, +0.0011973940055212, +0.0017674721771380]**. 세 개 모두 양수다.
26. **regime worst:** leave-one-H-regime-out 최저 fold **0.6006542695774002**, 평균 **0.6356987563234016**. H matched 최소 Δ **-0.0012063679528282**로 붕괴 기준을 넘지 않았다. 원래 OOF의 H slice 최저는 별도 지표이며 약 **0.610047**이다.
27. **mechanism weak point:** supported DEPENDENCE slice Δ **+0.004261**(CI 약 **[+0.002878,+0.005666]**)와 SCALE slice Δ **-0.004355**(CI 약 **[-0.007285,-0.001237]**)의 교환이다. 낮은 절대 점수는 WEAK/AMBIGUOUS 약 **0.585671**이다. MEAN 21, TAIL 86 breaks는 지원 표본 기준 미달이다. 기존 SCALE label도 decrease 79/increase 53으로 나누면 각각 100-break 기준에 못 미친다.
28. **SEAL_A opened?** false, **500개 미개봉**.
29. **SEAL_B opened?** false, **500개 미개봉**.
30. **SEAL_C opened?** false, **500개 미개봉**.
31. **SEAL_D opened?** false, **500개 미개봉**.
32. **reduced new usage:** **0**. 기존 quota 사용 2/2 상태와 ledger bytes를 보존했다.
33. **public submission new usage:** **0**. 제출/SDK test를 실행하지 않았다.
34. **지금 seal을 열 필요가 있는가?** **아니오.** 세대 교체 기준 미달이고 SCALE·이상치·long-null 약점이 남았다. 다음 가설과 기준을 먼저 고정한다.
35. **다음 세션 최우선 3가지:** (1) ARCH/tail scale-down에서 signed current 정보가 downstream evidence/memory/tree에 어떻게 반영되는지 분리하고, real DEV 손실 없는 수정을 사전 등록한다. (2) 누적 false alarm을 줄이면서 늦은 변화의 clean detection을 유지하는 memory/reset 정책을 공동 검증한다. (3) DEV 적응적 선택 불확실성을 인정하고, 독립된 다음 검증 계획과 generation 기준을 먼저 고정한다. 네 seal은 계속 보존한다.

빠른 검토: `01_GPT_HANDOFF_NEXT2.md` → `CURRENT_GENERATION_COMPONENT_GRAPH.md` → 두 최종 모델/성능 표 → `FINAL_AUDIT_NEXT2.json`. 그래프는 `artifacts/next2/figures/`의 PNG/PDF 9쌍이다. 원본 competition 데이터와 개별 real OOF 예측은 ZIP에서 제외했다.

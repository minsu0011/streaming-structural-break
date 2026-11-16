"""Generate the Korean review handoff only after scientific and reproduction evidence passes."""
from pathlib import Path
import sys
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.session import clock
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    timing = clock(); assert timing['minimum_research_met'] and timing['audit_started']
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT2.json').read_text()); assert selection['status'] == 'FROZEN_NEXT2_DEV_SELECTION'
    reproduction = json.loads((ROOT/'REPRODUCTION_NEXT2.json').read_text()); assert reproduction['status'] == 'PASS'
    performance = selection['performance']; distilled = selection['distilled']; best = performance['full']
    previous = selection['previous_primary']['full']; delta = best['mean']-previous['mean']
    runtime = pd.read_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv', float_precision='round_trip').set_index('runtime_variant')
    j0, j, b0, b = (runtime.loc[name] for name in ('J_ORIGINAL', 'J_USED', 'BL80_COMPACT', 'BL80_SHARED_V1'))
    pair = performance['paired_all_time']; research_seconds = timing['elapsed_research_seconds']
    read_first = f'''# Structural Break NEXT2 — GPT가 먼저 읽을 문서

기존 프로젝트의 연구를 이어 수행했다. 성능 후보는 **J102/C205 고정 80/20**, 간소화 후보는 **예측이 동일한 J102 tree-used**다. 새 최고 DEV 평균은 **{best['mean']!r}**, 이전 PRIMARY 대비 **{delta!r}**다. 성능 후보 기준은 통과했지만 **generation-level 교체 기준은 미달**이다. 네 seal은 모두 미개봉이며, reduced/public 신규 사용은 0이다.

1. **실제 순수 연구 시간:** {research_seconds:.6f}초 = 6시간 7.243523초. 연구 종료 후 별도 감사 시간을 추가했다. 최종 감사 시간은 `FINAL_AUDIT_NEXT2.json`과 ZIP 외부 검증 기록을 기준으로 한다. 대기나 오프라인 공백을 연구 시간으로 추가하지 않았다.
2. **실험 수:** 새 이름 71개, 기존 정확 재사용 9개, 과학 모델 기록 총 80개. 연구 중 실제 learner fit 394회, 별도 원본 재학습 검증 3회로 총 397회다. 고정 혼합·동일성 재생·H 초기화·합성 진단은 learner fit으로 세지 않았다. 원래 10개와 보조 2개를 포함해 runtime 12개, component 비용/반례 모델 28개를 측정했다.
3. **이전 PRIMARY exact score:** `J102_SERIAL_VARIANCE_512`, 5-fold mean **{previous['mean']!r}**. Y001 **0.621631469960325**, B104 **0.6214890347069302**도 원본 모델/설정을 보존했다.
4. **새 best exact score:** `BL2_C205_DEPENDENCE_80`, mean **{best['mean']!r}**, median **{best['median']!r}**. 구현은 `shared_blend_v1`, entry는 `next2_shared_performance_main.py`다.
5. **delta:** 이전 J102 대비 **{delta!r}**. 이 수치는 동일한 원래 5-fold DEV 평가의 차이다.
6. **새 distilled candidate:** `J102_SERIAL_VARIANCE_512_RUNTIME_TREE_USED_V1`; entry `next2_canonical_distilled_main.py`. 학습 모델과 예측을 그대로 두고 실제 초기화/배열/사용하지 않는 tree 입력 계산을 줄였다.
7. **distilled score loss/gain:** mean **0.62163175031248**, Δ **0**. 전체 DEV, 원래 OOF, alternate/H holdout에서 예측이 정확히 같다.
8. **feature 수 감소율:** distilled의 최종 all-DEV tree 입력은 **77→48, {(1-48/77)*100:.6f}% 감소**. 과학적 feature 계약은 77개 그대로다. 내부 새 variance 통계 26개를 전부 23개로 줄였다는 뜻은 아니다. 기본 ABCD EWMA cell은 실제 **51→31**로 줄었다. 공유 성능 모델은 총 101개 과학적 입력 중 72개를 tree 입력으로 만든다.
9. **runtime 감소율:** distilled H 초기화 median **{j0.historical_init_median_us:.2f}→{j.historical_init_median_us:.2f} µs, {(1-j.historical_init_median_us/j0.historical_init_median_us)*100:.4f}% 감소**. point median **5.1→5.1 µs**로 동일, 직접 10M은 **51.6476629→50.7685899초**, 약 **1.702% 감소**다. 성능 모델의 공유 구현은 기존 80/20 compact 대비 **104.597764→93.1921636초**, **{(1-b.direct_10M_seconds/b0.direct_10M_seconds)*100:.4f}% 감소**했다.
10. **state memory 감소율:** distilled private 수치 배열 **29,173→11,729 bytes, {(1-11729/29173)*100:.6f}% 감소**. 공유 성능 모델 private state **16,441 bytes**, prepared model/shared 수치 배열 **193,472 bytes**다. Python 객체·JIT·allocator·일시적 H peak는 이 수치에 포함하지 않는다. 직접 10M 후 state 증가 0 bytes다.
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
21. **ensemble gain:** 고정 **0.8×J102 + 0.2×C205**가 **{delta!r}**. 90/10, 80/20, 67/33, 50/50만 사용했고 stacking/meta learner는 없다. 공유 연산은 두 100-tree ensemble과 float32 component rounding을 유지한다.
22. **새 best 5-fold scores:** `{best['fold_scores']}`.
23. **worst fold:** **{best['worst_fold']!r}**. J102 worst **{previous['worst_fold']!r}** 대비 **{best['worst_fold']-previous['worst_fold']!r}**다.
24. **paired bootstrap:** 2,000 whole-series draws; Δ CI **[{pair['ci_2_5']!r}, {pair['ci_97_5']!r}]**, Δ>0 비율 **{pair['probability_delta_positive']!r}**. 고정 OOF 조건부 불확실성이며 model fitting, 적응적 선택, 다중 비교를 보정하지 않는다. posterior 성공 확률이나 외부 일반화 보장으로 해석하지 않는다. 24개 scope를 저장된 fold별 score draw에서 재계산했다.
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
'''
    (ROOT/'00_GPT_READ_FIRST_NEXT2.md').write_text(read_first, encoding='utf-8', newline='\n')
    handoff = '''# Structural Break NEXT2 연구 인계

이전 프로젝트를 이어 진행한 DEV 전용 연구다. 최종 역할은 `DEV_CANDIDATES_NEXT2.json`에 동결했다. 기존 formal PRIMARY J102의 원본 파일을 교체하지 않았으며, 새 80/20 모델은 성능 후보 역할로 제공한다. generation-level 교체는 승인되지 않았다.

| 역할 | 과학 모델 | 실행 구현 | mean DEV TS-AUC | canonical 파일 |
| --- | --- | --- | ---: | --- |
| DISTILLED | J102_SERIAL_VARIANCE_512 | tree_used_v1 | 0.62163175031248 | artifacts/next2/canonical_deployment/DISTILLED_tree_used_v1/model.sbnext2 |
| PERFORMANCE | BL2_C205_DEPENDENCE_80 | shared_blend_v1 | 0.6232684128173095 | artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2 |

공통 학습 규칙은 고정 100-tree LightGBM, W3, 기존 group 5-fold다. 성능 모델은 100-tree 두 개의 80/20 산술 혼합이다. synthetic 점수로 weight/threshold를 고르지 않았다. 실제 두 entry의 train/infer를 새 환경에서 원본 DEV로 재실행했으며, model bytes·학습 feature/ID/target hash·전체 4,014,405개 예측 fingerprint를 맞췄다. 학습 시 모든 seal ID는 값 접근을 거부하는 opaque object로 주었고 2,000개를 건너뛰었다. 재학습의 all-DEV 예측은 동일성 확인이며 in-sample 품질 점수가 아니다.

`model.sbnext2`는 JSON+zlib의 검증 가능한 inference 전용 형식이다. 기존 joblib은 pickle alias metadata 때문에 byte 동일성이 실패한 기록이 있어, predictive arrays/Booster 내용이 같은 것을 먼저 확인한 뒤 canonical 형식으로 해결했다. 기존 실패 결과를 삭제하지 않았다. 모델 파일 크기는 distilled 35,180 bytes, performance 68,999 bytes다. 프로세스/JIT 메모리 크기와 혼동하지 않는다.

연구 결론은 세 가지다. 첫째, 기존 0.6216대 상승에는 ARCH/innovation energy/forecast와 long-age Bayes가 기여하며 serial correction 자체의 이득은 거의 0이다. 다만 GLR/serial 제거 후보는 사전 비열등성 확률 조건을 통과하지 못했다. 둘째, 학습 예측을 바꾸지 않고 J102 private state를 약 59.80% 줄이고 H 초기화를 약 40.89% 단축했다. 셋째, 낮은 상관의 조건부 dependence 신호 C205가 실제 성능 혼합에 기여했다. 성공 조건 B와 D를 충족했지만 전반적 robustness 개선이나 generation jump는 아니다.

합성 연구는 18개 원래 시나리오와 4개 long-null 배경을 고정했다. 원래 all-DEV J/Y/B/C/blend, 원래 fold-0의 component 28개, 기존 5개 CV fit의 민감도, 교차 배경 scale, 늦은 변화/첫 경보, 단일 이상치 잔류, 내부 증거와 predictive TreeSHAP을 구분했다. 표의 `fitted_fold`, `artifact_kind`, `record_type`, `diagnostic_suite`를 함께 읽는다. H heldout을 다시 학습한 점수와 원래 OOF를 H별로 자른 점수도 구별한다. runtime 비용의 28개 component 진단은 CV0 모델이며, 최종 all-DEV runtime은 별도의 12개 측정 표를 기준으로 한다.

ARCH/tail scale-down 실패를 단순한 신호 부재로 설명하지 않는다. H calibration에서 extreme energy가 표준편차에 큰 영향을 주고, 음의 current-energy 범위가 좁아졌다. 그러나 paired current-energy와 forecast는 모두 감소에 맞는 signed 반응을 유지했다. 뒤의 evidence/memory/트리 결합이 다음 연구 대상이다. limiting ARCH innovation의 fourth moment 계산은 고정 DGP에 대한 수학적 성질이며, 유한 simulation sample이나 real competition DGP의 무한 모멘트를 주장하지 않는다. `docs/NEXT2_SCALE_DOWN_MECHANISTIC_SYNTHESIS.md`에 근거와 한계를 연결했다.

rank를 보존하는 odds age-decay는 TS-AUC를 바꾸지 않지만, 49k의 늦은 변화에서 clean detection을 없앨 수 있었다. 높은 TS-AUC, finite-horizon FPR, 유용한 first alarm은 다른 평가다. 누적 GLR memory는 null에서 자라며, 이상치 하나의 영향은 수백 포인트 남는다. 제거 ablation은 일부 반례를 완화하지만 real DEV 조건을 충족하지 못해 최종 모델에 반영하지 않았다.

## 재현 명령

Windows PowerShell에서 압축 해제한 디렉터리로 이동한 뒤:

```powershell
py -3.13 -m venv .review-venv
.\\.review-venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.review-venv\\Scripts\\python.exe -m pip check
.\\.review-venv\\Scripts\\python.exe scripts/replay_next2_portable.py --output portable_replay.json
.\\.review-venv\\Scripts\\python.exe scripts/replay_next2_report_bootstrap.py
```

첫 replay는 raw competition 데이터 없이 18개 원래 synthetic scenario의 64개 role/control cell, 524,288개 public-API 출력을 기존 결과와 비교한다. 두 번째는 최고 모델의 24개 scope paired bootstrap 수치를 재계산한다. 압축본에는 필요한 두 모델, source/default config, 고정 generator, 작은 bootstrap 통계 배열이 포함된다. 원래 프로젝트 전체 테스트 536개 중 raw/old cache에 의존하는 검증은 원본 workspace가 필요하므로, ZIP만으로 모든 과거 테스트가 실행된다고 주장하지 않는다.

그림을 다시 만들려면 별도 figure 환경에 `requirements-next2-figures-lock.txt`의 13개 고정 패키지를 설치한다. 실제 그림 환경의 버전 목록과 일치하며, main 학습 환경의 63개 lock을 바꾸지 않는다. `scripts/plot_next2_research.py`, `scripts/plot_next2_internal_evidence.py`, `scripts/plot_next2_outlier_persistence.py`, `scripts/plot_next2_scale_channels.py`가 포함된 요약 CSV에서 PNG/PDF를 생성한다.

원본 재학습 검증은 `scripts/prepare_next2_independent_reproduction.py`, `scripts/complete_next2_reproduction_dependencies.py`, `scripts/run_next2_independent_role.py` 및 독립 child script에 기록했다. 기존 결과를 덮어쓰지 않도록 실행을 거부하는 시간/경로 guard가 있다. 그대로 재실행하려면 새 audit session/새 출력 경로가 필요하다. 원본 raw 파일의 필요 SHA는 `RAW_DATA_REQUIRED_NEXT2.json`, 실제 새 환경·module 경로·fit·예측 확인은 `REPRODUCTION_NEXT2.json`과 `artifacts/next2/independent_reproduction/`에 있다. 학습 iterator API로 재현할 때도 frozen DEV boundary를 유지해야 한다.

## 읽을 파일

`00_GPT_READ_FIRST_NEXT2.md`는 요청한 35개 답이다. `MODEL_RESULTS_NEXT2.csv`/`CV_RESULTS_NEXT2.csv`/`PAIRED_BOOTSTRAP_NEXT2.csv`는 전체 과학 결과, `RUNTIME_PRUNING_RESULTS.csv`/`LATENCY_NEXT2.json`/`STATE_MEMORY_NEXT2.json`은 실행 비용, 나머지 component/robustness CSV는 세부 근거다. `docs/NEXT2_*PROTOCOL.md`와 각 `POLICY.json`은 시점·메뉴·seed를 보존한다. 원래 experiment receipt의 `*_PENDING`은 당시 학습 단계 상태일 수 있으므로 최종 역할은 `DEV_CANDIDATES_NEXT2.json`과 최종 표의 disposition을 기준으로 읽는다.

`FINAL_AUDIT_NEXT2.json`은 별도 감사 단계의 결과, `SOURCE_MANIFEST_NEXT2.json`은 ZIP entry별 SHA/크기다. 모든 ZIP entry SHA와 CRC를 다시 읽어 확인한 외부 기록은 원본 프로젝트 `artifacts/next2/handoff/DESKTOP_ZIP_VERIFICATION_NEXT2.json`에 남긴다. ZIP 자기 hash를 ZIP 내부에 재귀적으로 넣지 않는다. 같은 agent가 연구 후 검증 단계를 분리했으며 독립된 인간/다른 agent의 심사를 받았다고 주장하지 않는다.
'''
    (ROOT/'01_GPT_HANDOFF_NEXT2.md').write_text(handoff, encoding='utf-8', newline='\n')
    limitations = '''# NEXT2의 한계와 보존한 실패

- DEV 적응적 탐색 71개 새 이름과 이전 세션의 선택이 누적됐다. bootstrap은 고정 fitted OOF에 조건부이며 fitting/selection/multiplicity를 보정하지 않는다. 다른 seed나 external data의 성능을 보장하지 않는다.
- 최고 평균 상승 +0.0016366625는 performance gate를 통과하지만 generation 기준 +0.0025/+0.0035에 미달한다. DEPENDENCE 이득과 SCALE 손실, 이상치 false alarm 악화가 공존한다. 모든 약점을 보완한 모델은 아니다.
- Gaussian/AR scale-down은 강하나 ARCH/tail 배경에서 약하다. B104의 반례 개선은 real DEV 비열등성 미달이다. mean/tail 및 SCALE 방향별 작은 DEV subset은 충분한 support가 없다.
- raw/conditional scale likelihood, winsor/Huber/log/abs/robust/ECDF, ARCH2/EWMA, memory/age 대안은 정해진 finite menu에서 검증했다. failed screen을 이유 없이 full CV로 올리지 않았다. 이 메뉴 밖 모든 방법이 실패한다고 일반화하지 않는다.
- 순간 channel 정보, 누적 evidence, tree 확률과 first alarm utility는 서로 다르다. TreeSHAP은 예측 기여 설명이며 feature의 독립적 인과 효과가 아니다. family 제거 후 재학습은 제한된 구성 수준의 대조다.
- 고정 threshold는 길이가 늘면 FPR이 증가한다. age-decay의 낮은 null FPR은 늦은 변화 검출 손실을 동반한다. bootstrap에서 zero clean detection interval이 [0,0]인 것은 모집단 확률이 정확히 0이라는 증명이 아니다.
- synthetic DGP는 유한 burn-in을 거친 근사 초기화다. N2/N3의 limiting stationary innovation fourth moment 조건을 유한 generated sample, clipped feature 또는 competition DGP에 그대로 적용하지 않는다. 실제 variance factor와 관측된 window-SD 방향은 다르다.
- 단일 이상치 뒤 age-grid 경계는 순수 expiry만 뜻하지 않는다. 상대 256과 절대 online 512의 최초 eligibility가 겹치는 등 startup confounding이 있다. 342-pair long-offset subgroup과 512-pair 짧은 분석을 합쳐 하나의 curve로 만들지 않았다.
- latency는 같은 장비에서 등록한 순차 측정 한 번의 결과다. 작업 프로세스는 분리했지만 OS/power는 통제하지 않았다. H/point/feature/tree timing 경계가 달라 부분 median 합은 total median이 아니다. 28개 component의 10M 값은 1M 외삽이며 12개 runtime의 10M 직접 측정과 다르다.
- tree-used 48은 최종 all-DEV 모델의 materialized input 수다. 내부 계산이 전부 77→48로 줄었다는 주장이 아니다. NumPy private/shared bytes는 Python/JIT/allocator/RSS와 다르다.
- strict joblib 재학습 byte check는 pickle alias metadata 때문에 실패했다. 실제 predictive arrays/Booster와 전 DEV 예측이 같은 것을 확인하고 canonical 형식으로 해결했다. 실패한 2회 fit도 총 fit 수에 포함했다.
- C210 sign complement는 adapter 이름 prefix 불일치로 0-feature BUG가 났으며 learner fit 전에 중단했다. 같은 가설을 C211로 고쳐 2-fold screen을 수행했지만 점수가 낮아 중단했다.
- alternate runtime proof V1은 row ordering/hash 초기화가 잘못된 harness 문제로 예측 비교 전에 중단했다. 원본 실패를 보존하고 V2에서 전체 원래 좌표 순서로 검증했다.
- J_PRECOMPUTED는 예측은 같고 약 6.21% 빨랐지만 사전 10% 교체 기준 미달이며 state 1,200 bytes 증가로 채택하지 않았다. 기준을 낮추지 않았다.
- 연구 중 advisory checkpoint pointer 경합, read-only NUL device 판정, figure 환경의 pandas 누락, unqualified Python의 numba 누락을 별도 기록했다. 마지막 summary는 pinned 환경에서 동일 bytes로 재생했다.
- 독립 검증 최초 복사 목록에 기존 import-time JSON default 8개가 빠져 static review로 학습 전에 보완했다. 최초 manifest/준비 기록과 보완 기록을 모두 보존했다.
- SealGuard와 Python audit hook은 application-level 경계다. Parquet mixed row-group의 내부 byte decoding/전체 파일 SHA 확인을 seal 관측값/label 평가와 혼동하지 않는다. native library를 강제하는 OS sandbox를 만들었다고 주장하지 않는다.
- all-DEV refit 예측은 재현 fingerprint로만 썼다. 온라인 inference에서 tau/ID/미래 길이/suffix를 feature로 사용하지 않았다. ZIP에는 raw competition 관측/label/real point OOF, credentials, 환경, 기존 archive를 넣지 않는다.
'''
    (ROOT/'LIMITATIONS_NEXT2.md').write_text(limitations, encoding='utf-8', newline='\n')
    plan = '''# 다음 세션

1. **ARCH/tail scale-down의 downstream 정보 손실을 분리한다.** J/Y/B를 그대로 보존한다. signed energy/forecast response가 유지되는 기존 반례에서 local/Bayes/GLR-memory/legacy 그룹의 역할을 정해진 matched 대조로 확인한다. 새 변형은 real DEV 비열등성/성능 기준과 함께 사전 등록하고, 합성 결과만으로 선택하지 않는다. long-age 256과 512는 별도로 대조해야 한다.
2. **장기 null과 late first-alarm utility를 함께 설계한다.** 새 memory/reset/threshold 정책의 finite-horizon FPR, premature alarm, clean detection, conditional delay를 동시에 잠근다. rank-preserving post-map이나 낮은 FPR만으로 개선을 선언하지 않는다. 현재 blind age-decay는 채택하지 않는다.
3. **독립 검증 기회를 보호한다.** 이번 performance gain은 generation jump가 아니다. 다음 고정 가설·메뉴·중단 조건·최소 gain과 fitting/selection uncertainty 검증 계획을 먼저 작성한다. C205의 DEPENDENCE 이득을 보존하면서 SCALE/이상치 손실을 줄일 수 있는지 한정된 대조로 본다. 추가 무작위 weight/seed/hyperparameter 탐색을 하지 않는다.

네 LOCAL SEAL과 reduced/public 사용은 계속 금지 상태로 둔다. 다음 세션에서도 기존 SHA·split·quota·source 상태를 먼저 확인한다. 이전 *_PENDING 로그만 보고 완료된 실험을 다시 학습하지 말고 최종 CSV, independent reproduction receipt와 SOURCE_MANIFEST를 우선 확인한다.
'''
    (ROOT/'NEXT_SESSION_PLAN_NEXT2.md').write_text(plan, encoding='utf-8', newline='\n')
    (ROOT/'SEAL_OPEN_RECOMMENDATION_NEXT2.md').write_text('''# Seal 개봉 권고: 개봉하지 않음

SEAL_A/B/C/D는 각 500개이며 모두 미개봉이다. DEV/seal overlap은 0, reduced 신규 사용과 public 신규 제출은 0이다. 최고 성능 후보는 +0.0016366625의 DEV 이득으로 performance gate를 통과했지만 generation-level 기준에는 못 미쳤고, SCALE·이상치·장기 null 약점이 남았다.

이번 연구 결과를 확인하려고 seal을 열 이유가 없다. 다음 가설과 검증 계획을 먼저 고정하고 독립 검증 기회를 보존한다. 이 문서는 미래 개봉이나 제출을 승인하지 않는다.
''', encoding='utf-8', newline='\n')
    files = ['00_GPT_READ_FIRST_NEXT2.md', '01_GPT_HANDOFF_NEXT2.md', 'LIMITATIONS_NEXT2.md', 'NEXT_SESSION_PLAN_NEXT2.md', 'SEAL_OPEN_RECOMMENDATION_NEXT2.md']
    write_json(ROOT/'artifacts/next2/final_reports/RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'script_sha256': sha256(Path(__file__)), 'selection_sha256': sha256(ROOT/'DEV_CANDIDATES_NEXT2.json'),
        'reproduction_sha256': sha256(ROOT/'REPRODUCTION_NEXT2.json'), 'files_sha256': {p: sha256(ROOT/p) for p in files},
        'required_numbered_answers': 35, 'language': 'Korean', 'research_seconds': research_seconds})
    print('NEXT2 KOREAN FINAL REPORTS WRITTEN', flush=True)


if __name__ == '__main__': main()

# NEXT4 한계

- 모든 모델 선택·ALT/H·기전 분석은 같은 DEV 8000개에서 이루어졌다. 네 seal은 미개봉이며 독립 확인 점수는 없다.
- Paired bootstrap은 fitted OOF 조건부 whole-series 재표본화다. learner 재학습과 적응적 선택의 전체 불확실성을 포함하지 않는다. 14개 원래 전체 CI는 모두 0을 포함한다.
- B2 개선은 주로 늦은 관측 구간에 집중되며, heavy-tail 분산 감소와 반복 이상치 기억에서 악화가 있었다.
- SCALE 변화 support는 132개다. 원래 raw SD 방향 subset 79/53, MEAN 21, TAIL 86에서는 supported 기전 일반화를 주장하지 않는다.
- H-SCALE 이식성이 전체 H 평균과 다를 수 있다. B1은 전체 H 평균이 좋아도 SCALE에서 음수 점 추정치를 보였다.
- 합성 수치는 고정 모델/생성기/seed/표본 범위에 대한 진단이다. fold-0 모델과 all-DEV 모델 결과를 같은 학습 범위처럼 비교하지 않는다.
- 1024점 null maximum으로 보정한 threshold는 무한 horizon 보장이 아니다. Paired alarm CI는 threshold calibration 불확실성을 포함하지 않는다.
- TreeSHAP margin 분해와 sigmoid secant allocation은 feature 제거 재학습이나 인과적 ablation이 아니다.
- Sign reflection은 원래 DEV의 고정 입력 stress이며 독립 데이터 또는 새 선택 후보가 아니다. 반전 ensemble/augmentation 학습은 하지 않았다.
- D3는 흡수 hazard approximation이다. full BOCPD 또는 null-safe anytime evidence라고 부르지 않는다.
- B2의 cached all-DEV 진단 fit은 fresh raw reproduction이 아니다. 새 모델이 실제 채택된 경우에만 별도 raw→feature→fit→inference 재현을 했다.
- CPU 시간은 이 PC·정확한 runtime·표본·paired 실행 조건에 한정된다. 배열 bytes는 전체 RSS가 아니다.
- 최초 cold import 실패와 native worker crash를 보존했다. 짧은 probe로 native crash는 재현되지 않아 원인은 확정하지 못했다.
- 일부 개발 unit/prototype 검사가 후보 등록보다 먼저 수행됐다. 실제 DEV fit/score 이전 등록과 source freeze는 검증했다. 45분을 넘긴 bottleneck 검토 간격은 실제 timestamp로 남겼다.
- 전체 ZIP은 raw/OOF/cache/환경을 제외한다. 다른 PC의 새 환경 설치나 외부 제출 시스템 실행을 완료했다고 주장하지 않는다.

별도 NEXT4_ROBUSTNESS는 D2_DISCRETE_SCALE_MIXTURE, 평균 0.6235961690, Δ 0.0003277562. Primary champion 교체는 아니다.

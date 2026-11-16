# NEXT2의 한계와 보존한 실패

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

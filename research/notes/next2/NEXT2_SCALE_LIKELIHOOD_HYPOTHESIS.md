# 사전 가설: energy의 평균 변화와 분산 변화 우도는 다르다

기존 variance bank는 H에서 보정한 energy score의 합을 정규 평균 변화 통계에 넣는다. 제곱 잔차는 음의 방향이 제한되고 큰 양의 값에 민감하다. 이 비대칭이 scale 감소 반례와 연결될 수 있다는 소스 기반 가설을 추가한다. 아직 이 가설의 새 DEV 결과를 보지 않았다.

고정 H 평균과 기준 분산 아래 `S = Σ z²`, window length n, `r=S/n`일 때 정규 분산의 최대화 log likelihood ratio는 직접 계산하여 `0.5*(S-n-n*log(r))`가 된다. r<1과 r>1을 각각 lower/upper evidence로 유지한다. 정규 표본에서 표본 분산과 기준 분산을 비교하는 고전 검정의 관계는 [NIST 분산 검정 설명](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm)을 참고했다. 실제 H 추정, ARCH, clipping 때문에 그 검정의 chi-square 임계값을 이 detector에 적용하지 않는다.

양의 분산 비율 θ에 `InvGamma(a,b)` prior를 쓰며 a=2 또는 8, b=a−1로 prior 평균을 1에 둔다. shape/scale 밀도 매개화는 [Stan 공식 함수 문서](https://mc-stan.org/docs/functions-reference/positive_continuous_distributions.html#inverse-gamma-distribution)에서 확인했다. 직접 적분하면 null θ=1 대비 log BF는 `a*log(b)-lgamma(a)+lgamma(a+n/2)-(a+n/2)*log(b+S/2)+S/2`다. 별도 수치 적분 테스트로 식을 검증한다.

H 잔차 제곱, 조건부 forecast, 정규화된 잔차 제곱은 실제로 동일한 likelihood 모형을 따르지 않는다. 따라서 아래 통계는 후보 feature이며, calibrated Bayes factor·posterior·e-process·anytime-valid 경보라고 주장하지 않는다. H에서만 energy 기준값을 계산하고 O에서는 고정 age 1/2/4/.../512의 prefix sum을 갱신한다. 여러 age의 선택/혼합도 경보의 유의수준 보장을 제공하지 않는다.

고정할 작은 비교는 (1) raw energy+forecast의 12개 scale 통계, (2) raw cap 4만 추가, (3) forecast를 conditional normalized energy로 교체, (4) 기존 Y001에 lower evidence 2개만 추가, (5) 기존 Y001에 short/reference와 short/long log ratio 4개만 추가다. 모든 후보는 이미 잠근 J102 screen/5-fold/paired/alternate 기준을 그대로 따른다. 이 가설을 위한 임계값 또는 learner 튜닝은 하지 않는다.

수치 표현도 학습 전에 고정한다. GLR feature는 log1p, log BF feature는 부호를 유지한 log1p 절댓값으로 저장한다. 정규화된 energy의 누적 입력에는 모든 변형에 같은 수치 보호 한계 1e6을 적용한다. 이는 robust |z|=4 비교와 별개다. 영변화 H와 매우 큰 유한 관측의 finite-output 테스트를 포함한다.

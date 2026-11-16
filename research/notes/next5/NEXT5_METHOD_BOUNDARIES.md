# NEXT5 순차 증거의 해석 범위

S 계열은 분산비 0.5, 0.7, 1.5, 2.0을 귀무 분산비 1과 비교한다. 표준편차 비율과 혼동하지 않는다. Gaussian log likelihood ratio는 `-log(v)/2 + z²(1-1/v)/2`다. Student-t5는 귀무 분산을 1로 맞추므로 SciPy의 scale 인자에 `sqrt(3/5)`를 사용한다. 독립적인 `scipy.stats.norm/t.logpdf` 계산으로 구현을 검증한다. [SciPy Student-t 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)

AR(8), ARCH와 채널 중심·표준편차는 해당 시계열의 H만으로 추정한다. 이후 업데이트는 현재 관측까지만 받는다. 수치 안정화를 위한 ±12 제한은 사전에 고정한다. H의 추정 오차, 꼬리와 의존성 때문에 실제 데이터의 likelihood 특징을 정확한 귀무분포의 확률 증거로 해석할 수 없다.

E 계열의 고정 혼합은 순차 검정 문헌에서 아이디어를 얻지만, 실제 대회 분포에서 anytime-valid alpha를 보장한다고 주장하지 않는다. 그런 보장은 적합한 귀무 아래의 test martingale/e-process 조건을 요구한다. 고정 시점의 점수 개선과 long-null에서의 경험적 경보율을 각각 보고한다. [Ramdas 등, Game-theoretic statistics and safe anytime-valid inference](https://arxiv.org/abs/2210.01948)

새 Shadow는 NEXT5에서의 추가 적응을 줄이기 위한 사전 잠금이다. 원래 DEV 8,000개는 과거 연구에서 사용됐으므로 독립적인 원래 LOCAL SEAL과 같은 증거가 아니다. Shadow에서 후보·특징·가중치·분할을 다시 선택하지 않는다.

## P3와 E1이 실제 검사한 설계

P3의 2026-09-11 05:44:39 UTC 사전등록은 빠르게 줄어드는 peak envelope와 모든 현재 evidence를 받는 느린 EWMA64를 비교한다. `persistence_step`의 shock-recovery branch에는 confirmation 여부에 따른 업데이트 차단이 없다. 따라서 사용자 예시의 “확인된 evidence가 있을 때만 persistent state가 증가”하는 구조를 구현한 결과로 부르면 안 된다. P1의 fast/slow confirmation feature와 P2의 H-threshold occupancy는 별도 후보다. 이번 P3 결과는 peak와 slow amplitude를 구분하는 이 제한된 설계의 결과이며, confirmation-gated 상태 전체의 성공 또는 실패를 판정하지 않는다.


두 후보의 원래 source·사전등록·결과를 점수 확인 후 다른 구조로 바꾸지 않는다. 실제 구현 범위와 요청의 더 넓은 연구 아이디어 사이의 차이는 최종 한계와 후속 계획에 남긴다.

## Median과 worst fold 정의의 차이

원래 `median_delta`와 bootstrap의 `estimate_median_of_folds_delta`는 `median(candidate scores) − median(reference scores)`다. 대응 fold 차이의 중앙값인 `median(candidate − reference)`와 같지 않다. 원래 bootstrap 구간 역시 중앙값끼리의 차이를 대상으로 했으므로 대응 차이의 중앙값에 옮겨 붙이지 않는다. 최종 표에 `median_matched_fold_delta`와 원래 0 하한의 통과 여부를 별도 추가한다.

P2는 중앙값끼리의 차이가 +0.000334489713857522이나 대응 차이의 중앙값은 −0.00013614896394376075다. X2는 각각 +0.002210418424208882와 +0.0010587484998473284다. 19개 full 중 중앙값 하한 통과 여부가 두 정의에서 달라지는 후보는 P2뿐이다. 모든 후보가 이미 평균 개선 기준에 미달하므로 finalist는 어느 정의에서도 없다.

동결된 연구 구현의 `worst_delta`는 `min(candidate fold scores) − min(reference fold scores)`다. 이는 `minimum_fold_delta = min(candidate − matched reference)`와 다를 수 있다. 사용자 요구의 “worst fold ≥ −0.001 degradation”을 모든 대응 fold의 악화 한도로 읽으면 후자가 직접적인 비교다. 최종 표에는 두 수치와 최소 대응 fold의 하한 통과 여부를 함께 둔다.

S4A에서는 최저 점수끼리의 차이는 약 +0.0000523이지만 최소 대응 fold 차이는 약 −0.0012384다. 따라서 두 검사의 통과 여부가 다르다. 이 구현 해석 차이는 한계로 명시하며 원래 사전등록·분석 결과를 사후 덮어쓰지 않는다. 19개 full 후보 모두 평균 증가분 +0.0012에 미달하므로 어느 해석에서도 이번 finalist 자격이나 역할 결정은 달라지지 않는다.

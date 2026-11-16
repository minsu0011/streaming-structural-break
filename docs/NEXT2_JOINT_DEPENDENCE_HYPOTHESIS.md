# 후속 가설: 확인된 dependence 신호의 작은 feature 통합

고정한 C205의 standalone DEV 평균은 0.5755159686249883, J102와의 평균 시간별 순위 상관은 0.42193977197036003이다. 사전 고정 80/20 blend의 평균은 0.6232684128173095로 J102 대비 +0.0016366625048295225다. 이는 새로운 학습의 최종 승격이 아니라, 독립적인 dependence 정보를 더 조사할 근거다.

다음 두 개의 작은 통합만 사전등록한다. J102의 기존 77개 feature에 조건부 분산 정규화 innovation의 lag 1/2/4/8 product를 (1) EWMA half-life 32 또는 (2) Bayes prior variance .25로 요약한 4개 feature를 추가한다. 두 후보 모두 81개 입력과 기존의 100-tree LightGBM/W3를 쓴다. 데이터 ID, tau, primary model prediction은 학습 feature가 아니다. 각 원본 fold에서 causal 통계와 해당 training labels로 단일 모델을 학습한다.

어떤 후보도 고정 screen/paired/alternate/engineering 기준을 면제받지 않는다. 현재 blend 결과를 본 뒤 제안한 적응적 실험이라는 사실을 ledger에 남긴다. 작고 동결된 두 가지 선택이며 continuous blend weight나 learner hyperparameter 탐색으로 확장하지 않는다.

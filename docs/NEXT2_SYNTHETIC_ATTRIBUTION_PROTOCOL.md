# 동결 합성 반례에서 예측 기여도 분해

분산 감소 반례의 낮은 점수가 raw energy, forecast, 기존 ABCD, dependence 중 어느 입력군과 함께 나타나는지 검사한다. 사전에 고정한 18개 시나리오와 512개 seed 전체를 재사용한다. 변경 시나리오는 같은 seed의 control/intervention 쌍을 유지한다. 모델은 이미 고정한 all-DEV J102/Y001/B104/C205 네 개이며 새 학습은 없다.

관측 시점은 t=0, t=31, 예정 변경 직전, 변경 직후, 변경 후 31/127/511, 마지막 t=1023이다. 겹치는 시점은 이름이 다른 진단 항목으로 같은 값을 사용한다. 예정 변경 시점은 합성 결과의 분석 좌표로만 사용하고 모델에는 전달하지 않는다. 무변화/단발성 이상치에서 tau가 -1인 계약도 그대로 유지한다.

원래 학습된 LightGBM의 `pred_contrib`를 사용한다. 마지막 열은 expected-value 항이며 입력군별 기여도 합과 이 항이 원래 raw margin을 복원하는지 검증한다. API의 출력 계약은 [LightGBM 공식 Booster 문서](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html)에 따른다. 기여도는 raw log-odds 단위다. 확률을 혼합하는 BL80의 log-odds 기여도로 두 모델의 값을 단순 합산하지 않는다.

입력군은 기존 ABCD, raw energy, log variance forecast, conditional dependence로 나누고, variance/dependence 통계는 local, GLR, Bayes, memory로 세분한다. control과 intervention의 같은 seed·분석 시점에서 기여도 차이를 계산한다. 이는 **동결된 예측 함수의 설명**이다. 상관된 feature 사이의 분배에 의존하므로 입력군의 실제 인과 기여나 구성요소 제거 실험을 대신하지 않는다. 성능 승격, 가중치 선택, 합성 데이터 학습에는 쓰지 않는다.

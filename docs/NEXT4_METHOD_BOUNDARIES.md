# NEXT4 방법과 검증의 범위

기준 모델은 NEXT2 frozen BL80이다. NEXT3 I2는 exploratory reference이고 E003은 최종 canonical 승인이 끝나지 않은 구현이다. NEXT3를 다시 가동하거나 미완료 배포 감사를 끝내는 작업으로 NEXT4 연구를 대체하지 않는다.

Family A는 H에서 정한 AR8/ARCH 정규화 뒤 innovation, magnitude, sign의 lag 1/2/4/8 상관 변화량을 두 고정 시간 척도에서 묶는다. 일반적인 Ljung–Box 통계는 여러 lag의 자기상관 제곱을 결합한다. [NIST 정의](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4481.htm), [R 공식 Box.test 설명](https://www.stat.ethz.ch/R-manual/R-devel/library/stats/html/box.test.html)

현재 구현은 EWMA 국소 모멘트와 H 상대 변화량을 사용하므로 그 검정과 동일하지 않다. 적합한 H, clipping, ARCH, 짧은 prefix와 의존성 때문에 chi-square p-value·검정 수준·e-value를 주장하지 않는다. 예측 입력의 의미는 경험적인 공동 상관 변화량이다. 개별 raw lag products를 일부 제거하거나 lag를 하나 추가한 실험과 구별한다.

순위 학습에서도 같은 time을 query group으로 묶었다는 사실만으로 TS-AUC와 목적이 같아지지 않는다. LightGBM의 기본 ranking objective/metric은 LambdaRank/NDCG이며 pair 가중치가 다르다. [LightGBM 4.7 공식 LGBMRanker 문서](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html). 과거 K/O 계열의 boosted RankNet과 NDCG 관련 실험을 그대로 반복하지 않고, Family E는 train 내부의 inner OOF 예측에 대한 작은 pairwise 결합을 별도 사전등록한다. 성공은 실제 fixed-fold TS-AUC로 판단한다.

NEXT4의 performance gate는 mean +0.0013, median −0.0005 이상, worst −0.001 이상, whole-series paired bootstrap P(Δ>0)≥0.90 및 95% CI lower>0, ALT 2/3 이상 positive, H·supported mechanism 손실 제한과 인과성/실행 비용 통과를 모두 요구한다. generation은 +0.0025, P>0.95, ALT 3/3, worst non-worse 및 새 독립 정보 또는 명확한 robustness를 요구한다. 정확한 authority는 `configs/next4/RESEARCH_POLICY_LOCK.json`이다.

모델과 split seed는 기존 값을 유지한다. NEXT4 bootstrap은 어떤 새 점수보다 먼저 고정한 seed 20260908+fold로 동일한 whole-series multiplicities를 참조·모든 후보에 공유한다. 과거 NEXT2/NEXT3 bootstrap seed 202609094+fold로 만든 배열과 섞지 않으며, NEXT4 reference 배열을 한 번 계산해 재사용한다. 새 draw를 선택하거나 seed를 바꿔 유리한 CI를 고르는 작업은 없다. interval은 fixed fitted OOF 조건부이고 adaptive DEV 선택이나 fitting 불확실성을 보정하지 않는다.

최소 연구 시간은 **관측된 연구·구현·검증 21,600초**다. 단순 wall 6시간으로 대체하지 않는다. 첫 감독 프로세스가 시작되기 전 시간과 이후 관측 공백은 소급 인정하지 않는다. 주기적인 감독 heartbeat나 stack dump만으로 productive time을 늘리지 않는다. stack dump는 별도 진단 파일에 기록해 작업 progress 판정과 분리한다. 최종 감사는 이 연구 기준을 충족한 후 별도로 45–60분 수행한다.

Family C의 bin 경계는 H에서 적합한 ARCH 잔차 좌표의 −2/−1/−0.5/0/0.5/1/2로 고정한다. H 기준 확률 질량과 causal EWMA 질량의 차이를 비교한다. C1은 squared Hellinger, C2는 자연로그 Jensen–Shannon divergence이며 거리의 제곱근을 사용하지 않는다. 독립 oracle로 SciPy의 Jensen–Shannon distance를 제곱해 비교했다. [SciPy 공식 정의](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html). C3은 중앙·중간·극단 bin과 비대칭 비율을 사용한다. 기존 rank-CDF uniformity feature의 bin 수를 조정한 실험은 아니다.

Family D의 mean/scale mixture는 고정한 대립가설 격자와 4~512의 sparse age만 계산하는 Gaussian working likelihood다. hazard 후보는 3개의 고정 hazard에서 단일 변화 이후 상태를 흡수 상태로 유지한다. run-length 전체를 추적하고 각 구간의 모수를 적분하는 완전한 Bayesian online changepoint detection을 구현했다고 주장하지 않는다. [Adams–MacKay 원 논문](https://arxiv.org/abs/0710.3742). 출력은 ranking feature이며 실제 데이터에서 posterior·Bayes factor·e-value의 보정된 확률 해석은 검증하지 않았다.

Family H는 H로 정규화한 네 채널의 고정 두 단계 선형 recurrence와 세 시간 척도를 사용한다. 전이 모수는 학습하지 않고 32개 state feature의 작은 logistic readout만 각 outer train에서 학습한다. 유한 prefix white-noise 분산 공식은 표준화 도구이며 ARCH·heavy-tail 데이터에서 유효한 검정 분산이라는 주장은 없다.

G1은 H 요약을 J102의 기존 77개 feature 뒤에 추가한다. G2는 main effect 없이 H가 J/C의 점수 차이 계수만 바꾸는 작은 nested pairwise 모형이다. 두 모형이 사용하는 H 길이는 관측 완료된 historical 길이이며, online의 최종 길이는 받지 않는다. 진단에서 사용하는 tau·기제 label·원본 ID는 학습 feature 및 현재 관측값을 처리하는 API로 전달하지 않는다.

합성 진단은 원래 고정 AR/ARCH 생성기의 seed와 독립 calibration/evaluation/long-null partition을 유지한다. variance multiplier 0.7/0.5는 표준편차 배율이 아니라 혁신 분산 배율이다. 각 모델의 별도 1,024점 null calibration maximum으로 임계값을 정하고 4,096/16,384점에서도 그대로 사용한다. 장기 eventual alarm의 증가는 여러 시점의 기회가 누적된 결과일 수도 있으므로 1,024점 calibration을 anytime 보장으로 해석하지 않는다. Fold-0 모델은 같은 fold-0의 J/C/BL 기준과 비교하고, all-DEV 모델은 all-DEV 기준과 비교한다. B2의 추가 all-DEV fit은 합성 진단용이며 fresh raw 재현 또는 승격을 의미하지 않는다.

14개 full 후보의 joint uncertainty 진단은 기존 2,000회 paired draw를 그대로 사용한다. 각 후보의 centered bootstrap deviation을 empirical SD로 나눈 뒤, 같은 draw에서 후보별 최대값을 계산하는 단일 단계 기술 분석이다. 의존성을 이용하는 resampling multiple testing의 문헌 맥락은 [Romano–Wolf (2005)](https://www.econ.uzh.ch/dam/jcr%3Affffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf)를 참고했다. 이 분석은 그 논문의 stepdown 구현이 아니며, 전체 적응적 탐색이나 반복 학습을 포함한 unconditional FWER 보장도 아니다. 14개 후보 snapshot의 조건부 민감도만 보고하고 원래 promotion gate를 대체하지 않는다. Bootstrap에서 최고점인 빈도는 posterior 확률이 아니며 정확한 동점은 정렬된 candidate ID 순서로 처리한다. 상관 행렬의 participation ratio도 후보 수 제한을 늘리는 근거가 아니다.

시계열 하나를 제외하는 진단은 이미 학습된 OOF를 유지하고 그 시계열의 평가 행만 제거한다. 재학습한 leave-one-out CV나 새로운 신뢰구간이 아니다. 모든 시계열은 공식 최종 점수에 그대로 포함하며 영향력이 큰 관측을 찾아 제거하지 않는다. 시간 구간별 점수 기여는 해당 fold의 실제 positive×negative 비교쌍 질량으로 정규화하여 합이 원래 TS-AUC 변화량과 일치하는지 확인한다.

후속 memory 진단은 이미 등록한 동일 합성 예측만 읽는다. 단일 또는 16점 burst의 마지막 교란 직후 256점에서 clean 대비 양의 점수 잔류를 비교하고, 32점 연속으로 excess≤0.01인 첫 구간을 기술한다. 그 시작을 알려면 해당 구간을 다 관측해야 하므로 실시간 stopping rule이 아니다. 음의 excess는 양의 오염이 사라진 것으로만 처리하며, 영구 회복을 보장하지 않는다. Null에서는 1,024점 단위의 marginal score·구간 알람·이전 무알람 시계열에 대한 새 알람을 구별한다. Paired alarm/excess interval은 학습 모델과 이미 정한 calibration threshold에 조건부이며 calibration 불확실성과 모델 선택을 포함하지 않는다.

D3의 absorbing single-change filter에서 대안 k의 null 대비 odds는 `R[t,k] = LR[t,k] * (R[t-1,k] + h/K) / (1-h)`이다. 따라서 이상적인 iid N(0,1) null에서 기존 odds에 곱해지는 항의 기대 log는 `-KL(null || alternative) - log(1-h)`다. h=1/32와 mean ±0.25 대안에서는 이 값이 약 +0.0004987이다. 이 계산은 현재 코드의 대수적 성질이고 실제 whitened/clipped residual이나 trained D3 점수의 오경보율을 측정한 결과가 아니다. 모든 LR=1인 중립 경계에서는 변화 posterior가 `1-(1-h)^t`가 된다는 사실도 구별해서 기록한다. [Adams–MacKay의 원래 BOCPD](https://arxiv.org/abs/0710.3742)는 최근 change 이후 run length를 추론한다. 이 논문을 현재의 absorbing filter 또는 위 drift 계산 자체의 출처로 제시하지 않는다. 검증 후 hazard나 candidate를 바꾸는 실험은 추가하지 않는다.

이상치 기여도는 [LightGBM의 `pred_contrib` 출력](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)으로 raw margin의 feature 기여도와 expected-value 열을 구하고, 그 합을 native raw margin과 대조한다. [Tree SHAP 문헌](https://arxiv.org/abs/1802.03888)은 이 모델 설명 방식의 배경이다. Clean/교란 margin 차이에 대한 sigmoid secant를 각 기여도 차이에 곱한 뒤 기존 blend 비율을 적용한다. 이 값은 합이 맞는 확률 변화 분배이며 probability-space SHAP 값으로 부르지 않는다. Frozen partner의 변화와 float32 rounding 잔차도 따로 남긴다. 상관된 feature를 가진 예측 모델의 설명이므로 실제 데이터 생성 기제에 대한 인과적 기여나 feature 제거 효과를 의미하지 않는다. 첫 128개 원래 합성 seed·두 outlier 시나리오·모델 네 개·feature 묶음은 해당 합성 점수를 보기 전에 고정하고, 이 결과로 모델을 수정하지 않는다.

B2의 일부 합성 감소 조건에서는 높은 TS-AUC와 낮은 경보 탐지율이 함께 관측돼 후속 설명 분석을 등록했다. 이 분석은 관측된 현상에서 출발한 사후 진단이다. 기존 독립 null trajectory maximum의 임계값과 이미 계산한 예측을 그대로 두고, 변화 이후 maximum의 분포·임계값과의 차이·변화 이전 경보가 이후 first detection을 가린 비율을 대조한다. TS-AUC는 같은 시점의 positive/negative 순위이고, 경보는 모델별 null 최대치 임계값의 최초 통과이므로 두 결과는 다를 수 있다. 경험적 null-max percentile을 p-value나 anytime 보장으로 부르지 않으며, 새 threshold·weight·candidate 또는 추가 seed를 선택하지 않는다.

추가 sign-reflection 진단은 원래 DEV의 H와 현재 online 값에 각각 -1을 곱하는 단 하나의 고정 변환이다. 가역적인 부호 반전은 변화 시점 label을 보존하지만, 이미 학습된 signed feature와 tree의 출력은 달라질 수 있다. J102/C205/BL80 및 B1/B2/D2의 원래 5-fold 모델을 고정하고, reflected score를 계산하기 전에 대상·변환·원래 2,000개 bootstrap multiplicity 재사용을 등록했다. 실제 결과를 본 후 반전 ensemble, augmentation fit, 새로운 gate나 후보를 만들지 않는다. 이 진단은 새로운 독립 데이터나 배포 recipe가 아니다.

실제 강건성 finalist의 all-DEV 적합은 원래 등록한 feature·constant-column 처리·W3 weight·LightGBM 설정·blend를 그대로 사용한다. 새로운 동작은 fitted artifact를 현재 scalar API에 연결하는 것뿐이며, 연구 순위를 높이기 위한 state 최적화는 하지 않는다. Nomination은 승격이 아니다. 같은 실제 artifact에 대해 원래 all-DEV 크기의 전체 합성 suite와 동일 memory 통계를 다시 확인한다. 이 추가 scope는 finalist nomination에서 score 이전에 고정하며, 원래 fold-0 증거와 구별한다. 반례 생성기·seed·threshold·horizon·bootstrap 규칙은 변경하지 않는다.

새 강건성 runtime 측정은 원래 128개 real sample과 실제 10M update를 사용하되 연구 단계에서 BL80/후보, 후보/BL80의 두 고정 쌍, 별도 감사에서 한 고정 쌍으로 정했다. 이는 B2 탐색 진단의 다섯 쌍 측정과 별개의 사전 명시된 유한 검증 범위다. 같은 scalar timing 함수를 재사용하며 감사 시 activity phase 표기만 바꾼다. 기존 모델 파일과 mathematical prediction은 변경하지 않는다.

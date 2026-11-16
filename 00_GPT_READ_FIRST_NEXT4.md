# NEXT4 GPT 전달 — 완료된 과정과 결과

NEXT4_ROBUSTNESS_ONLY_NO_PERFORMANCE_PROMOTION

주 성능 모델은 **BL2_C205_DEPENDENCE_80, 0.6232684128173095**를 유지한다. 탐색 최고 B2의 원래 CI 실패를 유지했다. 별도 NEXT4_ROBUSTNESS는 D2_DISCRETE_SCALE_MIXTURE, 평균 0.6235961690, Δ 0.0003277562. Primary champion 교체는 아니다.

1. **실제 productive research 시간**

22531.457초 (6.2587시간). 21,600초 이후 시작한 별도 audit 관측 시간은 2819.425초다. 소급 산입하지 않았다.

2. **wall-clock**

최종 보고서 측정 시점 2026-09-10T19:06:44.395107+00:00, 세션 경과 26564.937초. 관측되지 않아 연구·감사로 인정하지 않은 시간은 1214.249초. 감사 실제 경과 2899.605초. ZIP 생성 이후 시각은 외부 delivery receipt에 별도 기록된다.

3. **새 hypothesis family 수**

사전 등록 8개 A–H. 실제 새 DEV score가 있는 family는 7개다. Family별 screen/full 수는 A 3/3, B 2/2, C 3/3, D 3/3, E 2/1, F 0/0, G 2/1, H 1/1이다. F의 두 slot은 기존 고정 조합 중복과 유효한 세 번째 expert 부재로 새 fit/score 없이 닫았다.

4. **screen candidate 수**

16개. E2와 G1은 고정 screen 실패 후 종료했다.

5. **full candidate 수**

14개. 상한 24개를 넘지 않았고, 합성·입력 반전 진단을 새 배포 모델로 선택하지 않았다.

6. **learner fit 수**

새 실제 DEV learner fit 완료 166회. 단계별 {'selection_screen': 32, 'selection_full': 70, 'robustness_ALT_1': 15, 'robustness_ALT_2': 15, 'robustness_ALT_3': 15, 'robustness_LEAVE_ONE_H_REGIME_OUT': 15, 'all_DEV_diagnostic_all_dev_v1': 1, 'all_DEV_nominated_all_dev_v1': 1, 'negative_control': 1, 'fresh_raw_reproduction': 1}. 선택용 102회 중 base/state 86회, nested meta 16회. 과거 nested base 56개를 재사용했고 새 frozen-base fit은 0회다. 개발용 합성·mock fit은 이 수에 포함하지 않았다.

7. **시작 champion**

BL2_C205_DEPENDENCE_80 = 0.8 J102 + 0.2 C205.

8. **시작 exact score**

0.6232684128173095 — 원래 primary 5-fold TS-AUC의 동일 가중 평균.

9. **탐색 최고 score**

B2_LOG_ENERGY_SEMITAIL: 0.6245720862584431, Δ 0.0013036734411336682. 탐색 최고와 승격은 다르다.

10. **최종 선택 score**

FROZEN_PERFORMANCE 유지: 0.6232684128173095. 별도 NEXT4_ROBUSTNESS는 D2_DISCRETE_SCALE_MIXTURE, 평균 0.6235961690, Δ 0.0003277562. Primary champion 교체는 아니다.

11. **최종 delta**

주 성능 champion은 시작 대비 0.0. 별도 NEXT4_ROBUSTNESS는 D2_DISCRETE_SCALE_MIXTURE, 평균 0.6235961690, Δ 0.0003277562. Primary champion 교체는 아니다.

12. **새 generation 여부**

없음. NEXT4_ROBUSTNESS_ONLY_NO_PERFORMANCE_PROMOTION. +0.0025 generation 문턱을 넘은 후보가 없다.

13. **strongest new idea**

B2의 raw/conditional lower·upper log-energy 분리는 가장 큰 원래 평균 점 개선을 보였다. 다만 CI 하한이 음수이고 heavy-tail 분산 감소 및 반복 이상치에서 한계가 확인됐다. 새 독립 신호로 확정하지 않았다.

별도 강건성 근거는 D2_DISCRETE_SCALE_MIXTURE의 고정 분산 likelihood 신호다. 제한된 SCALE support와 실제 파일 반례 확인을 근거로 유지하며, 전체 성능 향상이나 독립 expert 확증으로 확대하지 않는다.

14. **dependence V2 결과**

A1_INNOVATION_PORTMANTEAU: 0.6234185702 (Δ 0.0001501574); A2_MAGNITUDE_PORTMANTEAU: 0.6234974619 (Δ 0.0002290491); A3_SIGN_PORTMANTEAU: 0.6233875848 (Δ 0.0001191720). 원래 전체 CI가 모두 0을 포함한다. magnitude/sign/innovation portmanteau 신호가 이번 고정 learner/혼합에서 유의한 성능 승격으로 이어지지 않았다.

15. **scale asymmetry 결과**

B1_SPLIT_ENERGY_BRANCHES: 0.6238162402 (Δ 0.0005478274); B2_LOG_ENERGY_SEMITAIL: 0.6245720863 (Δ 0.0013036734). B1은 SCALE 및 네 분산 감소 반례 조건을 통과했지만 H 제외 SCALE 효과가 불확실하고 반복 이상치 오염은 더 컸다. B2는 전체 CI와 SCALE CI 하한 조건을 모두 통과하지 못했다.

16. **distribution result**

C1_FIXED_BIN_HELLINGER: 0.6233029967 (Δ 0.0000345839); C2_FIXED_BIN_JS: 0.6233163430 (Δ 0.0000479301); C3_SHAPE_RATIO_DISTRIBUTION: 0.6233859867 (Δ 0.0001175739). H 고정 bin의 Hellinger/JS/shape-ratio 신호를 시험했다. 원래 전체 CI는 모두 0을 포함한다.

17. **sequential likelihood result**

D1_DISCRETE_MEAN_MIXTURE: 0.6230707564 (Δ -0.0001976564); D2_DISCRETE_SCALE_MIXTURE: 0.6235961690 (Δ 0.0003277562); D3_HAZARD_MIXTURE: 0.6231231986 (Δ -0.0001452142). D3는 제한된 흡수 hazard mixture이며 full BOCPD가 아니다. 단순한 null에서도 작은 mean 대안과 hazard가 장기 누적 압력을 만들 수 있음을 분석했다.

18. **ranking objective result**

E1 nested pairwise linear Δ -0.0000467485; E2 interaction은 screen Δ -0.0031355385로 중단. 학습 pair loss 감소가 원래 held-out TS-AUC 개선으로 연결되지 않았다. Outer-validation base prediction을 meta 학습에 쓰지 않았다.

19. **H-context result**

G1 H-context base screen Δ -0.0348259444로 중단. G2 honest nested pairwise context full Δ +0.0000017733. G1에서 H-only feature가 split gain 약 66–68%를 차지했지만, 이것은 인과적 실패 원인 검증이 아닌 학습 구조 관찰이다.

20. **새 complement result**

D2 standalone 평균 약 0.60007529, 고정 10% 조합 개선 +0.00032776였으나 신규 독립 expert 채택의 CI/P 조건을 통과하지 못했다. F1은 기존 72/18/10 조합과 중복, F2는 유효한 세 번째 성분 부재로 미실행. H1 고정 선형 상태 readout도 full Δ -0.00068465로 종료했다.

21. **5-fold scores**

최종 주 성능 BL80: [0.628814140424981, 0.6371811063970156, 0.6216260123520351, 0.618712391914872, 0.6100084129976436]. 탐색 B2: [0.6297026893327257, 0.6382528524422982, 0.6232881288442872, 0.6204512881442719, 0.6111654725286327]. 별도 강건성: [0.6288927015007282, 0.639007947929381, 0.621325451505788, 0.6184411062775395, 0.6103136378823955]

22. **median**

BL80 0.6216260123520351; B2 0.6232881288442872.

강건성 0.621325451505788.

23. **worst fold**

BL80 0.6100084129976436; B2 0.6111654725286327.

강건성 0.6103136378823955.

24. **bootstrap CI**

B2−BL80 원래 paired 95% CI [-0.000018936917, 0.002632276899]. 하한이 음수다. 원래 2,000개 whole-series multiplicity를 동일하게 재검증했으며 추가 표본으로 CI를 뒤집지 않았다. Frozen 모델 자신과의 차이는 정의상 0이며 새로운 성능 근거가 아니다.

강건성 전체 CI [-0.000273254935, 0.000909263558]이며 전체 성능 승격 근거가 아니다.

25. **P(delta>0)**

B2−BL80 = 0.973. 높은 P만으로 양수 CI 하한 조건을 대체하지 않는다. 재학습·적응적 선택 불확실성을 포함한 확률이 아니다.

강건성 전체 P=0.8545.

26. **ALT_1/2/3**

B2 원래 고정 대체 split의 Δ: ALT_1 0.0022978133 CI [0.000926619496, 0.003560132588], ALT_2 0.0013215833 CI [0.000108660753, 0.002415019364], ALT_3 0.0013606900 CI [0.000130476389, 0.002649442327]. 모두 기존 DEV의 다른 grouped partition이며 독립 신규 데이터가 아니다.

강건성 역할의 전체 ALT Δ: ALT_1 0.0004318193 CI [-0.000162679024, 0.001006537073], ALT_2 0.0004192810 CI [-0.000147062479, 0.001020208813], ALT_3 0.0005440281 CI [-0.000065807808, 0.001187917519].

27. **H-regime worst**

B2의 H 제외 matched 최저 Δ -0.0014296837999978607. 전체 H 평균 Δ는 약 +0.0004389151. REGIME 표의 원래 OOF 조건부 H slice와 H-regime 전체 제외 후 재학습을 구별했다.

강건성 역할의 H 제외 평균 Δ -0.0002154336748672936, matched 최저 Δ -0.0026844631619427073, 전체 CI [-0.001113612256, 0.000719166240]. SCALE 이득과 전체 H 이득은 다르다.

28. **mechanism trade-offs**

SCALE 실제 변화 support 132개. B2 SCALE Δ +0.00621335이나 CI [-0.00079629,+0.01387189]로 불확실하다. raw SD 방향 subset은 감소 79/증가 53개로 각 100개 미만이므로 supported variance-direction 일반화로 주장하지 않는다. MEAN 21, TAIL 86도 표본 부족이다.
강건성 역할 D2_DISCRETE_SCALE_MIXTURE: 원래 SCALE Δ 0.0089358282, CI [0.005362693966, 0.012762924907], 변화 시계열 132개. 원래 전체 CI는 [-0.000273254935, 0.000909263558]이다. 상세 ALT/H·기전·반례 한계는 [채택 검토](docs/NEXT4_FINAL_TRANSFER_REVIEW.md)에 남겼다.


강건성 SCALE의 재학습 split별 Δ와 CI: ALT_1 0.0087447034 [0.004938745192, 0.012911346060]; ALT_2 0.0081215284 [0.003958124506, 0.012678690474]; ALT_3 0.0099338265 [0.005342790034, 0.015467068863]; LEAVE_ONE_H_REGIME_OUT 0.0044282495 [0.001668642865, 0.007987531291]. 같은 DEV 재분할에 대한 조건부 검증이다.

29. **early/late fracture**

B2는 time 0–99에서 손실, 200 이후에서 주 이득을 보였다. time≥200의 실제 pair-mass 기여 +0.00151536이 전체 +0.00130367보다 커서 앞부분 손실을 상쇄한다. time 0–9에서 실제 scoring-positive 시계열은 182개이며 전체 eventual-break 3944개가 아니다.

강건성 원래 time 구간별 Δ: time_0-9 -0.0004774100; time_10-24 -0.0010249941; time_25-49 -0.0002156802; time_50-99 -0.0001070176; time_100-199 0.0002779077; time_200-399 0.0001667546; time_400+ 0.0007691810. 신뢰구간은 TIME_FRACTURE_NEXT4.csv에 모두 보존했다.

30. **long-null**

모델마다 독립 1024점 null maximum으로 임계값을 정한 뒤 1024/4096/16384에 고정했다. B2 N3 16384점의 누적 alarm fraction은 α=.01에서 0.1875, α=.05에서 0.517578125. 각각 BL80 0.146484375, 0.546875. 명목 α를 모든 길이에 보장하는 anytime 검정은 아니다.

실제 all-DEV 강건성 파일의 16384점 alarm fraction (각각 후보/동시 기준 BL80): N0 α=0.01 0.080078125/0.07421875; N0 α=0.05 0.3359375/0.330078125; N1 α=0.01 0.15234375/0.146484375; N1 α=0.05 0.388671875/0.3984375; N2 α=0.01 0.078125/0.08203125; N2 α=0.05 0.478515625/0.46875; N3 α=0.01 0.154296875/0.146484375; N3 α=0.05 0.515625/0.546875.

31. **outlier**

B2의 512쌍 반복 이상치 OB16에서 이후 평균 excess 차이 +0.033498, 마지막 64점 +0.057189로 BL80보다 컸다. 고정 첫 128 seed의 별도 TreeSHAP 분해에서는 upper-energy와 기존 ARCH 상태가 기여했다. SHAP subset과 paired 통계 표본 수를 섞지 않았다.

실제 all-DEV 강건성 파일의 512쌍 이후 평균 signed excess 차이: O1 Δ -0.019297926753779393 CI [-0.01986783582471503, -0.01868381102408989]; OB16 Δ 0.004762802824188839 CI [0.003946145333384265, 0.005599417345865732]. 음수는 동일 반례에서 기준보다 오염이 작다는 뜻이다. D2는 단발 이상치에서는 개선했지만 반복 이상치는 악화했다. fold-0에서의 반복 이상치 개선이 실제 전체 학습 파일에는 유지되지 않았다. 전반적인 outlier repair로 주장하지 않는다.

32. **latency**

B2의 실제 scalar 10M, 5회 paired 측정 median ratio 2.012434, scalar median ratio 1.963731, H 초기화 포함 amortized ratio 2.239702. 모두 3배 이하. 오래된 90초대 기록과 직접 비율을 계산하지 않았다. 추가 강건성 역할의 연구·감사 측정은 LATENCY_NEXT4.json 참조.

강건성 연구 paired 측정의 중앙 비율: {'ten_million_ratio': 1.8760807062020808, 'scalar_median_ratio': 1.8840579710144927, 'amortized_ratio': 1.678413980429725}. 별도 감사에서 원래 고정 순서의 10M paired 실행을 다시 확인했다.

33. **state size**

고유 mutable NumPy 배열 기준 BL80 16,925 bytes, B2 35,029 bytes, 길이에 따라 증가하지 않음. Python 객체·RSS·불변 tree 배열 전체 크기는 별도 개념이다. 강건성 25821 bytes.

34. **leakage tests**

최종 전체 검사 1183개 PASS, 실패/오류/skip 0. DEV-only guard, scorer, nested 구조, 현재 관측 하나당 출력 하나, 첫 None, 미래 길이·suffix trap을 검증했다. label permutation control expert는 0.58698915→0.46606936. 제한된 ID rename mock 및 실제 20개 시계열 역순 reset 검사를 구별한다.

35. **determinism**

최종 감사에서 BL80와 B2 및 강건성 모델의 DEV 8000개/4,014,405개 출력에 대해 offline/stream과 원래 fingerprint가 정확히 일치했다. 새 프로세스 cold import 및 복사된 raw-free payload 실행도 확인했다. 최초 B2 cold mkdir 실패는 보존했고 import-only 수정 후 통과했다.

36. **fresh reproduction**

선택한 강건성 모델을 fresh output에서 raw DEV→새 feature→동일 learner→inference로 재현했다. feature·학습 좌표와 inference fingerprint가 기존 materialization과 정확히 같았다. 기존 frozen base는 다시 학습하지 않았다.

37. **DEV saturation risk**

HIGH. 원래 full 14개의 전체 CI가 모두 0을 포함한다. 1위 B2−2위 B1 차이 +0.0007558460의 CI는 [-0.0005184872,+0.0019573212], P=0.8835다. B2 ALT gain의 표본 SD는 약 0.0005526835. 원래 draw에서 B2가 최고인 빈도 0.7935는 posterior 확률이 아니다. 기존 draw의 joint max-statistic은 조건부 탐색 진단이며 unconditional FWER나 새 데이터 확인을 대체하지 않는다. 새 seed·split·미세 blend를 추가하지 않았다.

38. **SEAL_A/B/C/D opened?**

아니오. 네 seal 500개씩 전부 미개봉. raw 전체 바이트 SHA 확인과 seal 관측/label 읽기를 구별했다.

39. **reduced usage?**

이번 NEXT4 새 사용 0. 과거 누적 2/2는 변경하지 않았다.

40. **public submission?**

이번 NEXT4 public 0, cloud 0. SDK test/submit을 호출하지 않았다.

41. **seal open recommendation**

NO. 원래 performance/generation gate 통과가 없고 반복 DEV 탐색 위험이 높다. 별도 SCALE 강건성 역할이 있더라도 독립적인 일반화 확증은 아니다. 실제 개봉은 하지 않았다.

42. **다음 연구 병목 3개**

① H/ARCH·heavy-tail 조건이 달라져도 재현되는 분산 감소 evidence. ② 반복 이상치 뒤 상태의 회복과 긴 null에서 누적 alarm 제어. ③ 제한된 DEV에서 objective/독립 신호 개선을 정직하게 확인할 검증력. 닫힌 family의 seed·weight shopping을 재개하지 않는다.

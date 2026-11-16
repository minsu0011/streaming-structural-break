# NEXT5 GPT 전달 — 완료된 연구와 검증

기존 BL80 primary를 유지한다. 새 NEXT5 역할은 없고 Shadow와 네 LOCAL SEAL은 미개봉이다.

1. **productive research time**

47964.728초. 실제 CPU/log 또는 실질 작업 신호로 관측한 시간이며 36,000초 이상이다. 별도 final audit 3882.218초는 연구 시간에 합산하지 않았다.

2. **wall-clock**

54305.680초. 관측 불가 시간 2459.053초를 연구 시간으로 소급 인정하지 않았다. 최종 보고서 시점의 수치이며 압축 검증 종료 시각은 Desktop 영수증에 별도 기록한다.

3. **family 수**

S/P/W/E/R/M/X의 7개. S3는 reference 정보 분해로 새 candidate나 learner fit이 아니다.

4. **screen 수**

20개. 고정 2560개 ResearchPool subset의 fold 0/4에서 평가했다. M2 1개는 탈락했다.

5. **full 수**

19개. 고정 ResearchPool 6400개, grouped 5-fold. 원래 한도 20개를 초과하지 않았다.

6. **learner fit 수**

295개: screen 40, full 95, PRIMARY reference 24, label permutation 1, ALT/H reference 30, ALT/H candidate 105. 집계 결과와 unit fixture 학습은 중복 계산하지 않았다.

7. **시작 primary**

기존 BL2_C205_DEPENDENCE_80(BL80), next2_shared_performance_main.py. 원래 frozen 파일을 유지한다.

8. **시작 score**

0.6232684128173095. 이전 8000개 DEV의 역사적 점수다. 이번 동일 RP 학습 범위의 BL80은 0.6203652100698587이며 두 모집단 점수를 직접 증감 비교하지 않는다.

9. **ResearchPool best**

X2_SCALE_TAIL: mean 0.6215608600890692, Δ 0.0011956500192103591. 사전 기준 +0.0012에 4.349980789640755e-06만큼 미달한다. 반올림으로 승격하지 않았다.

10. **Shadow finalist 목록**

없음, 0개. 모든 full 후보가 공통 평균 증가분 기준에 미달했고 M2는 screen에서 탈락했다. 빈 manifest를 해시로 고정했다.

11. **Shadow scores**

없음. 미평가를 0점이나 패배로 바꾸지 않았다. Shadow 1600개에 대한 후보 평가·훈련·특징 분석을 하지 않았고 분석 경로의 반환 행은 0이다. Parquet 내부 디코딩까지 0임을 독립 계측한 것은 아니다.

12. **최종 selected model**

FROZEN_PRIMARY=기존 BL80 유지. 기존 D2는 NEXT4_ROBUSTNESS의 SCALE 보조 역할로 유지한다. 새 NEXT5_PERFORMANCE/NEXT5_ROBUSTNESS는 없다. B2는 research near-miss reference다.

13. **최종 estimated gain**

이번에 채택하거나 주장하는 새 primary 증가분은 0이다. X2의 관측 RP 증가분을 hidden-test 개선 추정치나 채택된 증가분으로 사용하지 않는다.

14. **B2/D2 fusion 결과**

S1_DIRECTIONAL_SCALE Δ 0.0002066288; S2_STUDENT_SCALE Δ -0.0000693860; S4A_AUGMENTED_J_SCALE Δ -0.0002944242; S4B_UNIFIED_SCALE_SPECIALIST Δ 0.0009173129. X 결합 결과: X1_SCALE_PERSISTENCE Δ 0.0010930454; X2_SCALE_TAIL Δ 0.0011956500; X3_DEPENDENCE_SCALE Δ 0.0009979032. S4B의 분리 scale/tail 은행과 X 결합은 일부 RP 개선을 보였으나 새 역할을 얻지 못했다. 동일 RP B2 composite Δ +0.0011102871043433328, D2 composite Δ +0.0007385961732173874. S3 pair-rescue와 damage를 함께 기록했다. RP 최상위 X2의 고정 합성 감소 조건에서는 X2 V3 평균Δ -0.0114157209 (고정 파일 범위 -0.0183667436∼-0.0032145725, 양수 0/5); X2 V4 평균Δ -0.0090880415 (고정 파일 범위 -0.0162378194∼-0.0035064584, 양수 0/5). 원래 RP SCALE 구간의 양수 이득과 다른 배경의 감소 손실을 함께 보존하며, 이 범위를 신뢰구간이나 hidden-test 효과로 해석하지 않는다.

15. **scale likelihood 결과**

S1 Gaussian과 S2 unit-variance Student-t5의 lower/upper likelihood를 분리했다. 후보 분산 배율은 0.5/0.7/1.5/2.0이며 표준편차 배율과 구별한다. S1_DIRECTIONAL_SCALE Δ 0.0002066288; S2_STUDENT_SCALE Δ -0.0000693860; S4A_AUGMENTED_J_SCALE Δ -0.0002944242; S4B_UNIFIED_SCALE_SPECIALIST Δ 0.0009173129. 합성 반례의 다섯 고정 파일에서는 S4B V3 평균Δ -0.0143382156 (고정 파일 범위 -0.0194717068∼-0.0058266225, 양수 0/5); S4B V4 평균Δ -0.0075542355 (고정 파일 범위 -0.0137301487∼-0.0000791402, 양수 0/5); S4A CB_N3_050 평균Δ 0.0054129155 (고정 파일 범위 -0.0342192247∼0.0519937803, 양수 3/5). S4A의 유리한 첫 파일을 전체 학습 파일의 안정성으로 일반화하지 않는다. 범위는 신뢰구간이 아니다.

16. **robust null 결과**

R1_H_ADAPTIVE_TAIL Δ 0.0003532675; R2_CONDITIONAL_SCALE_PIT Δ 0.0008891720; R3_QUANTILE_SELF_NORMALIZED Δ 0.0006882955. H-only tail/PIT/self-normalization을 사용했다. 고정 합성 조건에서 R2 CB_N3_050 평균Δ 0.0117041463 (고정 파일 범위 0.0071587564∼0.0186948520, 양수 5/5); R3 V3 평균Δ 0.0047647181 (고정 파일 범위 0.0022884835∼0.0079304016, 양수 5/5). 별도 학습 후보의 조건별 결과이며 변환 하나의 인과적 ablation은 아니다. H에서 추정한 종속 working null이므로 정확한 distribution-free calibration이나 anytime 보장을 주장하지 않는다.

17. **persistence 결과**

P1_TWO_TIMESCALE_CONFIRMATION Δ 0.0007606047; P2_SOFT_PERSISTENCE_COUNT Δ 0.0002388035; P3_SHOCK_RECOVERY Δ 0.0005604080. 5개 학습 artifact별 단발/반복 충격, late genuine break, long-null을 함께 검사했다. P1: late 15개 조건 중 고정 파일 평균 TS-AUC Δ가 음수인 조건 1개; P2: late 15개 조건 중 고정 파일 평균 TS-AUC Δ가 음수인 조건 6개; P3: late 15개 조건 중 고정 파일 평균 TS-AUC Δ가 음수인 조건 3개. onset=512의 기존 85개 paired seed에 한정되며 새 확증 gate가 아니다. 조건별 평균·범위·알람 방향은 01_GPT_HANDOFF_NEXT5.md의 표에 모두 남겼다. P3는 peak와 ungated slow EWMA를 비교하며 confirmation-only 상태는 미구현이다. P3의 결과로 그 더 넓은 구조를 반증하지 않는다. 0.9 BL80 혼합의 점수 진폭 감소와 실제 expert 추가 응답을 분해했으며 낮은 raw excess만으로 회복 개선을 선언하지 않는다.

18. **windowed evidence 결과**

W1_RECENT_SCALE_RATIO Δ 0.0002533847; W2_TRIMMED_RECENT_ENERGY Δ 0.0007209444; W3_MULTIWINDOW_DISAGREEMENT Δ 0.0001483177. 고정 길이 recent evidence를 사용한다. trimmed W2의 장점과 감소 조건별 손실을 전체 fitted-fold 표에서 함께 확인해야 한다.

19. **e-process 결과**

E1_SCALE_MIXTURE_EVIDENCE Δ 0.0000145112; E2_SPARSE_START_MIXTURE Δ 0.0003851381. E1/E2는 누적·sparse-start likelihood evidence를 ranking 특성으로 사용한다. E1은 사전등록된 unreset 누적형이며, 최근 age 1–256 제한은 별도 E2에만 적용했다. 둘은 각각 학습한 후보여서 age 제한만의 인과적 ablation은 아니다. 종속성과 H nuisance 추정 때문에 엄밀한 e-process 또는 anytime-valid alpha를 주장하지 않는다. 원래 4개 long-null 코호트에서 누적 Gaussian LR를 clipped 혁신 에너지 합으로 재현했다. 최대 산술 오차는 1.1823431123048067e-10다. 양의 working likelihood 비율은 학습 모델의 알람 확률이 아니며 원인의 기여를 분리한 결과로 해석하지 않는다.

20. **mean specialist 결과**

M1_ROBUST_LOCATION_LIKELIHOOD Δ 0.0004514076; M2 screen Δ -0.0029645427, standalone 0.4803233164로 탈락, full 미실행. M1은 H 제외 평균 Δ -0.0010261474, 조건부 CI [-0.0020377821,-0.0000275769]로 일반화 약점을 보였다.

21. **new independent expert 여부**

확증된 새 독립 expert 없음. 같은 fold·time 안의 상대 순위를 모은 상관에서 X2 expert–BL80은 0.6748937188, X2 composite–BL80은 0.9964253232다. M2는 screen 상관 -0.0264847268이어도 성능이 나빠 탈락했다. X 제품항과 parent 결합의 기여를 분리한 ablation은 없으며, pair ordering·unrealized rescue는 배포 가능한 oracle 성능이 아니다.

22. **5-fold ResearchPool 결과**

X2 fold 0–4: 0.6299753794808978, 0.6128803572648576, 0.6230912200910537, 0.6185902011119535, 0.623267142496583. 대응 Δ: 0.0010587484998473284, -0.00012838742493948008, 0.002210418424208882, 0.0009707298976112666, 0.001866740699323799. median 0.6230912200910537, worst 0.6128803572648576. 중앙값 점수끼리의 차이는 +0.002210418424208882, 대응 fold 차이 중앙값은 +0.0010587484998473284다. 최저 점수끼리의 차이와 최소 matched-fold Δ도 별도 열로 보고하며, 원래 중앙값 차이 CI를 대응 차이 중앙값에 옮기지 않는다.

23. **ALT**

ALT_A Δ 0.0011933215, CI [0.0004819220, 0.0018488467], P=0.9995; ALT_B Δ 0.0010763823, CI [0.0004133746, 0.0017324193], P=1.0. 다른 seed나 split을 추가하지 않았고, ALT 양수가 PRIMARY 기준 실패를 취소하지 않는다.

24. **regime**

H_EXCLUSION Δ 0.0009105181, CI [-0.0001148495, 0.0021231183], P=0.9525. H classifier는 NEXT5 전에 전체 DEV H로 만든 고정 변환이라 역사적 transductive 요소가 있다. RP H descriptor를 새로 계산했으며 post-hoc mechanism label을 partition input으로 쓰지 않았다. 평균은 크기가 다른 5개 regime 차이의 동일 가중 평균이다. 학습 크기 2620–6267의 변화와 배경 변화가 함께 포함되며 pooled RP 점수나 순수 domain-shift 효과가 아니다.

25. **mechanism**

mechanism_SCALE_DOMINANT Δ 0.0132749763, CI [0.0077602447, 0.0196739409]; mechanism_DEPENDENCE_DOMINANT Δ 0.0007178307, CI [-0.0001762731, 0.0016043769]. SCALE positive support 106, MEAN 16, TAIL 64. MEAN/TAIL을 supported 확증으로 주장하지 않는다. Post-hoc signature는 실제 DGP 정답이나 inference feature가 아니다.

26. **time fracture**

time_50_99 Δ 0.0024333603, CI [0.0007399203, 0.0040359550]; time_100_199 Δ 0.0019085210, CI [0.0006466971, 0.0032149746]; time_400_plus Δ 0.0010705370, CI [0.0001935787, 0.0019508285]. 7개 고정 구간 전체와 pair-mass 가중 기여를 보고한다.

27. **break age**

age_0_4 Δ 0.0007962482, CI [-0.0002781994, 0.0018560747]; age_50_plus Δ 0.0011389738, CI [0.0004081296, 0.0019140378]. 각 age의 positive와 같은 시간의 negative를 비교하고, 이른 구간 개선만으로 전체 TS-AUC를 대체하지 않는다.

28. **repeated outlier**

P1_TWO_TIMESCALE_CONFIRMATION 평균Δ -0.0078301241 (fold 범위 -0.0116003125∼-0.0033247912); R2_CONDITIONAL_SCALE_PIT 평균Δ -0.0039242528 (fold 범위 -0.0077171162∼0.0006398401); X2_SCALE_TAIL 평균Δ -0.0031774143 (fold 범위 -0.0062683181∼0.0013733124). 선형 분해의 고정 파일 평균: P1 전체Δ -0.0078301241 = BL80 비중 감소 -0.0146128235 + expert 추가 응답 0.0067826994; R2 전체Δ -0.0039242528 = BL80 비중 감소 -0.0146128235 + expert 추가 응답 0.0106885707; X2 전체Δ -0.0031774143 = BL80 비중 감소 -0.0146128235 + expert 추가 응답 0.0114354092. 새 전체 RP 학습 파일의 결과가 아니며 진폭 감소만으로 회복을 주장하지 않는다. 기존 실제 D2 파일은 NEXT4 OB16에서 악화했던 한계를 유지한다.

29. **single outlier**

P1_TWO_TIMESCALE_CONFIRMATION 평균Δ -0.0112690434 (fold 범위 -0.0131280559∼-0.0094337283); R2_CONDITIONAL_SCALE_PIT 평균Δ -0.0108864050 (fold 범위 -0.0127636862∼-0.0090324042); X2_SCALE_TAIL 평균Δ -0.0093486639 (fold 범위 -0.0113412580∼-0.0075723834). 선형 분해의 고정 파일 평균: P1 전체Δ -0.0112690434 = BL80 비중 감소 -0.0120497708 + expert 추가 응답 0.0007807275; R2 전체Δ -0.0108864050 = BL80 비중 감소 -0.0120497708 + expert 추가 응답 0.0011633658; X2 전체Δ -0.0093486639 = BL80 비중 감소 -0.0120497708 + expert 추가 응답 0.0027011069. lag 0–512, 1024/2048 horizon과 복구 censoring을 별도 기록했다. 첫 16연속 tolerance 충족은 영구 복구 보장이 아니다.

30. **long-null**

X2 alpha=0.05, horizon 16384의 fold 0–4 후보/BL80 eventual alarm: N0: 0.33203125/0.3203125, 0.3515625/0.3515625, 0.34765625/0.35546875, 0.359375/0.37890625, 0.3046875/0.34375; N1: 0.3984375/0.40234375, 0.40625/0.42578125, 0.50390625/0.4453125, 0.38671875/0.36328125, 0.453125/0.453125; N2: 0.4609375/0.4609375, 0.43359375/0.43359375, 0.4453125/0.44921875, 0.48046875/0.484375, 0.4375/0.4296875; N3: 0.41796875/0.4140625, 0.46875/0.48828125, 0.421875/0.45703125, 0.40625/0.39453125, 0.45703125/0.48828125. 임계값은 독립 1024 calibration에서 고정했다. 16384에 같은 alpha 보장을 붙이지 않는다.

31. **bootstrap**

X2 Δ CI [0.0005291145, 0.0018663422], P(Δ>0)=1.0. 원래 고정한 2000 whole-series draw만 사용했다. ALT/H는 PRIMARY-stratified multiplicity를 ID로 제한했으며 새 draw나 calibration/learner uncertainty를 포함하지 않는다.

32. **winner gap**

X2−X1 +0.00010260464973255168, CI [-0.00010943783344274093,+0.000317392683116807], P=0.837. 기준 0.0007 미만이므로 TIE/UNCERTAIN이다.

33. **family multiplicity**

모든 family와 best-of-N 결과를 남겼다. 공통 draw의 centered max statistic은 고정 OOF 후보군에 대한 조건부 진단이다. feature 설계, screen, 전체 RP를 본 X parent 선택과 과거 DEV 탐색까지 교정하는 FWER 보장이 아니다. X2 bootstrap winner 빈도 0.6515도 진짜 최선일 확률이 아니다.

34. **shadow confirmation**

NOT_EVALUATED_NO_ELIGIBLE_CANDIDATE. Shadow에서 성공하거나 실패한 것으로 해석하지 않는다. 부적격 후보를 구제하기 위해 Shadow를 소비하지 않았다.

35. **latency**

FROZEN_PRIMARY: point median/p95/p99 9.200/10.600/12.400 μs, H 초기화 median 0.004676초, 실제 10M 94.919초; X2_SCALE_TAIL: point median/p95/p99 17.300/19.600/22.500 μs, H 초기화 median 0.009331초, 실제 10M 172.907초. 각 수치는 3개 고정 순서 블록 통계의 중앙값이다. S4B_UNIFIED_SCALE_SPECIALIST median 1.696배, 10M 1.634배, 사전 고정 median/amortized/10M의 3배 기준 True; P1_TWO_TIMESCALE_CONFIRMATION median 1.511배, 10M 1.478배, 사전 고정 median/amortized/10M의 3배 기준 True; W2_TRIMMED_RECENT_ENERGY median 1.533배, 10M 1.516배, 사전 고정 median/amortized/10M의 3배 기준 True; E2_SPARSE_START_MIXTURE median 1.957배, 10M 1.924배, 사전 고정 median/amortized/10M의 3배 기준 True; R2_CONDITIONAL_SCALE_PIT median 1.543배, 10M 1.500배, 사전 고정 median/amortized/10M의 3배 기준 True; M1_ROBUST_LOCATION_LIKELIHOOD median 1.522배, 10M 1.471배, 사전 고정 median/amortized/10M의 3배 기준 True; X2_SCALE_TAIL median 1.880배, 10M 1.814배, 사전 고정 median/amortized/10M의 3배 기준 True. 비율은 원래 BL80 대비다. 9개 artifact × 3블록 × 실제 10M=270M 연속 업데이트를 측정했다. 반복 buffer 점수로 quality를 평가하지 않았다. 원래 D2 latency 수치는 역사적 NEXT4 측정과 구별한다.

36. **state memory**

FROZEN_PRIMARY: 10M mutable state 16925 bytes, immutable tree 169000 bytes, 실제 H별 state 16925∼16925 bytes; X2_SCALE_TAIL: 10M mutable state 93404 bytes, immutable tree 253500 bytes, 실제 H별 state 57644∼119676 bytes. 10M 연속 업데이트에서 state 배열 크기가 증가하지 않는지 확인했다. LATENCY_NEXT5.json에 RSS도 별도 기록하며 Python 전체 객체 크기와 숫자 배열 bytes를 혼동하지 않는다.

37. **leakage**

RP admission-before-read, grouped train/held-out 분리, train-only column/weight, raw/lock SHA와 negative control을 확인했다. tau/ID/final length/future suffix는 feature API에 전달하지 않았다. 애플리케이션 guard 검증이며 OS 수준 파일 sandbox의 증명은 아니다.

38. **prefix invariance**

각 새 family에서 100개 이상 random prefix 검증. 대표 7개 실제 infer API에서는 각각 300개 이상 suffix/length 변이 검증과 모든 RP OOF point parity를 수행했다. 이전 역할 BL80/D2의 실제 entry와 reverse-series reset도 별도 검증했다.

39. **determinism**

동일 artifact의 반복 실행 fingerprint, 저장/로드와 native/compact 예측, 실제 current-point API의 OOF parity를 검증했다. 정확 일치 여부와 최대 오차는 DETERMINISM_NEXT5.json의 원 영수증을 따른다. 의미적 예측 일치와 joblib 바이트 일치를 구별한다.

40. **reproduction**

새 NEXT5 role이 없어 §87의 새 raw→fit 재현은 NOT_REQUIRED다. 기존 BL80/D2의 RP offline/stream parity와 BL80/B2/D2의 새 프로세스·별도 복사 payload 합성 fingerprint를 확인했다. 최종 전체 test suite 1363개 통과, failure/error/skip 0. 이것을 새 raw 재학습으로 부르지 않는다.

41. **DEV saturation risk**

HIGH. 이번 유한 메뉴는 종료했다. 12개 이상 가설·best gain<0.0007인 형식적 Case H는 해당하지 않지만, 과거부터 반복한 DEV 선택과 작은 winner gap의 위험이 사라지는 것은 아니다. X parent 선택은 전체 RP 점수 기반이며 nested 선택이 아니다.

42. **shadow contamination risk**

NEXT5의 반환 행·캐시·훈련·평가·특징 분석 기준 Shadow 사용 0이다. 그러나 이 1600개 시계열은 과거 전체 DEV 연구에 포함됐다. 완전히 새롭고 독립적인 holdout이라고 주장하지 않는다. Arrow 내부 page/row-group 디코딩 경로 전체를 계측한 증거는 없다.

43. **A/B/C/D seal opened?**

아니오. 500개씩 총 2000개를 모델 평가·훈련·분석에 사용하지 않았고 guard는 전부 미개봉이다. ID 검증과 raw byte hashing은 관측/label 분석과 구별했다. 내부 Parquet 디코딩까지 0이었다는 독립 증명은 아니다.

44. **reduced usage**

NEXT5 새 사용 0. 과거 누적 사용 2/2와 원래 usage ledger는 변경하지 않았다.

45. **public usage**

NEXT5 public submission 0, cloud submission 0. 외부 제출을 실행하지 않았다.

46. **seal open recommendation**

NO. 새 PRIMARY 기준 통과나 Shadow 확인이 없고 반복 DEV 선택 위험이 크다. LOCAL SEAL을 열지 않았다.

47. **다음 병목 3개**

① H/ARCH/heavy-tail 조건이 달라져도 유지되는 분산 감소 신호. ② 반복 이상치의 장기 영향과 late genuine break 검출을 함께 다루는 상태. ③ 제한된 기존 DEV에서 독립 신호의 이득을 확인할 검증력. 다음 세션에도 과거 Shadow 이력을 지우거나 이번 실패의 seed/weight 미세 탐색을 재개하지 않는다.

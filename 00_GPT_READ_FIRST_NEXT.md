# Structural Break NEXT — 먼저 읽기

1. 실제 연구시간: 10.2511시간. 재부팅·중단 공백 2.1481시간은 제외했다. 별도 최종 감사 시간은 final_audit/RESULTS.json과 ZIP 검증 영수증에 기록한다. 시계 복구와 제외 근거는 scheduler/recovery_1 및 SUBSTANTIVE_RESEARCH_DURATION.json에 보존했다.

2. NEXT 실험 수: 142개 후보. 상태별 {'SCREEN_FAIL': 104, 'ROBUSTNESS_PENDING': 16, 'FULL_FAIL': 13, 'NEW_GENERATION': 8, 'BUG': 1}. 재현·보조진단 fit 수는 후보 수와 구별한다.

3. 기존 PRIMARY 평균: 0.611292686371.

4. 새 PRIMARY: J102_SERIAL_VARIANCE_512, 평균 0.621631750312.

5. 기존 대비 평균 Δ: +0.010339063941.

6. 고정 원본 분할에서 새 PRIMARY의 5-fold 점수: [0.625548994582903, 0.6353471840978637, 0.6214373863953904, 0.6156787076692104, 0.6101464788170321].

7. 새 PRIMARY fold 중앙값: 0.621437386395.

8. 새 PRIMARY 최저 fold: 0.610146478817.

9. paired whole-series bootstrap 평균 Δ 95% CI: [+0.005022829, +0.015520130], 2000회. 선택 이후 조건부 구간이며 다중 탐색 보정 구간이 아니다.

10. P(Δ>0): 1.0000.

11. 고정 대체 split 3개의 평균 Δ: [0.010445282935763633, 0.010369884291587694, 0.007759749751261813]. 각 split은 5개 fold를 새로 학습했다.

12. H-regime 전체 제외 후 재학습한 최저 점수: 0.599216782044. 원본 OOF의 regime별 절단 통계와 다르다.

13. Mechanism별 기존 대비 결과: MEAN_DOMINANT: Δ=-0.001273, CI=[-0.051439,+0.025876], n=21, supported=False; SCALE_DOMINANT: Δ=+0.028903, CI=[+0.004604,+0.051743], n=132, supported=True; DEPENDENCE_DOMINANT: Δ=+0.008332, CI=[+0.001280,+0.015964], n=1525, supported=True; TAIL_DOMINANT: Δ=+0.019772, CI=[+0.005891,+0.034209], n=86, supported=False; MIXED: Δ=+0.039370, CI=[+0.020858,+0.058984], n=210, supported=True; WEAK_OR_AMBIGUOUS: Δ=+0.008151, CI=[+0.000202,+0.015668], n=1970, supported=True

14. Whitening: H 내부 forward holdout에서 잔차 |ACF1| 평균은 AR1 0.07622, AR4 0.04801, AR8 0.04616, AR12 0.04620. 더 복잡한 whitening 자체가 DEV 개선을 보장하지 않았다.

15. ECDF Gaussianization: H 잔차 kurtosis 중앙값 3.1753→2.9568로 정규 모양에 가까워졌지만 독립적인 성능 개선 근거는 부족하다. NORM 실험과 H 진단은 별도로 보존했다.

16. GLR: 나이 후보를 128→512로 넓힌 Y001−V007 평균 Δ +0.0023342, 조건부 CI [+0.0004450,+0.0042894]. B104는 explicit GLR와 GLR-memory 10개 열을 제거하고 local+Bayes 16개 새 열을 유지했다. 제거 효과의 CI는 0을 포함하므로 GLR 자체의 필요성을 단정하지 않는다.

17. Bayes-lite: 긴 age 조건의 B102(Bayes 제거)−Y001 평균 Δ −0.0027746, 조건부 CI [−0.0047041,−0.0008476]. 짧은 age의 제거 효과는 불분명했다. 정확한 사후확률 모델이라고 주장하지 않는다.

18. Ranking: O104 rank objective 평균 0.6182062로 이진 목적의 대응 V007 0.6192973보다 낮았다. W0/W1/W2도 W3를 일관되게 대체하지 못했다. 세부 실패와 screen 통과 여부는 RANKING_OBJECTIVE_RESULTS.csv 및 WEIGHTING_RESULTS_NEXT.csv.

19. 가장 강한 새 아이디어: 변동성으로 나눈 잔차만 사용하지 않고 원시 innovation energy와 변동성 예측 상태의 변화 증거를 함께 보존한 것. 장기 age 추가가 후속 개선을 보였다.

20. 주요 실패/한계: 단독 sequential 통계, 무분별한 Gaussianization·ARCH 복잡도, 25-tree 축소, 일부 weighting/ranking, forecast-only의 최저 fold 저하. J102−Y001은 +0.0000002804, CI [−0.0013714,+0.0012768], P=0.5015로 자기상관 보정의 뚜렷한 이득이 아니다. 합성 stress에서도 tail/ARCH 반례가 있다. 1,024-point에서 정한 5% 임계값의 16,384-point 무변화 오경보 범위(5개 합성 regime): PRIMARY: 14.7–78.2%; V007_VARIANCE_FORECAST: 33.3–46.2%; Q005_VARIANCE_NO_MEMORY: 40.6–49.3%; Y001_LONG_VARIANCE_ARCH1: 34.5–50.5%. 이 실험은 고정된 CV fold-0 모델만 사용했으며 최종 all-DEV 모델의 측정값이 아니다. 합성 AR0.3/ARCH0.65 Gaussian의 scale 감소 조건에서 V007−기존 HIGHEST_MEAN은 -0.238058, 조건부 CI [-0.252767,-0.223654]였다. 20개 방향별 비교 전체는 보조 진단 문서에 있다. 이 CV fold-0 반례를 최종 J102/B104 all-DEV 모델의 직접 측정으로 해석하지 않는다.

21. Latency: 점당 median 7.700 μs; H 초기화 포함 28.394 μs/point. 전체 원자료는 LATENCY_NEXT.json.

22. SEAL A/B/C/D opened: false / false / false / false. 총 2,000개 sealed ID는 계속 미개봉이다.

23. Reduced 신규 사용: 0회. 과거 사용 2/2는 그대로 유지했다.

24. Public leaderboard 신규 제출: 0회. SDK test 호출 및 실제 reduced 평가도 하지 않았다.

25. 다음 세션 seal 논의 가능 여부: frozen rule상 True. 이는 이번 세션의 개봉 허가가 아니며 실제 seal은 0회 사용했다. 적응적 DEV 선택 편향과 장기 null alarm을 함께 검토해야 한다.

26. 다음 우선순위 3개: (1) 최종 감사·장기 null horizon 결과와 한계를 외부 GPT가 재검토, (2) 기본 모델 및 비용을 고정하고 재학습 없는 독립 평가 설계 논의, (3) 명시적 다음 세션 승인 이후에만 seal A 단일 사전등록 평가를 검토.

| 역할 | 후보 | 평균 | 중앙값 | 최저 fold |
| --- | --- | --- | --- | --- |
| PRIMARY_DEV_CANDIDATE | J102_SERIAL_VARIANCE_512 | 0.62163175031248 | 0.6214373863953904 | 0.6101464788170321 |
| ROBUST_DEV_CANDIDATE | B104_LONG_VARIANCE_NO_GLR_MEMORY | 0.6214890347069302 | 0.6201193885482584 | 0.6114405720853687 |
| FAST_DEV_CANDIDATE | F101_VARIANCE_TREES50 | 0.6171824646565767 | 0.6178571134418307 | 0.6072673445305061 |

[성능·속도 비교](artifacts/next/figures/actual_artifact_tradeoffs_v2/actual_tradeoffs.png) · [고정 변경의 paired 비교](artifacts/next/figures/fixed_interventions_v2/paired_interventions.png) · [장기 오경보: 5% 기준](artifacts/next/figures/long_null_v2/null_horizon_alpha_5.png)

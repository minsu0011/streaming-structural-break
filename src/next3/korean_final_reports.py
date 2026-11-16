"""Human-readable report templates; all completed numeric evidence is supplied."""


def duration(seconds):
    seconds = float(seconds)
    return f'{seconds:.3f}초 ({int(seconds//3600)}시간 {int(seconds%3600//60)}분 {seconds%60:.3f}초)'


def read_first(f, run):
    p = f['performance']; d = f['distilled']; score = p['full']
    old_pair = p['paired_all_time']; m = f['maintenance']
    cold = f['latency']['PERFORMANCE']['cold_process']
    speed = f['latency']['PERFORMANCE']; state = f['memory']['PERFORMANCE']
    dep = f['inherited_mechanism']['DEPENDENCE_DOMINANT']; scale = f['inherited_mechanism']['SCALE_DOMINANT']
    mass_dep = f['pair_mass_contributions']['DEPENDENCE_DOMINANT']
    mass_scale = f['pair_mass_contributions']['SCALE_DOMINANT']
    r1 = f['recipe_pairs']['R1_SCORE_GAP']; r3 = f['recipe_pairs']['R3_DEPTH2_GATE']
    ewma = f['legacy_EWMA_state_transfer']['127']
    preview = run['preview']
    label = '검토용 초안 — 최종 재현·감사·ZIP 완료 전' if preview else 'NEXT3_ENGINEERING_IMPROVED — 과학 모델 유지, 메모리 비용 개선'
    final_tests = ('최종 감사의 새 전체 테스트는 아직 실행하지 않았다. 기존 전체 실행과 후속 targeted 테스트의 통과 기록은 maintenance의 원래 XML/receipt로 확인한다. 실행 전의 예상 전체 테스트 수를 통과 수로 적지 않는다.'
        if preview else f"최종 감사에서 새 전체 테스트 **{run['tests']['tests']}개 통과**, 실패·오류·skip 0, 실행 시간 **{run['tests']['seconds']:.3f}초**다. XML과 검사 대상 source/config/test SHA를 별도 기록했다.")
    raw = ('최종 감사 시간에 새 환경을 만들어 raw DEV부터 학습하는 검증은 아직 실행하지 않았다. 이를 통과하기 전에는 최종 전달로 인정하지 않는다.'
        if preview else '별도의 새 환경에 63개 고정 패키지를 설치하고, 원본 bytes의 일반 복사본에서 DEV 8,000개를 다시 학습했다. distilled 1회와 performance 2회, 총 3회 재학습으로 canonical 모델 bytes와 역할별 4,014,405개 public API 예측 fingerprint가 일치했다. 과거 feature cache는 사용하지 않았다. all-DEV 예측은 품질 점수가 아닌 동일성 검증이다.')
    final_runtime = ('최종 재학습 모델로 수행할 public API 30M 검증은 아직 실행 전이다.' if preview else
        f"최종 감사에서는 재학습 DISTILLED·원래 참조 PERFORMANCE·선택 PERFORMANCE의 실제 public API를 각각 10M씩, 총 **{run['runtime']['total_actual_public_API_points']:,}점** 실행했다. 선택/참조 시간 비율은 **{run['runtime']['selected_performance_vs_reference_public_API_10M_ratio']:.9f}**다. 이 값은 매 yield마다 현재 점 접근을 확인하는 Python 검증 비용과 H 초기화를 포함하며, 아래의 direct kernel 측정과 경계가 다르다.")
    answers = [
        ('총 wall-clock', f"보고서 작성 시점 **{duration(run['wall_seconds'])}**. 시작은 `{run['started_utc']}`, 이 문서의 관측 시점은 `{run['observed_utc']}`다. 연구 단계 wall **{duration(run['research_wall_seconds'])}**, 별도 감사 wall **{duration(run['audit_wall_seconds'])}**. 최종 종료 시각은 `FINAL_AUDIT_NEXT3.json` 및 실제 ZIP 외부 검증 기록을 우선한다."),
        ('실제 productive research time', f"감독 프로세스가 활동 근거로 누적한 연구·유지보수 productive 시간은 **{duration(run['productive_research_seconds'])}**, 별도 감사 productive 시간은 **{duration(run['audit_productive_seconds'])}**다. 보수적으로 미관측 처리한 시간 **{duration(run['unobserved_seconds'])}**를 연구 시간에 합치지 않았다. 이 수치는 `{run['accounting_timestamp']}`의 관측 스냅샷이다. wall 시간과 실제 계산 CPU 시간도 서로 다르다."),
        ('실험 수', f"새 과학 recipe **13개: full 12개, screen 탈락 1개**. 기존 sign 실패 I1의 정확 재사용 1개와 실행 보류 S4 1개는 새 fit으로 세지 않았다. 과학 fit **129회**는 nested base 56 + 후보 learner 51 + 작은 router 14 + residual 7 + R3 all-DEV 1이다. 최종 raw 재현 fit은 **{run['raw_fits']}회** 별도이며 현재 총 **{129+run['raw_fits']}회**다. zero-fit 진단 9건(F001/F002/D001–D007), 실행 구현 계열 E001–E003 3개를 과학 후보 수에 더하지 않았다."),
        ('장애/재시작 수', f"실제 child 실패 **{m['failed_child_attempts']}건**, 초기 등록 버그 **{m['prelaunch_registration_bugs']}건**, 성공 의존성 이름 오류로 추가 사전 차단 **{m['additional_dependency_guard_prelaunch_rejections']}건**, supervisor 오류 **{m['supervisor_errors']}건**, 계획한 supervisor 교체 **{m['planned_supervisor_restarts']}회**다. 실행 없이 대체한 작업 **{m['superseded_without_execution']}건**은 실패 child와 구분한다. 원본 실패 source·정책·로그를 보존했고, 성공 기록으로 덮어쓰지 않았다."),
        ('시작 performance score', f"`{p['experiment_id']}`: 원래 group 5-fold mean **{score['mean']!r}**. 시작부터 J102/C205 고정 80/20 역할이었다."),
        ('종료 best performance score', f"실제 선택한 성능 역할은 **{score['mean']!r}**로 같다. 새 실험의 탐색상 최고 I2는 **{f['exploratory_best']['full_mean']}**였지만 고정 승격 기준 미달로 채택하지 않았다. 탐색상 최고와 배포 선택을 구분한다."),
        ('delta', f"**이번 세션 선택 모델 − 시작 performance = {f['selected_delta']!r}**. 기존 BL80 − J102의 **{old_pair['estimate_mean_delta']!r}**는 NEXT2에서 이미 얻은 역사적 차이이며 이번 세션 이득이 아니다."),
        ('generation gate 통과 여부', '**아니오.** 새 정보와 +0.0025 이상 등 고정 요건을 충족한 후보가 없다. +0.0035 강한 세대 교체도 없다. runtime의 예측 동일성과 메모리 감소를 TS-AUC 상승으로 해석하지 않는다.'),
        ('새 performance candidate', '**없음.** `BL2_C205_DEPENDENCE_80` 과학 역할을 유지한다. 선택 실행 구현은 `LEAN_PACKED_SHARED`이며, 역할 수를 늘리지 않는다.'),
        ('새 robustness candidate', '**없음.** 일부 SCALE와 이상치 진단이 좋아졌지만 전체·지원된 slice·ARCH 두 감소 gate를 함께 통과한 후보가 없다.'),
        ('distilled fallback', f"`{d['experiment_id']}`, mean **{d['full']['mean']!r}**. `next2_canonical_distilled_main.py`와 tree-used 구현을 유지한다. NEXT3 score 변화는 0이다."),
        ('현재 가장 큰 병목', '반복 사용한 DEV에서 작은 양의 차이가 독립 검증으로 이어지는지 알 수 없다는 B4가 가장 큰 판단 병목이다. 모델의 구체적 미해결 약점은 ARCH/tail scale-down과 이상치 이후 장기 증거 잔류다. 수치 재계산과 exact 재현만으로 일반화 불확실성을 없앨 수 없다.'),
        ('처음 예상한 병목과 실제 병목이 달랐는가', f"처음 상태는 B4 **{f['initial_bottleneck_statuses']['B4']}**, B1 **{f['initial_bottleneck_statuses']['B1']}**, B2 **{f['initial_bottleneck_statuses']['B2']}**, B8 **{f['initial_bottleneck_statuses']['B8']}**였다. 초기에 전체 이득과 SCALE 손실의 위치를 확인했고, 유한 메뉴 종료 후 실제 진행 병목은 B4의 선택 불확실성과 B8의 검증·배포 신뢰성으로 이동했다. 감소 신호의 완전한 부재라는 단순 설명은 맞지 않았고, 이상치 잔류에는 legacy EWMA 묶음의 영향이 직접 상태 대조에서 확인됐다."),
        ('DEPENDENCE gain', f"기존 BL80 − J102 supported DEPENDENCE slice 차이 **{dep['estimate_mean_delta']!r}**, 조건부 CI **[{dep['ci_2_5']!r}, {dep['ci_97_5']!r}]**, 변화 시계열 **{int(dep['break_series'])}개**다. 공식 pair 질량 비중 **{mass_dep['fraction_of_official_pair_mass']!r}**, 전체 점수에 대한 정확한 기여 차이 **{mass_dep['blend_minus_J102_contribution']!r}**는 별도 정의다. 이번 선택 역할의 추가 이득은 0이다."),
        ('SCALE loss', f"기존 BL80 − J102 supported SCALE slice 차이 **{scale['estimate_mean_delta']!r}**, CI **[{scale['ci_2_5']!r}, {scale['ci_97_5']!r}]**, 변화 **{int(scale['break_series'])}개**다. pair 질량 비중 **{mass_scale['fraction_of_official_pair_mass']!r}**, 전체 점수 기여 차이 **{mass_scale['blend_minus_J102_contribution']!r}**를 slice 차이와 합산하지 않는다. 관측상 감소 79/증가 53개는 각각 지원 기준 100개 미달이다."),
        ('router가 trade-off를 개선했는가', f"부분적으로만 개선했다. R1_SCORE_GAP의 전체 Δ **{r1['all_time']['estimate_mean_delta']!r}**, SCALE Δ **{r1['mechanism_SCALE_DOMINANT']['estimate_mean_delta']!r}**이나 ARCH V3/V4 개선 폭이 각각 0.01 미만이다. R3의 전체 Δ **{r3['all_time']['estimate_mean_delta']!r}**, SCALE Δ **{r3['mechanism_SCALE_DOMINANT']['estimate_mean_delta']!r}**, DEPENDENCE Δ **{r3['mechanism_DEPENDENCE_DOMINANT']['estimate_mean_delta']!r}**로 의존성 손실을 남겼다. R2/R3의 확률 손실 감소가 실제 TS-AUC 개선으로 이어지지 않았으며, 이것이 logistic loss의 모든 조건에서의 ranking 불일치 증명은 아니다."),
        ('scale-down downstream 원인', '고정 합성 1,312개 단계 집계와 4,096개 prefix 검증에서 raw/conditional current 및 signed Bayes에 감소 정보가 남아 있었다. 최종 트리 점수의 구분력은 약해 downstream 누적·선택·학습 결합을 후속 대상으로 좁혔다. 이 관찰만으로 단일 트리 feature나 특정 memory가 모든 scale-down 실패의 인과 원인이라고 확정하지 않는다. D005/D006의 상태 교체는 이상치 잔류에 대한 대조로, scale-down 원인과 자동 동일시하지 않는다.'),
        ('ARCH/tail 감소 문제 개선 여부', '배포 선택의 개선은 없다. S1은 screen +0.002610 뒤 full -0.000454 및 median -0.002663으로 탈락했고, S2는 screen -0.003843에서 중단했다. S3 음의 Bayes 추가의 전체 +0.000023과 SCALE 조건부 CI는 해결 근거가 부족했다. R1/R3의 부분 진단 이득을 고정 synthetic +0.01 gate 통과로 바꾸지 않았다.'),
        ('C205에서 실제 필요한 component', '원래 24개 입력은 lag 1/2/4/8 × EWMA8/32·CUSUM 양/음·Bayes 0.25/1이다. lag 1/2만, lag 4/8만, local만, Bayes만인 네 subset은 모두 전체 평균을 낮췄다. 현재 메뉴로 더 작은 채택 가능한 집합을 찾지 못했으며, 각 24개 입력이 모두 독립적으로 필수라는 증명은 아니다.'),
        ('새로운 complement 발견 여부', f"승격 가능한 새 complement는 없다. I2 H 고정 midrank 곱의 최고 탐색 Δ **{f['exploratory_best']['delta_vs_frozen_performance']}**는 고정 +0.0015 기준 미달이다. O1 strict nested 선형 residual도 전체 음의 차이였다. C205의 기존 보완 역할을 유지한다."),
        ('5-fold scores', f"선택 성능 역할: **`{score['fold_scores']}`**. 원래 고정 group folds이며 새로운 seed나 fold 선택으로 만든 최고값이 아니다."),
        ('median', f"**{score['median']!r}**. 시작 성능 역할 대비 변화 0."),
        ('worst fold', f"**{score['worst_fold']!r}**. 시작 성능 역할 대비 변화 0."),
        ('bootstrap CI', f"선택 − 시작의 CI는 **[0.0, 0.0]**인 해석적 동일성이다. 새 bootstrap draw 0개다. 별도로 기존 BL80 − J102의 2,000 whole-series draw 조건부 CI는 **[{old_pair['ci_2_5']!r}, {old_pair['ci_97_5']!r}]**다. D007의 다른 pair-matrix 알고리즘은 기존 fold score 90,045개와 대비 배열 24,012개를 정확히 재현했다. 수치 독립성을 새 통계적 독립 검증으로 해석하지 않는다."),
        ('P(delta>0)', f"선택 − 시작의 엄격한 **P(Δ>0)=0.0**이다. 동일 예측에서 Δ=0이므로 1이나 0.5로 적지 않았다. 기존 BL80 − J102의 bootstrap 양수 비율은 **{old_pair['probability_delta_positive']!r}**다. 어느 쪽도 hidden test 성공의 posterior 확률이 아니다."),
        ('alternate split 3개', f"기존 성능 역할의 평균 **`{[row['mean'] for row in p['alternate_splits']]}`**, 기존 J102 대비 Δ **`{[row['mean_delta'] for row in p['alternate_splits']]}`**. 세 split의 기존 fit을 상속했고 NEXT3에서 ALT 학습을 새로 하지 않았다. 새 runtime이 기존 held-out 출력과 정확히 같은지는 별도 전수 검사했다. 이번 선택 − 시작 Δ는 세 split 모두 0."),
        ('regime worst', f"기존 leave-one-H-regime-out worst **{p['heldout_H']['worst']!r}**, mean **{p['heldout_H']['mean']!r}**, J102 대비 최소 matched Δ **{p['heldout_H']['minimum_matched_fold_delta']!r}**. 원래 OOF를 H별로 자른 slice와 다른 평가다. NEXT3 추가 H fit 0회, runtime 출력 변화 0."),
        ('long-null 상태', f"미해결이다. 기존 N0/α=0.05 고정 threshold의 선택 성능 역할 eventual FPR은 horizon **`{[row['horizon'] for row in f['null_N0_alpha05']]}`**에서 **`{[row['eventual_alarm_rate'] for row in f['null_N0_alpha05']]}`**로 증가한다. 배경·horizon별 threshold와 진단을 구분한다. 무한 horizon false-alarm 보장은 없고, null-only TS-AUC는 N/A다. FPR만 낮추면서 늦은 변화 검출을 버리는 age-decay는 채택하지 않았다."),
        ('outlier persistence', f"선택 모델의 기존 O1 alarm excess **{f['inherited_outlier_excess'][p['experiment_id']]!r}**, J102 **{f['inherited_outlier_excess'][d['experiment_id']]!r}**로 약점이 남는다. D006에서 이상치 뒤 offset 127에 legacy EWMA 묶음을 matched control 상태로 교체하면 이후 [256,512) 평균 score 차이는 **{ewma['delta_from_destination']!r}**, CI **[{ewma['delta_from_destination_ci_low']!r}, {ewma['delta_from_destination_ci_high']!r}]**다. 실행하지 않은 reset 정책의 효과나 특정 EWMA160 cell 하나의 원인으로 일반화하지 않는다. 342쌍 late cohort는 512쌍과 분리했다."),
        ('runtime', f"E003의 warm point median 요약 **{speed['median_of_trial_summary_statistics']['point_distribution']['median_us']!r} µs**, H 초기화 median 요약 **{speed['median_of_trial_summary_statistics']['historical_initialization']['median_us']!r} µs**. 세 번의 per-trial 요약의 중앙값이며 pooled quantile이 아니다. E002 대비 fresh-process post-warm RSS 중앙값 비율 **{cold['ratio_of_median_postwarm_RSS_vs_E002']!r}**, 감소율 **{100*cold['relative_postwarm_RSS_reduction']:.6f}%**다. import→first-ready-None 중앙값 비율 **{cold['ratio_of_median_first_none_seconds_vs_E002']!r}**로 20% startup 개선 기준은 미달, memory 개선 경로를 통과했다."),
        ('state memory', f"선택 성능 runtime의 private per-series 수치 state **{state['private_per_series_numeric_bytes']:,} bytes**, shared selector **{state['shared_selector_bytes_reachable_from_state']:,} bytes**, model/prepared 공유 배열 **{state['model_and_prepared_shared_numeric_bytes']:,} bytes**. selector를 중복 세지 않은 one-stream+model 총 **{state['total_one_stream_plus_one_model_numeric_bytes']:,} bytes**다. canonical 파일은 **{state['canonical_model_file_bytes']:,} bytes**. 10M 후 retained 수치 state 증가 0. Python/JIT/allocator/RSS나 H 일시적 peak를 이 배열 크기와 혼동하지 않는다."),
        ('10M point 예상/실측', f"point median으로 외삽한 선택 성능 시간 **{speed['estimated_10M_seconds_from_point_median']!r}초**; 세 번 실제 direct 10M **`{speed['three_actual_10M_seconds']}`**, 중앙값 **{speed['median_actual_10M_seconds']!r}초**다. E002 대비 paired 시간 비율 중앙값 **{speed['median_direct_time_ratio_vs_E002']!r}**로 처리량 개선을 주장하지 않는다. distilled의 기존 직접 실측은 **{f['latency']['DISTILLED']['continuous_stream']['measured_prediction_seconds']!r}초**다. {final_runtime}"),
        ('tests', final_tests),
        ('reproducibility', raw+' 원래 프로젝트 접근·사전 expected weight 읽기·외부 파일 쓰기·network 시도를 별도로 검사한다. Python audit hook은 OS sandbox가 아니다. 실제 ZIP의 새 폴더 추론 재생은 ZIP을 만든 뒤 외부 receipt에서 최종 확인한다.'),
        ('SEAL_A opened?', '**false**, 500개 미개봉.'),
        ('SEAL_B opened?', '**false**, 500개 미개봉.'),
        ('SEAL_C opened?', '**false**, 500개 미개봉.'),
        ('SEAL_D opened?', '**false**, 500개 미개봉.'),
        ('reduced 신규 사용?', '**0회**. 기존 사용 2/2와 reduced ledger bytes를 보존했다.'),
        ('public submission?', '**0회**, cloud 신규 제출도 0회. SDK `.test`를 실행하지 않았다.'),
        ('지금 seal을 열어야 하는가?', '**아니오.** 새 과학 승격·generation 교체가 없고 적응적 DEV 불확실성이 남는다. 다음 독립 검증 계획부터 고정한다.'),
        ('public submission을 할 가치가 있는가?', '이번 연구 개선을 확인하기 위한 제출은 권고하지 않는다. 선택 예측이 기존과 같아 새 과학 증거를 기대할 근거가 없다. 향후 순수 운영 검증의 가치 판단과 실제 제출은 별도 결정이며 이번 세션에 실행하지 않았다.'),
        ('DEV research saturation 위험', '반복 탐색 위험은 높다. 다만 규칙의 연속 10개 screen 실패 조건은 충족하지 않아 `DEV_RESEARCH_SATURATION=NOT_FORMALLY_TRIGGERED`로 기록했다. 새 full 최고 Δ도 +0.0007 미만이며 유한 가설 계열 메뉴가 끝나 CASE G에 따라 검증·재현·비용 유지보수로 전환했다. 통과한 screen을 실패로 다시 세지 않았다. 12후보 공동 조건부 envelope도 선택 과정을 모두 보정한 CI가 아니다.'),
        ('다음 단계 3개', '(1) B4: 독립 평가 기회를 쓰기 전에 가설·최소 효과·오류 통제·중단 기준을 고정한다. (2) B2: ARCH/tail 감소 문제에서 지원된 real DEV 손실 없이 새 정보를 주는 소수 대조만 설계한다. (3) B6: legacy EWMA 등 잔류 memory의 후보 수정을 long-null·단일 이상치·늦은 변화의 clean detection과 함께 검증한다. 현재 frozen 모델과 모든 seal을 먼저 보존한다.')]
    assert len(answers) == 44
    heading = '# Structural Break NEXT3 — 연구 개요\n\n'
    lead = f'**{label}**\n\n기존 프로젝트를 이어 수행했다. 최종 과학 역할은 두 개이며, 새 성능·robustness 후보는 없다. 선택 성능 mean **{score["mean"]!r}**, 시작 대비 Δ **0.0**. 실행 구현의 메모리 비용만 개선했다. 네 seal 미개봉, reduced/public/cloud 신규 사용 0이다.\n\n'
    return heading+lead+'\n\n'.join(f'{number}. **{question}:** {answer}' for number, (question, answer) in enumerate(answers, 1))+'\n\n근거를 읽는 순서: `RESEARCH_REPORT_NEXT3.md` → 최종 결과표와 `DEV_CANDIDATES_NEXT3.json` → `FINAL_AUDIT_NEXT3.json` → ZIP 외부 검증 receipt. 이전 실험의 중간 상태 문자열은 기록 당시 상태이며 최종 선택을 뜻하지 않는다.\n'


def handoff(f, run):
    p = f['performance']; d = f['distilled']
    tag = '검토용 초안: 최종 raw 재현·전체 테스트·30M API·실제 ZIP 검증 전이다.' if run['preview'] else '최종 raw 재현·전체 테스트·30M API 증거가 확보됐다. 최종 core 감사와 실제 ZIP 재생의 완료는 각각 별도 receipt를 확인한다.'
    return f'''# Structural Break NEXT3 연구 인계

{tag}

이번 세션은 예측 성능을 올린 세션으로 보고하지 않는다. 새 과학 recipe 13개와 기존 실패 재사용 1개를 검토했지만 새 성능·robustness 승격은 없었다. 고정된 두 과학 역할을 보존하고, 성능 역할의 실행 의존성과 준비 배열을 줄였다. 판단 범주는 `NEXT3_ENGINEERING_IMPROVED`이며 hidden test 개선을 뜻하지 않는다.

| 역할 | 과학 모델 | 실행 entry | 원래 DEV mean | NEXT3 quality Δ |
| --- | --- | --- | ---: | ---: |
| DISTILLED | {d['experiment_id']} | next2_canonical_distilled_main.py | {d['full']['mean']!r} | 0.0 |
| PERFORMANCE | {p['experiment_id']} | next3_lean_performance_main.py | {p['full']['mean']!r} | 0.0 |

실제 전달 모델 디렉터리는 `DEPLOYMENT_ENTRY_POINTS_NEXT3.json`의 `entries`를 따른다. 최종 raw 재현으로 만든 두 모델의 canonical SHA는 각각 `{d['canonical_model_sha256']}`, `{p['canonical_model_sha256']}`와 같아야 한다. 기존 J102 joblib과 NEXT2 모델 디렉터리는 그대로 보존한다. E003의 원래 `PROVISIONAL_ENGINEERING_ELIGIBLE` 설정 파일도 재현에 쓰인 bytes를 보존하며, 최종 전달 경로의 authority는 root 선택·감사·실제 archive receipt다.

연구 결론은 세 가지다. 첫째, 라우터로 SCALE 일부 손실을 회복할 수 있었지만 고정 real DEV/ARCH 기준을 함께 넘지 못했다. R2/R3의 검증 W3 확률 손실 감소와 TS-AUC 하락은 유한 후보에서 목적의 순서가 다를 수 있음을 보였다. 이를 logistic loss가 모든 분포·모델 클래스에서 ranking에 부적합하다는 정리로 일반화하지 않는다. 둘째, C205의 네 축소 subset과 strict nested residual은 현재 80/20 혼합을 대체하지 못했고, I2의 작은 이득은 효과 기준에 미달했다. 셋째, 이상치 뒤 legacy EWMA 묶음과 serial evidence의 잔류 영향은 직접 상태 대조에서 확인했으나 새 reset 규칙을 학습·검증·채택하지 않았다.

학습·통계 경계는 다음과 같다. primary metric은 공식 TS-AUC다. R2/R3/O1의 meta 입력은 각 outer train 안의 inner OOF에서 만들었고, outer validation을 이용해 같은 행의 계수를 맞추지 않았다. 새 과학 fit은 129회다. 원래 ALT/H fit은 새로운 독립 검증으로 세지 않았고, runtime 검증은 고정 예측과 저장된 좌표가 일치하는지만 봤다. D007은 다른 pair-matrix 표현으로 원래 bootstrap 배열을 재현했지만 적응적 선택·model fitting 불확실성을 보정하지 않았다. 합성 threshold·first alarm·지연은 보조 진단이며 TS-AUC와 교환 가능한 점수가 아니다.

등록·학습 시각도 별도 검사했다. 기존 메타데이터의 172개 UTC 연결은 25개 후보 단계와 연구 학습 129회를 포괄하며 기록된 순서 역전은 없었다. 폴드별 완료 시각 86개와 nested 정책 생성 시각 2개가 없는 점은 명시했고, 외부 시각 공증으로 해석하지 않는다. 상세 범위는 `docs/NEXT3_REGISTRATION_CHRONOLOGY_AUDIT.md`와 해당 원본 receipt를 따른다.

실행 구현은 세 계열만 평가했다. E001 precompute는 사전 10% 시간 절감 기준 미달이었다. E002 packed tree는 prepared 수치 배열을 줄였고, packed+precompute의 마지막 tie-break 비율은 0.950224로 고정 0.95 경계에 못 미쳤다. 반올림해서 채택하지 않았다. E003 lean은 inference 경로에서 학습·표 처리 dependency를 지연하고 canonical 내용을 classless 배열로 준비했다. fresh-process post-warm RSS 중앙값 약 26.35% 감소를 확인했으며 import→first-ready-None 약 5.60% 감소는 20% startup 경로 미달이다. 세 paired direct 10M 시간 비율 중앙값 1.000074는 허용 범위 내 동일 비용 근거이며 throughput 향상을 주장하지 않는다.

## 검토 및 재생

먼저 `RESEARCH_OVERVIEW_NEXT3.md`의 44개 답을 읽는다. 전체 과학 후보와 실패 이유는 `MODEL_RESULTS_NEXT3.csv`, 고정 fold는 `CV_RESULTS_NEXT3.csv`, 조건부 대비는 `PAIRED_BOOTSTRAP_NEXT3.csv`에 있다. `record_type`·`validation_kind`·`evidence_origin`을 함께 확인한다. 새 13개 recipe, 기존 frozen 2역할, I1 재사용, S4 보류가 17개 행으로 구분된다. 누적 ledger는 원래 213개 행의 bytes를 prefix로 보존하고 29개 기록을 추가한다. 129회 과학 fit과 별도 raw 재학습 3회는 다른 회계다.

압축 해제한 디렉터리에서 Windows Python 3.13.2와 고정 의존성을 사용할 때:

```powershell
py -3.13 -m venv .review-venv
./.review-venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.review-venv/Scripts/python.exe -m pip check
./.review-venv/Scripts/python.exe -I scripts/replay_next3_portable.py --output review_next3_replay.json
```

이 replay는 포함된 두 모델로 18개 고정 합성 사례를 정방향·역방향 순서로 실행해 총 98,372개 출력을 비교한다. raw competition 입력·정답·학습·새 quality score가 필요 없다. 실행 결과는 새 파일에만 쓴다. 이 명령의 검토용 재생과, 최종 바탕화면 ZIP bytes에서 새 일반 폴더로 풀어 원본 프로젝트 접근을 차단한 실제 archive 감사는 별개다. 실제 archive 감사는 `artifacts/next3/handoff/DESKTOP_ZIP_VERIFICATION_NEXT3.json` 및 `actual_archive_replay/RESULTS.json`을 기준으로 한다.

추가로 원래 N0–N3 및 O1 합성 스트림 전체를 확인하려면 같은 환경에서 다음 선택 명령을 실행한다.

```powershell
./.review-venv/Scripts/python.exe -I scripts/replay_next3_long_horizon.py --output review_next3_long_horizon.json
```

이 명령은 PERFORMANCE만 3,072개 스트림·34,603,008점에서 검증한다. 합성 H/O를 다시 생성하고 작은 per-series CSV의 입력·출력 SHA와 비교하므로 저장된 point 예측 배열이 필요 없다. 일반 복사본에서 명령을 실제 실행해 통과했으며, 당시 249개 파일과 불러온 프로젝트 모듈 32개의 출처를 확인했다. 그 실행은 기존 환경을 사용한 유지보수 점검이다. 기본 98,372점 재생·최종 새 환경 raw 재학습·실제 ZIP bytes 감사와 별개의 증거이며 새 FPR·TS-AUC나 속도 benchmark가 아니다. 상세 범위는 `docs/NEXT3_OPTIONAL_LONG_HORIZON_REPLAY.md`에 있다.

raw부터 다시 학습할 때 필요한 원본 파일·SHA는 `RAW_DATA_REQUIRED_NEXT3.json`에 있다. competition 데이터는 ZIP에서 제외했다. 각 entry의 `train`은 frozen DEV 경계를 유지해야 한다. 이번 최종 감사의 실행은 `prepare_next3_independent_reproduction.py`, `run_next3_independent_reproduction.py`, `check_next3_independent_training.py`에 보존한다. 이들 session 감사 driver는 기존 결과를 덮어쓰지 않는 시간/경로 guard가 있어 무조건 재실행하는 명령이 아니다. 새 학습 검증에는 새 출력 workspace와 frozen split/guard가 필요하다.

그림은 `artifacts/next3/figures/` 아래 과학 메뉴 3쌍, coarse 이상치 상태 교체 3쌍, fine legacy 교체 1쌍, engineering 비용 1쌍으로 PNG/PDF 총 8쌍이다. PNG는 실제 표시해서 검토했다. 재생에는 별도 `requirements-next2-figures-lock.txt`의 13개 고정 패키지를 사용한다. source receipt와 CSV를 그림의 authority로 두고, plot을 새 통계 검증으로 세지 않는다.

기존 12개 후보의 조건부 bootstrap 집계도 원본 데이터와 모델 없이 재확인할 수 있다.

```powershell
./.review-venv/Scripts/python.exe -I scripts/replay_next3_conditional_statistics.py --output review_next3_conditional_statistics.json
```

이 명령은 이미 저장한 2,000개 draw와 관측 행에서 169개 수치를 대조한다. 15개 파일만 넣은 일반 복사본에서 실제 실행해 통과했다. 새 draw·fit·품질 평가나 선택 보정은 수행하지 않으며, 자세한 범위는 `docs/NEXT3_CONDITIONAL_STATISTICS_REPLAY.md`를 따른다.

완료 근거 25개의 연결과 과학 모델·runtime 표현의 관계는 `docs/NEXT3_EVIDENCE_NAVIGATION.md`에서 확인할 수 있다. 실제 트리의 5,800개 노드와 38,732개 인공 margin 검증은 `docs/NEXT3_TREE_STRUCTURAL_AUDIT.md`에 있다. 각 기록의 원래 상태와 해석 범위를 함께 적었으며 최종 배포 authority는 아래 외부 검증 순서를 따른다.

## 감사 범위와 보존 기록

`FINAL_AUDIT_NEXT3.json`은 source·원래 보호 파일·seal 상태·고정 gate·전체 테스트·raw 재학습·최종 runtime·수치 주장·전달 파일의 core 감사를 묶는다. `SOURCE_MANIFEST_NEXT3.json`은 packaging 시 확정하며, 실제 ZIP SHA/모든 member CRC·SHA/새 추론 재생은 ZIP 생성 후 외부 기록에 둔다. ZIP 내부에 자기 자신의 최종 hash를 재귀적으로 기록하지 않는다. 바탕화면의 정확히 20개 핵심 파일 폴더에는 19개 핵심 보고서와 실제 ZIP 외부 검증 receipt를 넣는다.

사용자 원문의 0–80번 절과 당시 증거 37개 사본을 연결한 중간 교차표는 `docs/NEXT3_REQUIREMENTS_TRACEABILITY_PREAUDIT.md`에 있다. 이는 최종 실행 전에 남은 작업과 조건부 미실행을 점검한 기록이다. 새 generation 후보가 없어 learning curve fit을 추가하지 않았고, 선택적 label-permutation/time-shift learner fit도 수행하지 않았음을 명시했다. 이후 최종 실행 여부는 root 및 외부 완료 증거를 따른다.

압축 해제 위치는 짧은 일반 폴더를 사용한다. 실제 Windows 복사에서 긴 중간 경로가 실패해 내부 staging 이름을 줄였고, 파일·디렉터리의 UTF-16 경로 길이를 복사 전에 검사한다. 보존된 기록의 member 이름과 byte는 유지했다. 이 검증은 임의로 깊은 압축 해제 경로까지 지원한다는 보장이 아니다. 당시 13,260개 파일과 실행 입력 904개를 확인한 유지보수 점검의 범위와 두 실패는 `docs/NEXT3_WINDOWS_PAYLOAD_REPAIR.md`에 있다.

같은 agent가 연구·구현·별도 검증 역할을 순환했다. 다른 agent나 독립된 인간의 심사를 받았다고 주장하지 않는다. Python audit hook·SealGuard는 application 경계이며 OS sandbox가 아니다. Windows의 고정 dependency와 검사한 스트림에 대한 exact 결과다. Linux/cloud/hidden test 성능은 검증하지 않았다. 자세한 실패 보존 및 한계는 `LIMITATIONS_NEXT3.md`, 다음 설계는 `NEXT_SESSION_PLAN_NEXT3.md`를 읽는다.
'''


def supporting_reports(f, run):
    limitations = '''# NEXT3 한계와 보존한 실패

- 반복 DEV 선택이 가장 큰 통계적 제약이다. 새 13개 recipe와 이전 세션의 탐색이 누적됐다. 개별 bootstrap과 12개 후보 공동 조건부 envelope는 고정 fitted OOF 조건의 수치이며, model fitting·적응적 선택·미래 검증의 무조건 coverage를 보장하지 않는다. D007의 다른 계산 알고리즘 exact 재현은 일반화 검증과 다르다.
- 선택한 과학 모델의 시작 대비 TS-AUC 변화는 0이다. 탐색상 최고 I2의 +0.0005699067486600873은 사전 성능 기준에 미달한다. performance·robustness·generation gate를 완화하지 않았다. formal saturation의 연속 10개 failed screen 조건은 실제로 충족하지 않았다.
- SCALE 전체의 변화 132개는 지원 기준을 충족하나 관측상 감소 79개/증가 53개는 각각 미달이다. 관측 창의 SD 방향을 DGP의 잠재 혁신 분산 배율과 동일시하지 않는다. MEAN 21개와 TAIL 86개도 미지원이다.
- R1은 일부 SCALE와 이상치 지표를 개선했으나 고정 ARCH V3/V4 증가 폭이 각각 0.01에 못 미쳤다. R3는 SCALE 회복과 DEPENDENCE 손실을 맞바꿨다. S1/S2/S3와 C205 축소·rank·residual 메뉴를 끝냈으나 모든 가능한 감소·의존성 방법을 배제한 것은 아니다.
- R2/R3의 W3 확률 손실 감소와 TS-AUC 하락은 유한 candidate/fit에서 관찰한 목적 순서 차이다. 양의 weighting과 충분한 함수 클래스 아래의 logistic/proper loss ranking consistency와 모순된다는 주장을 하지 않는다. 새 pairwise objective의 성공이나 새 정보 추가도 입증하지 않았다. `docs/NEXT3_RANKING_SURROGATE_METHODOLOGY_V2.md`가 수정된 수식 표기 authority다.
- D005/D006은 원래 동일 DGP 쌍의 recurrent state를 교체한 조건부 합성 대조다. hybrid는 실제 입력 이력으로 도달 가능한 상태가 아닐 수 있고, 여러 block의 효과는 비가산적이다. 기존 이상치 주입은 뒤의 latent DGP를 바꾸지 않는다. 반복 intervention CI를 다중 비교 보정하거나 deployment reset의 효과로 해석하지 않는다.
- legacy EWMA 전체 묶음의 잔류 효과를 EWMA160 cell 하나의 인과 효과로 좁히지 않았다. 학습 트리에서 쓰지 않은 CUSUM column의 score 효과가 0인 것은 그 state가 잔류하지 않는다는 뜻이 아니다. 512쌍 common cohort와 342쌍 late cohort는 섞지 않았다.
- 원래 all-DEV synthetic counterexample, 신규 router 진단, fold별 민감도, long-null threshold를 서로 다른 receipt로 구분한다. null-only와 outlier-only TS-AUC는 N/A다. 낮은 eventual FPR만으로 late clean detection을 버린 정책을 개선으로 채택하지 않았다. 무한 horizon alarm 보장은 없다.
- 기존 임계값의 calibration horizon은 1,024점이다. 그 최대값의 marginal rank 보정을 긴 horizon이나 특정 H의 조건부 경보 보장으로 확대하지 않는다. whole-series 교환가능성과 block 독립성은 다른 가정이다. `docs/NEXT3_HORIZON_CALIBRATION_SCOPE.md`의 수학적 대조는 새 모델의 FPR 전망이 아니다.
- E001 precompute는 10% 속도 기준을 넘지 못했다. E002 packed+precompute tie-break의 0.950224는 0.95보다 크므로 탈락했다. E003은 memory 경로를 통과했으나 startup 20% 경로와 throughput 개선은 통과했다고 주장하지 않는다. 비용 trial이 3쌍인 단일 장비 결과이며 OS/power/native library 일반성을 보장하지 않는다.
- RSS, retained NumPy state, model/prepared shared arrays, selector, 모델 파일 크기는 다른 양이다. cold 결과는 표준 library setup 후 import→ready None과 그 후 RSS를 측정했다. process spawn 전체 wall이나 peak H RSS가 아니다. direct stream, 최종 public-API per-yield 확인, median 외삽은 timing 경계가 다르다.
- 최초 E003의 57개 feature names 중 legacy 실제 51개 구간 추출이 잘못돼 unit 4개와 assembly가 실패했다. 원래 source/lock을 보존하고 explicit [:51] 계약으로 고쳤다. 이후 validation-only helper는 과거 CV 모델의 사용하지 않는 old_arch_hash 빈값만 허용했다. 공개 all-DEV load 계약은 느슨하게 바꾸지 않았다. 원래 OOF/ALT 실패와 corrected harness를 모두 남겼다.
- full-suite F410은 Windows venv redirector가 만드는 실제 2-process 트리에 대한 fixture 가정 1개가 틀렸다. payload unit M608은 Windows text newline 변환 fixture가 원인이었다. 두 경우 원래 실패 XML·로그를 보존하고 각각 process tree/명시적 byte fixture로 고친 뒤 통과했다.
- M612 failure-index는 O1의 원래 RESULTS에 역방향 policy hash가 없다는 schema 차이를 처리하지 못했다. 명시적인 O1 정책·hypothesis·source·nested binding을 확인하는 방식으로 고쳤으며 존재하지 않은 hash를 원본에 만들어 넣지 않았다.
- M624 wheel-cache 감사는 py 1.11.0의 합법적인 vendored nested METADATA를 최상위 wheel identity로 잘못 중복 집계했다. 다운로드한 63개 파일을 보존한 채 최상위 dist-info/METADATA만 identity로 검증하는 V2를 만들었다. 공식 PyPI SHA/크기·ZIP CRC·호환 tag를 다시 확인했다. 실패를 설치 실패나 모델 fit 실패로 세지 않았다.
- M649 입력 복사 smoke는 public entry보다 training 모듈을 먼저 import해 Numba cache 쓰기가 감지됐다. 원래 실패를 보존하고 실제 재현처럼 public bootstrap을 먼저 실행한 M650에서 통과했다. runtime은 바꾸지 않았다. 이 검증은 기존 환경에서 빈 학습 입력과 초기 None만 확인하므로 fresh raw 재현과 다르다. 상세 범위는 `docs/NEXT3_COPIED_INPUT_AND_LONG_HORIZON_FINDINGS.md`를 따른다.
- M653 checkpoint 집계는 XML 검증 함수의 필수 기준 시각 인자를 누락해 실패했다. 원래 개별 테스트·검증 증거는 보존됐으며 실제 테스트 worker 시작 시각을 연결해 고쳤다. 실패한 집계에 의존하던 preview 두 개는 실행 없이 대체했다. 이런 집계 실패와 의존성 교체를 scientific fit이나 모델 실패로 세지 않는다.
- 실행 전 최종 감사기는 S1/S2의 원래 CHANGE object를 문자열로 잘못 제한하고 있었다. 구조화된 등록을 허용하되 빈 값과 잘못된 형식은 거부하도록 검증기만 고쳤다. 원래 가설·모델·gate는 보존했으며 실제 child 실패를 추가한 수정은 아니다.
- M667 실제 트리 경계 감사는 Python oracle의 NumPy float32/Python float 비교가 threshold를 좁히는 문제로 witness 검증에서 실패했다. 원래 source·lock·17개 테스트 증거·실패 로그를 보존하고 명시적인 float64 비교와 인접 threshold 회귀 테스트를 추가했다. 수정된 19개 테스트와 5,800개 노드·38,732개 인공 full-forest margin 비교가 통과했다. 모델과 runtime은 바꾸지 않았으며 인공 feature vector의 실제 H/O 도달 가능성은 주장하지 않는다. `docs/NEXT3_TREE_STRUCTURAL_AUDIT.md`를 따른다.
- M677 payload 복사는 보존한 기록의 절대 경로가 274자가 되어 Windows에서 실패했다. 중간 폴더명을 줄인 M679에서는 복사 후 완전한 source-set 검사에서 과거 진입점·잠금 파일 7개 누락이 발견됐다. 두 실패와 복사본을 보존하고 전체 실행 소스 목록을 포함시킨 M681에서 13,260개 파일·904개 실행 입력이 검증됐다. 원래 보호 대상 joblib 모델 4개도 정확한 SHA로 유지했고 역직렬화하지 않았다. 경로·필터·압축 계약 73개와 source 경계 42개 테스트가 통과했다. 최종 복사는 별도 새 source lock 전체를 다시 대조한다.
- 성공 의존성 guard 도입 전 M609 source snapshot은 M608 테스트 실패와 별개로 실행됐다. 이 독립 사실을 숨기지 않고, 후속 작업에는 prerequisite child COMPLETE+exit0를 요구한다. M635 이름 오기는 guard가 실행 전 차단했으며, 원래 6개 등록 버그와 별도 기록했다. supervisor 교체는 계획한 유지보수였고 원본 프로세스 기록을 보존했다. 횟수는 실제 교체 receipt를 집계한 첫 보고서의 장애/재시작 답변을 따른다.
- 최종 감사 시작 전에 시간 집계의 phase 경계를 수정했다. 감사 시작이 heartbeat 구간 중간이면 관측된 구간을 실제 시작 시각에서 연구/감사 시간으로 나누며, 이전의 관측 인정 조건과 누적값은 유지한다. 감독 프로세스 V5는 작업이 없는 경계에서 교체했고 재시작 공백과 이후 활동이 관측되지 않은 시간은 미관측으로 남긴다. 이 수정은 과거 공백의 소급 인정이나 전체 wall을 실제 계산 시간으로 바꾸는 작업이 아니다. `docs/NEXT3_PHASE_ACCOUNTING_REVIEW.md`와 원본·수정 소스 및 71개 회귀 검사를 보존했다.
- 공식 contract 검토 최초 receipt의 human_reviewed_utc 이름은 실제 인간 검토를 뜻하지 않았다. 원본을 남기고 same-agent 의미로 correction을 기록했다. ranking 수식 V1의 escape 표기 오류도 V2와 원본을 함께 보존했다. 이 두 건은 scientific fit 실패가 아니다.
- fresh raw 재현과 archive replay는 Python audit hook과 source/module-origin/file hash 검사 범위다. native library의 모든 syscall을 강제하는 OS sandbox가 아니다. Parquet 파일 bytes hash 또는 mixed row-group 내부 decoding을 seal label 평가와 혼동하지 않는다. seal 값은 모델·평가·feature로 접근하지 않았다.
- 재현용 all-DEV 예측은 training fingerprint이고 품질 점수가 아니다. archive에는 raw competition 관측·label·real point OOF·credentials·환경·wheel cache·기존 ZIP을 넣지 않는다. ZIP만으로 raw/cache가 필요한 모든 과거 테스트나 학습을 할 수 있다고 주장하지 않는다.
'''
    plan = '''# NEXT3 다음 세션 계획

1. **B4: 독립 검증 설계를 먼저 고정한다.** 기존 8,000 DEV를 재사용한 작은 양의 차이는 새로운 독립 증거가 아니다. 다음 세션은 frozen role·source·split·seal·quota SHA를 복원한 뒤, 한정된 가설과 최소 효과·오류 예산·중단 조건·검증 기회 사용 방식을 사전 등록한다. 현재 seal 개봉이나 제출을 승인하는 문서가 아니다. 공동 조건부 envelope를 선택 보정으로 홍보하지 않는다.
2. **B2: 실제 지원을 갖춘 scale-down 변경만 검토한다.** raw/conditional signed 정보가 남는다는 F002를 출발점으로, 새 정보를 더하는 작은 변경과 단순 기존 정보 재표현을 구분한다. DGP 감소만 개선해도 real DEV 전체·median·worst·supported mechanism gate를 넘지 못하면 채택하지 않는다. 감소 79개 subset의 불확실성과 ARCH/tail 분포 차이를 보존한다. 이번 S1/S2/S3를 이름만 바꾸어 재학습하지 않는다.
3. **B6: 잔류 memory 수정의 utility를 공동 검증한다.** D005/D006이 가리킨 legacy EWMA 묶음과 serial evidence를 중심으로 소수의 causal 후보를 사전 등록한다. 고정 null horizon, 단일 이상치 excess, 늦은 변화 clean detection·premature alarm·조건부 delay와 TS-AUC를 동시에 평가한다. counterfactual 상태 교체를 그대로 배포하거나 EWMA160만 원인으로 단정하지 않는다. score compression·age decay로 FPR만 줄여 통과시키지 않는다.

후속 새 독립 데이터의 진단 표본 규모를 계획할 때는 `docs/NEXT3_DIAGNOSTIC_COHORT_SUPPORT_REVIEW.md`를 참고한다. 기존 DEV 집계 비율을 예시로 쓴 48개 가상 조합의 계산이며, 100개 진단 표본의 기대값과 확보 확률을 구분한다. 미래 분포와 IID 전제는 확인되지 않았고 TS-AUC 검정력이나 현재 미개봉 seal의 구성에 대한 추론이 아니다. 표시 확률 1은 유한 정밀도 반올림일 수 있다.

유지할 전달 역할은 DISTILLED J102 tree-used와 PERFORMANCE BL80 lean packed 두 개다. 과학 예측은 NEXT2와 같다. raw·원래 J102·NEXT2 artifacts·이전 ZIP·바탕화면 20개 폴더를 보존한다. 중간 *_PENDING 상태를 보고 완료된 실험을 재실행하지 말고 root final selection과 exact receipt를 확인한다. 미세 혼합비·CV seed/fold·큰 모델의 추가 탐색은 이 계획에 포함되지 않는다.
'''
    seal = '''# Seal 개봉 권고: 개봉하지 않음

SEAL_A/B/C/D는 각 500개이며 모두 미개봉이다. DEV 8,000개와 overlap 0, reduced/public/cloud 신규 사용 0이다. 이번 선택 성능의 시작 대비 Δ는 0.0이고 새로운 performance·robustness·generation 승격이 없다. 메모리 비용 개선을 확인하려고 seal을 열 필요는 없다.

다음 가설·효과 기준·검증 설계를 먼저 고정한다. 이 권고는 이후 seal 개봉이나 제출을 승인하지 않는다.
'''
    public = '''# Public submission 권고: 이번 결과 확인용 제출을 하지 않음

이번 세션에 public/cloud 제출과 SDK test를 실행하지 않았다. reduced 기존 사용 2/2와 ledger bytes를 보존했다. 선택된 lean runtime은 기존 BL80과 예측이 같아 새로운 과학 성능 개선을 확인할 근거가 없다.

공개 추론 API·인과적 current-point 접근·state 크기·Windows pinned 재현은 로컬에서 검증한다. 향후 다른 플랫폼의 운영 확인이 필요하다면 목적과 비용을 별도로 판단해야 한다. 이 문서는 제출 명령을 실행하거나 승인을 대신하지 않는다.
'''
    return {'RESEARCH_OVERVIEW_NEXT3.md': read_first(f, run),
        'RESEARCH_REPORT_NEXT3.md': handoff(f, run),
        'LIMITATIONS_NEXT3.md': limitations, 'NEXT_SESSION_PLAN_NEXT3.md': plan,
        'SEAL_OPEN_RECOMMENDATION_NEXT3.md': seal,
        'PUBLIC_SUBMISSION_RECOMMENDATION_NEXT3.md': public}

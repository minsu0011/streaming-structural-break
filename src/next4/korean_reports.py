"""Korean handoff explanations generated from completed scientific receipts."""
import csv
from src.next4.artifacts import BASE,ROOT,read_json


def number(value,digits=10):return f'{float(value):.{digits}f}'


def interval(p):return '['+number(p['ci_2_5'],12)+', '+number(p['ci_97_5'],12)+']'


def documents(context):
    selection=context['selection'];timing=context['timing'];fits=context['fits'];tests=context['tests'];apis=context['apis']
    primary=selection['roles']['FROZEN_PERFORMANCE']['original_primary'];robust=selection['roles'].get('NEXT4_ROBUSTNESS')
    best=read_json(BASE/'experiments/B2_LOG_ENERGY_SEMITAIL/full/RESULTS.json')
    bp=read_json(BASE/'bootstrap/B2_LOG_ENERGY_SEMITAIL/RESULTS.json')['paired_all_time']
    alt=read_json(BASE/'robustness/B2_LOG_ENERGY_SEMITAIL/RESULTS.json')
    b2lat=read_json(BASE/'latency/B2_DIRECT/RESULTS.json')['summary_median_ratios']
    def candidate(cid):return read_json(BASE/'experiments'/cid/'full/RESULTS.json')
    def family(prefix):
        rows=[read_json(p) for p in (BASE/'experiments').glob(prefix+'*/full/RESULTS.json')]
        return '; '.join(r['candidate']['experiment_id']+': '+number(r['mean'])+' (Δ '+number(r['mean_delta'])+')' if isinstance(r['candidate'],dict) else str(r['candidate'])+': '+number(r['mean'])+' (Δ '+number(r['mean_delta'])+')' for r in rows)
    robust_text='새 강건성 역할 없음.'
    robust_evidence=''
    robust_details={}
    if robust:
        cid=robust['experiment_id'];rp=robust['original_primary'];scale=robust['paired_SCALE']
        robust_text=f"별도 NEXT4_ROBUSTNESS는 {cid}, 평균 {number(rp['mean'])}, Δ {number(rp['mean_delta'])}. Primary champion 교체는 아니다."
        robust_evidence=f"\n강건성 역할 {cid}: 원래 SCALE Δ {number(scale['estimate_mean_delta'])}, CI {interval(scale)}, 변화 시계열 {scale['break_series']}개. 원래 전체 CI는 {interval(robust['paired_all_time'])}이다. 상세 ALT/H·기전·반례 한계는 [채택 검토](docs/NEXT4_FINAL_TRANSFER_REVIEW.md)에 남겼다.\n"
        transfer=read_json(BASE/'robustness'/cid/'RESULTS.json')
        scale_transfer=read_json(BASE/'robustness_forensic'/cid/'RESULTS.json')
        def csv_rows(path):
            with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
        long_rows=csv_rows(BASE/'synthetic/candidates'/(cid+'_all_dev')/'LONG_NULL.csv')
        memory=csv_rows(BASE/'memory_diagnostics'/(cid+'_all_dev')/'PAIRED_OUTLIER_EXCESS.csv')
        fractures=csv_rows(BASE/'forensic'/cid/'PAIRED_FRACTURES.csv')
        robust_details['strongest new idea']='별도 강건성 근거는 '+cid+'의 '+('고정 분산 likelihood' if cid.startswith('D2_') else '분리 energy')+' 신호다. 제한된 SCALE support와 실제 파일 반례 확인을 근거로 유지하며, 전체 성능 향상이나 독립 expert 확증으로 확대하지 않는다.'
        robust_details['ALT_1/2/3']='강건성 역할의 전체 ALT Δ: '+', '.join(s['split']+' '+number(s['mean_delta'])+' CI '+interval(s['paired_all_time']) for s in transfer['summaries'] if s['split'].startswith('ALT_'))+'.'
        h=next(s for s in transfer['summaries'] if s['split']=='LEAVE_ONE_H_REGIME_OUT')
        robust_details['H-regime worst']=f"강건성 역할의 H 제외 평균 Δ {h['mean_delta']!r}, matched 최저 Δ {transfer['H_minimum_matched_delta']!r}, 전체 CI {interval(h['paired_all_time'])}. SCALE 이득과 전체 H 이득은 다르다."
        robust_details['median']=f"강건성 {rp['median']!r}."
        robust_details['worst fold']=f"강건성 {rp['worst_fold']!r}."
        robust_details['bootstrap CI']='강건성 전체 CI '+interval(robust['paired_all_time'])+'이며 전체 성능 승격 근거가 아니다.'
        robust_details['P(delta>0)']='강건성 전체 P='+repr(robust['paired_all_time']['probability_delta_positive'])+'.'
        robust_details['mechanism trade-offs']='강건성 SCALE의 재학습 split별 Δ와 CI: '+ '; '.join(s['split']+' '+number(s['estimate_mean_delta'])+' '+interval(s) for s in scale_transfer['summaries'] if s['scope']=='SCALE')+'. 같은 DEV 재분할에 대한 조건부 검증이다.'
        robust_details['long-null']='실제 all-DEV 강건성 파일의 16384점 alarm fraction (각각 후보/동시 기준 BL80): '+ '; '.join(r['regime']+' α='+r['alpha']+' '+r['eventual_alarm_rate']+'/'+r['reference_eventual_alarm_rate'] for r in long_rows if int(r['horizon'])==16384)+'.'
        robust_details['outlier']='실제 all-DEV 강건성 파일의 512쌍 이후 평균 signed excess 차이: '+ '; '.join(r['scenario']+' Δ '+r['estimate_mean_delta']+' CI ['+r['ci_2_5']+', '+r['ci_97_5']+']' for r in memory if r['metric']=='mean_signed_excess')+'. 음수는 동일 반례에서 기준보다 오염이 작다는 뜻이다.'
        if cid=='D2_DISCRETE_SCALE_MIXTURE':robust_details['outlier']+=' D2는 단발 이상치에서는 개선했지만 반복 이상치는 악화했다. fold-0에서의 반복 이상치 개선이 실제 전체 학습 파일에는 유지되지 않았다. 전반적인 outlier repair로 주장하지 않는다.'
        robust_details['early/late fracture']='강건성 원래 time 구간별 Δ: '+ '; '.join(r['scope']+' '+number(r['estimate_mean_delta']) for r in fractures if r['scope'].startswith('time_'))+'. 신뢰구간은 TIME_FRACTURE_NEXT4.csv에 모두 보존했다.'
        robust_details['latency']='강건성 연구 paired 측정의 중앙 비율: '+repr(robust['direct_runtime_median_ratios'])+'. 별도 감사에서 원래 고정 순서의 10M paired 실행을 다시 확인했다.'
    outcome=selection['session_outcome']
    answers=[
        ('실제 productive research 시간',f"{timing['productive_research_seconds']:.3f}초 ({timing['productive_research_seconds']/3600:.4f}시간). 21,600초 이후 시작한 별도 audit 관측 시간은 {timing['audit_productive_seconds']:.3f}초다. 소급 산입하지 않았다."),
        ('wall-clock',f"최종 보고서 측정 시점 {timing['accounted_utc']}, 세션 경과 {timing['wall_seconds']:.3f}초. 관측되지 않아 연구·감사로 인정하지 않은 시간은 {timing['unobserved_seconds']:.3f}초. 감사 실제 경과 {timing['audit_wall_seconds']:.3f}초. ZIP 생성 이후 시각은 외부 delivery receipt에 별도 기록된다."),
        ('새 hypothesis family 수','사전 등록 8개 A–H. 실제 새 DEV score가 있는 family는 7개다. Family별 screen/full 수는 A 3/3, B 2/2, C 3/3, D 3/3, E 2/1, F 0/0, G 2/1, H 1/1이다. F의 두 slot은 기존 고정 조합 중복과 유효한 세 번째 expert 부재로 새 fit/score 없이 닫았다.'),
        ('screen candidate 수','16개. E2와 G1은 고정 screen 실패 후 종료했다.'),
        ('full candidate 수','14개. 상한 24개를 넘지 않았고, 합성·입력 반전 진단을 새 배포 모델로 선택하지 않았다.'),
        ('learner fit 수',f"새 실제 DEV learner fit 완료 {fits['real_DEV_learner_fit_completions']}회. 단계별 {fits['completed_fits_by_phase']}. 선택용 102회 중 base/state 86회, nested meta 16회. 과거 nested base 56개를 재사용했고 새 frozen-base fit은 0회다. 개발용 합성·mock fit은 이 수에 포함하지 않았다."),
        ('시작 champion','BL2_C205_DEPENDENCE_80 = 0.8 J102 + 0.2 C205.'),
        ('시작 exact score',repr(primary['mean'])+' — 원래 primary 5-fold TS-AUC의 동일 가중 평균.'),
        ('탐색 최고 score',f"B2_LOG_ENERGY_SEMITAIL: {best['mean']!r}, Δ {best['mean_delta']!r}. 탐색 최고와 승격은 다르다."),
        ('최종 선택 score',f"FROZEN_PERFORMANCE 유지: {primary['mean']!r}. "+robust_text),
        ('최종 delta','주 성능 champion은 시작 대비 0.0. '+robust_text),
        ('새 generation 여부','없음. '+outcome+'. +0.0025 generation 문턱을 넘은 후보가 없다.'),
        ('strongest new idea','B2의 raw/conditional lower·upper log-energy 분리는 가장 큰 원래 평균 점 개선을 보였다. 다만 CI 하한이 음수이고 heavy-tail 분산 감소 및 반복 이상치에서 한계가 확인됐다. 새 독립 신호로 확정하지 않았다.'),
        ('dependence V2 결과',family('A')+'. 원래 전체 CI가 모두 0을 포함한다. magnitude/sign/innovation portmanteau 신호가 이번 고정 learner/혼합에서 유의한 성능 승격으로 이어지지 않았다.'),
        ('scale asymmetry 결과',family('B')+'. B1은 SCALE 및 네 분산 감소 반례 조건을 통과했지만 H 제외 SCALE 효과가 불확실하고 반복 이상치 오염은 더 컸다. B2는 전체 CI와 SCALE CI 하한 조건을 모두 통과하지 못했다.'),
        ('distribution result',family('C')+'. H 고정 bin의 Hellinger/JS/shape-ratio 신호를 시험했다. 원래 전체 CI는 모두 0을 포함한다.'),
        ('sequential likelihood result',family('D')+'. D3는 제한된 흡수 hazard mixture이며 full BOCPD가 아니다. 단순한 null에서도 작은 mean 대안과 hazard가 장기 누적 압력을 만들 수 있음을 분석했다.'),
        ('ranking objective result','E1 nested pairwise linear Δ -0.0000467485; E2 interaction은 screen Δ -0.0031355385로 중단. 학습 pair loss 감소가 원래 held-out TS-AUC 개선으로 연결되지 않았다. Outer-validation base prediction을 meta 학습에 쓰지 않았다.'),
        ('H-context result','G1 H-context base screen Δ -0.0348259444로 중단. G2 honest nested pairwise context full Δ +0.0000017733. G1에서 H-only feature가 split gain 약 66–68%를 차지했지만, 이것은 인과적 실패 원인 검증이 아닌 학습 구조 관찰이다.'),
        ('새 complement result','D2 standalone 평균 약 0.60007529, 고정 10% 조합 개선 +0.00032776였으나 신규 독립 expert 채택의 CI/P 조건을 통과하지 못했다. F1은 기존 72/18/10 조합과 중복, F2는 유효한 세 번째 성분 부재로 미실행. H1 고정 선형 상태 readout도 full Δ -0.00068465로 종료했다.'),
        ('5-fold scores','최종 주 성능 BL80: '+repr(primary['fold_scores'])+'. 탐색 B2: '+repr(best['fold_scores'])+'. '+('별도 강건성: '+repr(robust['original_primary']['fold_scores']) if robust else '')),
        ('median',f"BL80 {primary['median']!r}; B2 {best['median']!r}."),
        ('worst fold',f"BL80 {primary['worst_fold']!r}; B2 {best['worst_fold']!r}."),
        ('bootstrap CI',f"B2−BL80 원래 paired 95% CI {interval(bp)}. 하한이 음수다. 원래 2,000개 whole-series multiplicity를 동일하게 재검증했으며 추가 표본으로 CI를 뒤집지 않았다. Frozen 모델 자신과의 차이는 정의상 0이며 새로운 성능 근거가 아니다."),
        ('P(delta>0)',f"B2−BL80 = {bp['probability_delta_positive']!r}. 높은 P만으로 양수 CI 하한 조건을 대체하지 않는다. 재학습·적응적 선택 불확실성을 포함한 확률이 아니다."),
        ('ALT_1/2/3','B2 원래 고정 대체 split의 Δ: '+', '.join(s['split']+' '+number(s['mean_delta'])+' CI '+interval(s['paired_all_time']) for s in alt['summaries'] if s['split'].startswith('ALT_'))+'. 모두 기존 DEV의 다른 grouped partition이며 독립 신규 데이터가 아니다.'),
        ('H-regime worst',f"B2의 H 제외 matched 최저 Δ {alt['H_minimum_matched_delta']!r}. 전체 H 평균 Δ는 약 +0.0004389151. REGIME 표의 원래 OOF 조건부 H slice와 H-regime 전체 제외 후 재학습을 구별했다."),
        ('mechanism trade-offs','SCALE 실제 변화 support 132개. B2 SCALE Δ +0.00621335이나 CI [-0.00079629,+0.01387189]로 불확실하다. raw SD 방향 subset은 감소 79/증가 53개로 각 100개 미만이므로 supported variance-direction 일반화로 주장하지 않는다. MEAN 21, TAIL 86도 표본 부족이다.'+robust_evidence),
        ('early/late fracture','B2는 time 0–99에서 손실, 200 이후에서 주 이득을 보였다. time≥200의 실제 pair-mass 기여 +0.00151536이 전체 +0.00130367보다 커서 앞부분 손실을 상쇄한다. time 0–9에서 실제 scoring-positive 시계열은 182개이며 전체 eventual-break 3944개가 아니다.'),
        ('long-null','모델마다 독립 1024점 null maximum으로 임계값을 정한 뒤 1024/4096/16384에 고정했다. B2 N3 16384점의 누적 alarm fraction은 α=.01에서 0.1875, α=.05에서 0.517578125. 각각 BL80 0.146484375, 0.546875. 명목 α를 모든 길이에 보장하는 anytime 검정은 아니다.'),
        ('outlier','B2의 512쌍 반복 이상치 OB16에서 이후 평균 excess 차이 +0.033498, 마지막 64점 +0.057189로 BL80보다 컸다. 고정 첫 128 seed의 별도 TreeSHAP 분해에서는 upper-energy와 기존 ARCH 상태가 기여했다. SHAP subset과 paired 통계 표본 수를 섞지 않았다.'),
        ('latency',f"B2의 실제 scalar 10M, 5회 paired 측정 median ratio {b2lat['ten_million_ratio']:.6f}, scalar median ratio {b2lat['scalar_median_ratio']:.6f}, H 초기화 포함 amortized ratio {b2lat['amortized_ratio']:.6f}. 모두 3배 이하. 오래된 90초대 기록과 직접 비율을 계산하지 않았다. 추가 강건성 역할의 연구·감사 측정은 LATENCY_NEXT4.json 참조."),
        ('state size','고유 mutable NumPy 배열 기준 BL80 16,925 bytes, B2 35,029 bytes, 길이에 따라 증가하지 않음. Python 객체·RSS·불변 tree 배열 전체 크기는 별도 개념이다. '+('강건성 '+str(apis[robust['experiment_id']]['state_array_bytes_min'])+' bytes.' if robust else '')),
        ('leakage tests',f"최종 전체 검사 {tests['executed_tests']}개 PASS, 실패/오류/skip 0. DEV-only guard, scorer, nested 구조, 현재 관측 하나당 출력 하나, 첫 None, 미래 길이·suffix trap을 검증했다. label permutation control expert는 0.58698915→0.46606936. 제한된 ID rename mock 및 실제 20개 시계열 역순 reset 검사를 구별한다."),
        ('determinism','최종 감사에서 BL80와 B2'+(' 및 강건성 모델' if robust else '')+'의 DEV 8000개/4,014,405개 출력에 대해 offline/stream과 원래 fingerprint가 정확히 일치했다. 새 프로세스 cold import 및 복사된 raw-free payload 실행도 확인했다. 최초 B2 cold mkdir 실패는 보존했고 import-only 수정 후 통과했다.'),
        ('fresh reproduction',('선택한 강건성 모델을 fresh output에서 raw DEV→새 feature→동일 learner→inference로 재현했다. feature·학습 좌표와 inference fingerprint가 기존 materialization과 정확히 같았다. 기존 frozen base는 다시 학습하지 않았다.' if robust else '새 NEXT4 승격이 없어 raw-training 재현은 미실행/불필요. Frozen 모델의 새 inference 검증은 수행했다. B2 cached all-DEV 진단 fit을 fresh raw reproduction이라고 부르지 않는다.')),
        ('DEV saturation risk','HIGH. 원래 full 14개의 전체 CI가 모두 0을 포함한다. 1위 B2−2위 B1 차이 +0.0007558460의 CI는 [-0.0005184872,+0.0019573212], P=0.8835다. B2 ALT gain의 표본 SD는 약 0.0005526835. 원래 draw에서 B2가 최고인 빈도 0.7935는 posterior 확률이 아니다. 기존 draw의 joint max-statistic은 조건부 탐색 진단이며 unconditional FWER나 새 데이터 확인을 대체하지 않는다. 새 seed·split·미세 blend를 추가하지 않았다.'),
        ('SEAL_A/B/C/D opened?','아니오. 네 seal 500개씩 전부 미개봉. raw 전체 바이트 SHA 확인과 seal 관측/label 읽기를 구별했다.'),
        ('reduced usage?','이번 NEXT4 새 사용 0. 과거 누적 2/2는 변경하지 않았다.'),
        ('public submission?','이번 NEXT4 public 0, cloud 0. SDK test/submit을 호출하지 않았다.'),
        ('seal open recommendation','NO. 원래 performance/generation gate 통과가 없고 반복 DEV 탐색 위험이 높다. 별도 SCALE 강건성 역할이 있더라도 독립적인 일반화 확증은 아니다. 실제 개봉은 하지 않았다.'),
        ('다음 연구 병목 3개','① H/ARCH·heavy-tail 조건이 달라져도 재현되는 분산 감소 evidence. ② 반복 이상치 뒤 상태의 회복과 긴 null에서 누적 alarm 제어. ③ 제한된 DEV에서 objective/독립 신호 개선을 정직하게 확인할 검증력. 닫힌 family의 seed·weight shopping을 재개하지 않는다.')]
    answers=[(question,answer+('\n\n'+robust_details[question] if question in robust_details else '')) for question,answer in answers]
    assert len(answers)==42
    readfirst='# NEXT4 연구 개요 — 완료된 과정과 결과\n\n'+outcome+'\n\n주 성능 모델은 **BL2_C205_DEPENDENCE_80, '+repr(primary['mean'])+'**를 유지한다. 탐색 최고 B2의 원래 CI 실패를 유지했다. '+robust_text+'\n\n'
    readfirst+='\n\n'.join(f'{i}. **{question}**\n\n{answer}' for i,(question,answer) in enumerate(answers,1))+'\n'
    handoff='# Structural Break Real-Time NEXT4 인계\n\n'+outcome+'\n\n[42개 답변](RESEARCH_OVERVIEW_NEXT4.md), [실험 표](MODEL_RESULTS_NEXT4.csv), [고정 역할](DEV_CANDIDATES_NEXT4.json), [최종 감사](FINAL_AUDIT_NEXT4.json)를 먼저 읽는다.\n\n'
    handoff+='원래 primary metric은 grouped 5-fold official TS-AUC의 평균이다. 관측 값/시계열 ID/시간/label의 역할, screen/full, standalone/composite, 조건부 H slice/실제 H 제외 학습을 섞지 않는다. 실제 선택 근거는 고정 정책과 원래 OOF이다.\n\n'
    handoff+='실험 메뉴는 A–H 8개 family/18개 slot으로 동결했다. 16개 screen, 14개 full, E2·G1 screen 실패, F1 중복·F2 조건부 종료를 모두 보존했다. 56개 기존 nested base를 재사용했고 16개 새 meta fit은 honest inner OOF를 사용했다.\n\n'
    handoff+='B2 평균 '+repr(best['mean'])+' (Δ '+repr(best['mean_delta'])+')는 숫자 개선 문턱을 넘었으나 CI '+interval(bp)+'로 탈락했다. ALT 3/3 양수, 직접 runtime 통과가 이 결정을 바꾸지 않는다. '+robust_text+'\n'+robust_evidence+'\n'
    handoff+='중요한 추가 근거: [B2 실패 해석](docs/NEXT4_B2_FAILURE_INTERPRETATION.md), [B1 강건성 범위](docs/NEXT4_B1_ROBUSTNESS_INTERPRETATION.md), [입력 부호 진단](artifacts/next4/sign_reflection/RESULTS.json), [방법 경계](docs/NEXT4_METHOD_BOUNDARIES.md), [프로토콜 예외](docs/NEXT4_PROTOCOL_SCOPE_NOTES.md).\n\n'
    if robust:
        handoff+=robust_details['outlier']+'\n\n[최종 채택 범위와 손실](docs/NEXT4_FINAL_TRANSFER_REVIEW.md)을 함께 읽는다.\n\n'
    handoff+='[입력 부호 민감도 해석](docs/NEXT4_SIGN_REFLECTION_INTERPRETATION.md)과 [D2의 SCALE/전체 점수 분할별 도표](artifacts/next4/figures/research_005/D2_DISCRETE_SCALE_MIXTURE_TRANSFER.png)도 포함했다.\n\n'
    handoff+='개발 중 unit fixture 오류와 두 실제 worker 실패를 원본과 함께 남겼다. B2 cold import의 캐시 mkdir은 기존 read-only bootstrap을 설치하는 추가 entry로 해결했다. 합성 worker의 Python 3.13 native access violation 원인은 확정하지 못했다. timed stack dump만 끄고 동일 과학 계산을 이어서 기존 116개 파일·18개 분석을 보존했다. 원래 실패를 PASS로 바꾸지 않았다.\n\n'
    handoff+='실행 환경은 requirements-lock.txt의 정확한 63개 pin이다. Figure 환경은 기존 requirements-next2-figures-lock.txt를 썼다. Python 3.13.2, Windows CPU에서 검증했다. 사용자 staged 변경 9개와 기존 NEXT3 중단 상태를 보존했다.\n\n'
    handoff+='전체 ZIP은 소스·설정·집계 결과·검증 receipt·필요 모델을 포함한다. raw 경쟁 데이터, 실제 point OOF 전체, feature cache, 환경, credentials, 과거 ZIP은 포함하지 않는다. 따라서 학습 재실행에는 RAW_DATA_REQUIRED_NEXT4.json의 SHA와 일치하는 raw X/y가 필요하다. 포함된 모델의 추론은 raw 없이 가능하며, 복사한 payload에서 새 Python 프로세스로 확인했다. 동일 pin의 기존 interpreter를 재사용했으므로 새 환경 설치 검증과는 다르다.\n\n'
    handoff+='배포 호출은 선택 JSON의 entry에서 `infer(datasets, model_directory_path)`를 사용한다. datasets는 (historical array, online iterator)이며 첫 출력은 None, 이후 현재 관측마다 scalar 하나다. 새 프로세스 read-only 검사에서는 Python bytecode 쓰기를 `-B`로 끄며 entry가 프로젝트의 선택적 Numba cache 쓰기를 억제한다.\n\n'
    handoff+='기존 NEXT3는 사용자가 중단한 상태를 유지한다. NEXT4의 최종 감사가 과거 NEXT3의 미완료 최종 감사를 완료로 소급하지 않는다. Desktop 20개 파일 폴더·ZIP과 전체 프로젝트 ZIP의 CRC 및 각 member SHA는 외부 delivery receipt에서 검증한다.\n'
    limitations='# NEXT4 한계\n\n'
    limitations+='- 모든 모델 선택·ALT/H·기전 분석은 같은 DEV 8000개에서 이루어졌다. 네 seal은 미개봉이며 독립 확인 점수는 없다.\n- Paired bootstrap은 fitted OOF 조건부 whole-series 재표본화다. learner 재학습과 적응적 선택의 전체 불확실성을 포함하지 않는다. 14개 원래 전체 CI는 모두 0을 포함한다.\n- B2 개선은 주로 늦은 관측 구간에 집중되며, heavy-tail 분산 감소와 반복 이상치 기억에서 악화가 있었다.\n- SCALE 변화 support는 132개다. 원래 raw SD 방향 subset 79/53, MEAN 21, TAIL 86에서는 supported 기전 일반화를 주장하지 않는다.\n- H-SCALE 이식성이 전체 H 평균과 다를 수 있다. B1은 전체 H 평균이 좋아도 SCALE에서 음수 점 추정치를 보였다.\n- 합성 수치는 고정 모델/생성기/seed/표본 범위에 대한 진단이다. fold-0 모델과 all-DEV 모델 결과를 같은 학습 범위처럼 비교하지 않는다.\n- 1024점 null maximum으로 보정한 threshold는 무한 horizon 보장이 아니다. Paired alarm CI는 threshold calibration 불확실성을 포함하지 않는다.\n- TreeSHAP margin 분해와 sigmoid secant allocation은 feature 제거 재학습이나 인과적 ablation이 아니다.\n- Sign reflection은 원래 DEV의 고정 입력 stress이며 독립 데이터 또는 새 선택 후보가 아니다. 반전 ensemble/augmentation 학습은 하지 않았다.\n- D3는 흡수 hazard approximation이다. full BOCPD 또는 null-safe anytime evidence라고 부르지 않는다.\n- B2의 cached all-DEV 진단 fit은 fresh raw reproduction이 아니다. 새 모델이 실제 채택된 경우에만 별도 raw→feature→fit→inference 재현을 했다.\n- CPU 시간은 이 PC·정확한 runtime·표본·paired 실행 조건에 한정된다. 배열 bytes는 전체 RSS가 아니다.\n- 최초 cold import 실패와 native worker crash를 보존했다. 짧은 probe로 native crash는 재현되지 않아 원인은 확정하지 못했다.\n- 일부 개발 unit/prototype 검사가 후보 등록보다 먼저 수행됐다. 실제 DEV fit/score 이전 등록과 source freeze는 검증했다. 45분을 넘긴 bottleneck 검토 간격은 실제 timestamp로 남겼다.\n- 전체 ZIP은 raw/OOF/cache/환경을 제외한다. 다른 PC의 새 환경 설치나 외부 제출 시스템 실행을 완료했다고 주장하지 않는다.\n\n'+robust_text+'\n'
    plan='# 다음 세션 계획\n\n'
    plan+='1. 먼저 NEXT4 최종 선택과 failure/stop 기록, 원래 frozen hashes를 확인한다. B2의 CI 실패를 새 bootstrap seed나 추가 draw로 지우지 않는다.\n2. 분산 감소 evidence가 H-regime과 heavy-tail에서 다르게 작동하는 원인을 조사한다. 새로운 연구를 할 경우 한정된 가설·학습/평가 좌표·중단 조건을 새 점수 전에 고정한다. 기존 반례를 보며 분모나 조건을 바꾸지 않는다.\n3. 반복 이상치 오염과 null horizon 증가를 구별한다. reset/forgetting 모델을 연구한다면 별도 scientific identity로 등록하고, 실제 TS-AUC와 단발·반복 이상치 trade-off를 함께 확인한다.\n4. 독립 expert 또는 TS-AUC 목적함수 연구는 이번에 종료한 A/C/E/G/H 메뉴의 결과를 먼저 검토한다. 고정 split의 낮은 검증력과 높은 DEV 탐색 포화를 비용에 포함한다.\n5. 성능 champion은 BL80을 유지한다. 실제 seal/reduced/public/cloud 실행은 이번 세션의 후속 자동 작업으로 예약하지 않는다.\n\n이번 세션의 미완료 필수 작업을 다음 세션으로 넘기기 위한 문서가 아니다. 완료 evidence와 명시적 한계를 바탕으로 별도 미래 연구 방향을 적었다.\n'
    seal='# Seal 개봉 권고: NO\n\n네 seal은 모두 미개봉이다. 원래 performance/generation gate를 통과한 새 모델이 없고 DEV 탐색 포화가 높다. SCALE 강건성 역할은 제한된 지원 표본에서의 근거이며 독립 일반화 확증으로 간주하지 않는다. 이번 세션에서 실제 seal을 열거나 개봉을 예약하지 않았다.\n'
    public='# Public submission 권고: NO\n\n이번 세션에서 public/cloud submission 및 SDK test를 실행하지 않았다. 기준 성능 모델을 유지하며, 탐색 최고 B2의 원래 CI 실패가 해결되지 않았다. 새 강건성 역할이 있어도 이를 새 generation 또는 leaderboard 개선으로 주장하지 않는다.\n'
    return {'RESEARCH_OVERVIEW_NEXT4.md':readfirst,'RESEARCH_REPORT_NEXT4.md':handoff,'LIMITATIONS_NEXT4.md':limitations,
        'NEXT_SESSION_PLAN_NEXT4.md':plan,'SEAL_OPEN_RECOMMENDATION_NEXT4.md':seal,'PUBLIC_SUBMISSION_RECOMMENDATION_NEXT4.md':public}

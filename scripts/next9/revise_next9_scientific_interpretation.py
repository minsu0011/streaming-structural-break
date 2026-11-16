"""Preserve the first report and revise precision/interpretation before delivery."""
from pathlib import Path
import sys,shutil,re,csv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

with activity('Review20answers and17field cards; preserve draft and make exact differences and EARLY failure explicit',
              'SCIENTIFIC_REPORTING','B4_EARLY_LATE_REPLICATION'):
    receipt=BASE/'SCIENTIFIC_REPORTS_DONE_NEXT9.json';old=read_json(receipt)
    assert 'supersedes_report_receipt' not in old
    preserved=(BASE/'repairs/scientific_report_v4_preserved').resolve()
    assert preserved.is_relative_to(BASE.resolve()) and not preserved.exists()
    preserved.mkdir(parents=True)
    for name,sha in old['files_sha256'].items():
        assert digest(BASE/name)==sha
        shutil.copyfile(BASE/name,preserved/name);assert digest(preserved/name)==sha
    shutil.copyfile(receipt,preserved/receipt.name);old_sha=digest(receipt)
    result=read_json(BASE/'MODEL_ONLY_FIVE_FOLDS_NEXT9.json')['results']
    normal=result['NORMAL']['contrasts']['M7__vs__M0']
    early_state=result['EARLY']['contrasts']['M3__vs__M2']
    assert len(early_state['fold_deltas'])==5 and all(v<0 for v in early_state['fold_deltas'])
    text=(BASE/'SCIENTIFIC_READ_FIRST_NEXT9_KO.md').read_text(encoding='utf-8')
    text=re.sub(r'^12\. .*$',f"12. NORMAL 평균 TS-AUC 차이(저장된 전체 정밀도): {normal['point']!r}.",text,flags=re.M)
    text=re.sub(r'^14\. .*$',f"14. 전체 원래 group을 2,000회 재표본화하고 5개 fold를 동일 가중했다. 95% CI=[{normal['ci_low']!r}, {normal['ci_high']!r}]. 고정 학습 절차 조건부 불확실성이며 RP 반복 사용 전체를 보정한 독립 확인이 아니다.",text,flags=re.M)
    text=re.sub(r'^(9\. .*)$',lambda m:m.group(1)+f" EARLY에서는 M3−M2={early_state['point']!r}, 95% CI=[{early_state['ci_low']!r}, {early_state['ci_high']!r}]로 5개 fold 모두 음수였다.",text,flags=re.M)
    with (BASE/'SELF_TRANSITION_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:self_rows=list(csv.DictReader(f))
    positive={}
    for r in self_rows:
        if r['mode']=='EARLY' and r['domain']=='ALL' and r['contrast']=='T2_SELF_RUN_SLOPE_H__vs__T3_SAME_INPUT_DEPTH3' and r['ci_low'] not in ('','None') and float(r['ci_low'])>0:
            positive.setdefault(r['feature'],set()).add(r['fold'])
    early_tree=sum(len(v)>=4 for v in positive.values())
    text=text.replace('같은 입력 T3도 이긴 후보는 NORMAL 0개다.',f'같은 입력 T3도 이긴 후보는 NORMAL 0개, EARLY {early_tree}개다.')
    text=re.sub(r'^19\. .*$', '19. 핵심 병목: 조건부 반대 방향의 반복 검증 실패, EARLY 상태 주효과의 5/5 fold 성능 저하, 자기 예측력과 필터 기억의 구분이다. RP 반복 사용은 미세 점수 차이의 독립 확인을 제한한다. 표본 위상과 전체 시점 자기 전이 검증은 별도 사후 진단으로 보고한다.',text,flags=re.M)
    hypotheses=('다음 세션의 사전등록 가설 3개: 초기 구간에서 학습한 상태별 계수의 불안정성이 EARLY 일반화를 제한하는가; '
        '긴 FLAT 구간을 가로지르는 유한 기억의 단절이 반전 관측을 왜곡하는가; 단일 충격에 대한 필터 응답으로 H-center 대비 자기 예측 이득을 설명할 수 있는가. '
        '이번 세션에서는 이를 이용한 후보·계수·임계값 조정을 하지 않는다.')
    text=re.sub(r'^20\. .*$', '20. '+hypotheses,text,flags=re.M)
    assert [int(v) for v in re.findall(r'^(\d+)\.',text,re.M)]==list(range(1,21))
    assert '??' not in text and '\ufffd' not in text
    atomic(BASE/'SCIENTIFIC_READ_FIRST_NEXT9_KO.md',text,immutable=False)
    plan='''# 다음 세션 사전등록 제안

현재 판정은 NO_MOVEMENT_VARIABLE_FEATURE이며 primary BL80, robustness D2, distilled J102를 유지한다. 아래 가설은 이번 결과에서 제안한 후속 질문이다. 이번 NEXT9 결과를 새 가설의 검증 표본으로 다시 사용해 성공을 선언하지 않는다.

1. 초기 구간에서 학습한 상태별 계수의 불안정성이 EARLY 일반화를 제한하는가. M3−M2가 5개 폴드에서 모두 음수였지만 상태 빈도 이동만으로 원인을 단정할 수 없다. 다음 세션에서 TRAIN 내부에 고정된 시간 절단을 두고 계수·캘리브레이션의 재현성을 조사한다. 선택 규칙과 정규화는 외부 점수를 열기 전에 동결한다.

2. 긴 FLAT 구간의 유한 기억 단절이 반전 관측을 왜곡하는가. 고정 16개 값 정의와 관측된 전체 prefix 진단을 비교한 결과를 먼저 읽는다. 다음 세션에서 대체 기억 규칙을 검토한다면 정답 없는 합성 경로로 먼저 동결하고, 현재 NEXT9 상태를 소급 교체하지 않는다. 관측된 reversal age를 true break age로 부르지 않는다.

3. 단일 충격의 필터 응답으로 H-center 대비 자기 예측 이득을 설명할 수 있는가. 같은 배경 잡음에 대한 단일 충격·무충격 경로와 TRAIN 평균 상수 분해를 이용해, 기계적 필터 기억과 잔여 동역학을 구분할 모형을 사전등록한다. 단순히 H 중앙값보다 좋다는 사실을 새로운 메커니즘 증거로 사용하지 않는다.

SELF_CADENCE_FINDING_NEXT9_KO.md와 SAMPLING_PHASE_FINDING_NEXT9_KO.md는 표본 간격과 순차 발행의 사후 진단이다. 추가 피처·state·parent 탐색이나 이번 B/C 인증에 사용하지 않는다. 같은 입력 T3 비교, 원래 group 제외, 다중비교와 공통 F 지원 조건을 계속 분리해 보고한다.

기존 RP는 반복 개발에 사용되었다. 독립 hidden 확인으로 표현하지 않는다. 새 검증 데이터 사용은 별도 승인 범위와 프로토콜을 따라야 하며 이번 세션의 Shadow/Seal 접근 금지는 그대로 유지한다.
'''
    atomic(BASE/'NEXT_SESSION_PLAN_NEXT9.md',plan,immutable=False)
    revised={**old,'timestamp':utc_now(),'revision':'INTERPRETATION_REVIEW_V5',
        'supersedes_report_receipt':str((preserved/receipt.name).relative_to(BASE)),
        'supersedes_report_receipt_sha256':old_sha,
        'changes':'Full-precision M7 delta and interval; EARLY M3-M2 negative in all5folds; actual EARLY same-input T3 count; explicit prospective hypotheses. No scientific results changed.',
        'files_sha256':{name:digest(BASE/name) for name in old['files_sha256']}}
    atomic(receipt,revised,immutable=False)
    cards=(BASE/'FEATURE_CARDS_NEXT9_KO.md').read_text(encoding='utf-8')
    sections=re.split(r'^## F\d+',cards,flags=re.M)[1:]
    assert len(sections)==9 and all([int(n) for n in re.findall(r'^(\d+)\. ',s,re.M)]==list(range(1,18)) for s in sections)
    atomic(BASE/'SCIENTIFIC_INTERPRETATION_REVIEW_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
        original_receipt_sha256=old_sha,revised_receipt_sha256=digest(receipt),
        cards=9,fields_per_card=17,actual_numbered_answers=20,original_files_preserved=True,
        no_candidate_model_score_or_threshold_changes=True))
    print(dict(status='PASS',answers=20,cards=9))

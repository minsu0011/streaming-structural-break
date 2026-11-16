"""Map the user's 81 numbered sections to observed evidence and pending final work."""
from pathlib import Path
import csv
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,digest,progress,read_json,utc_now

POLICY='configs/next3/RESEARCH_POLICY_LOCK.json'
PREFLIGHT='artifacts/next3/preflight/NEXT3_PREFLIGHT.json'
SCIENCE='docs/NEXT3_SCIENTIFIC_MENU_CLOSURE.md'
CHRONOLOGY='docs/NEXT3_REGISTRATION_CHRONOLOGY_AUDIT.md'
ROUTER='docs/NEXT3_ROUTER_METRIC_ALIGNMENT_FINDINGS.md'
EVIDENCE='docs/NEXT3_EVIDENCE_NAVIGATION.md'
MODELS='artifacts/next3/final_tables/prepared_v2/MODEL_RESULTS_NEXT3.csv'
FINAL='docs/NEXT3_FINAL_AUDIT_EXECUTION_PROTOCOL.md'
TESTS='artifacts/next3/maintenance/FULL_SUITE_CHECKPOINT_10_RESULT.json'
CLAIMS='artifacts/next3/maintenance/report_previews/review_v5_payload_and_support/claims/RESULTS.json'
BOARD='artifacts/next3/bottlenecks/BOTTLENECK_BOARD.json'
RULES={}


def assign(numbers,status,note,*evidence):
    for number in numbers:
        if number in RULES:raise ValueError('Repeated user section')
        RULES[number]={'status':status,'interpretation':note,'evidence':list(evidence)}


assign([0,75,76,78],'목표·원칙 유지, hidden 성능 미검증','TS-AUC가 primary다. 기존 두 과학 역할을 보존하며 engineering 비용 개선을 과학 점수 상승으로 바꾸어 쓰지 않는다.',POLICY,SCIENCE,EVIDENCE)
assign([1,2,3,4,5],'기존 상태 복원·검증, 최종 재확인 대기','8,000 DEV/2,000 seal과 원래 hash·scorer·quota·역할을 복원했다. 현재 유지와 별개로 최종 raw/guard 검사를 다시 실행해야 한다.',PREFLIGHT,'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
assign([6,68,69,70,80],'최종 필수 실행 대기','11시간 10분 이후 실제 final audit 시작, 별도 약 50분 감사와 fresh pinned raw→feature→train→infer가 남아 있다. 사전 준비를 최종 실행으로 세지 않는다.',FINAL,'docs/NEXT3_INDEPENDENT_TRAINING_PROTOCOL.md')
assign([7,8,10,55,56,63,64],'구현·실행 기록 보존','단일 소유 worker, 실제 heartbeat, 성공 의존성, immutable/atomic 기록과 QUARANTINE을 사용한다. BUG와 과학적 음성 결과를 분리하며 실패를 덮어쓰지 않는다. 모든 장애의 자동 복구를 보장하지 않는다.','docs/NEXT3_SUPERVISOR_LIFECYCLE_VALIDATION.md','docs/NEXT3_SUCCESS_DEPENDENCY_MAINTENANCE.md','artifacts/next3/maintenance/CHECKPOINT_13_REPAIR_COMPLETION.json')
assign([9],'정기 자원·tmp 및 현재 저장량 계측','RAM·disk·child·NEXT3 .tmp의 약 30분 기록이 있다. 전체 cache/환경/보존 복사본의 논리 크기는 별도 현재 metadata walk로 측정했다. 과거 전체 cache 크기를 30분마다 완전히 측정했다고 주장하지 않는다.','artifacts/next3/runtime/RESOURCE_LOG.csv','artifacts/next3/maintenance/storage_footprint/preaudit_01/RESULTS.json','artifacts/next3/maintenance/storage_footprint/source_label_correction/CORRECTION.json')
assign([11,12,13,39,58,61,62,77],'병목 재평가·유지보수 진행','동시 ACTIVE는 최대 두 개다. 유한 과학 메뉴 종료 후 B4/B8로 이동했고, 경로 실패와 소스 누락을 발견해 수정했다. 12시간 전체 종료는 아직 아니다.',BOARD,'artifacts/next3/maintenance/CHECKPOINT_13_REPAIR_COMPLETION.json')
assign([14,15,16,17,18],'유한 router 평가 종료, 새 역할 없음','R1/R2/R3를 제한된 메뉴와 nested base OOF로 검토했다. 일부 trade-off 개선은 있으나 모든 frozen gate를 넘는 대체 역할은 없었다.',ROUTER,CHRONOLOGY,MODELS)
assign([19,20,21,22],'감소 정보 진단·후보 평가, 해결 미달','S1/S2/S3와 관측 기반 SCALE subset을 평가했다. 감소 정보의 부재만으로 실패를 설명할 수 없지만 real DEV와 ARCH/tail repair gate를 동시에 넘지 못했다. 작은 subset은 지원 부족을 명시한다.',SCIENCE,MODELS,'artifacts/next3/final_tables/prepared_v2/MECHANISM_RESULTS_NEXT3.csv')
assign([23,24,25,26],'보완 신호·가지치기·고정 blend 평가 종료','P1–P4, I1 재사용, I2와 O1을 기록했다. 미세 혼합비·새 CV seed 탐색을 추가하지 않았고 새 complement 승격이 없다.',SCIENCE,MODELS,'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
assign([27,28,29],'원인·재현 진단, 배포 가능한 repair 없음','장기 null은 secondary다. D005/D006이 legacy EWMA 묶음과 serial 상태의 이상치 잔류 효과를 보여도 이를 reset 처방이나 특정 EWMA160 단독 원인으로 확정하지 않는다.','docs/NEXT3_OUTLIER_STATE_TRANSFER_FINDINGS.md','docs/NEXT3_LEGACY_STATE_TRANSFER_FINDINGS.md','docs/NEXT3_HORIZON_CALIBRATION_SCOPE.md')
assign([30],'우선순위상 보류','초기 online age의 작은 공식 pair mass를 기록했고 유한 메뉴 종료 후 early-only 새 모델을 추가하지 않았다.',BOARD,EVIDENCE)
assign([31,37,38,59,60],'유한 메뉴 종료, 일반화 불확실성 미해결','13개 새 recipe 중 12 full/1 screen 실패다. I1은 과거 실패 재사용, S4는 보류다. 10개 failed-screen 조건은 충족하지 않아 formal saturation은 NOT_FORMALLY_TRIGGERED이며 CASE G 유지보수 전환을 적용했다.',SCIENCE,MODELS,'docs/NEXT3_INDEPENDENT_VALIDATION_PLAN.md')
assign([32,33,34],'정책 보존, 새 scientific promotion 없음','performance·generation·robustness 수치 gate를 낮추지 않았다. I2의 작은 양의 DEV 차이나 R1/R3의 일부 진단 이득을 통과로 바꾸지 않는다.',POLICY,MODELS,'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
assign([35,65],'기존 두 과학 역할 동결','DISTILLED J102와 PERFORMANCE BL80를 유지한다. E003은 같은 과학 예측의 runtime 표현이며 최종 실제 전달 수락은 아직 남아 있다.','configs/next3/SCIENTIFIC_ROLE_FREEZE.json',EVIDENCE)
assign([36],'사전 screen/full 경로 기록','screen을 통과한 경우만 등록 경로에 따라 full로 이동했다. passing screen을 saturation용 failed screen으로 재분류하지 않는다.',CHRONOLOGY,MODELS)
assign([40,41],'기여·mechanism 진단 범위 유지','mechanism label은 diagnostic으로만 사용했다. pair-mass 기여와 subset 조건부 score를 구분하며 이를 서로 더할 수 있는 같은 수치로 취급하지 않는다.',EVIDENCE,'artifacts/next3/final_tables/prepared_v2/MECHANISM_RESULTS_NEXT3.csv')
assign([42,43],'metric·조건부 통계 수치 대조','TS-AUC와 W3/logistic 손실의 유한 후보 순서는 다를 수 있다. D007 및 169개 saved-statistics join은 수치 재현이며 adaptive 선택 보정이나 독립 일반화 검증이 아니다.','docs/NEXT3_RANKING_SURROGATE_METHODOLOGY_V2.md','docs/NEXT3_PAIR_MATRIX_BOOTSTRAP_AUDIT_FINDINGS.md','docs/NEXT3_CONDITIONAL_STATISTICS_REPLAY.md')
assign([44,45],'기존 locked 검증 유지, 새 ALT/H fit 없음','기존 ALT1/2/3와 H holdout 결과는 inherited로 구분한다. 새 후보는 앞선 gate에서 실패해 추가 ALT/H fit으로 선택을 이어가지 않았다. E003의 20개 기존 모델 runtime parity는 별도다.','artifacts/next3/final_tables/prepared_v2/ALT_SPLIT_RESULTS_NEXT3.csv','artifacts/next3/final_tables/prepared_v2/REGIME_RESULTS_NEXT3.csv',EVIDENCE)
assign([46],'조건 미충족으로 미실행','새 generation-level 후보가 없으므로 25/50/75/100% learning curve fit을 추가하지 않았다. 이런 곡선으로 memorization 여부를 검증했다고 주장하지 않는다.',SCIENCE,POLICY)
assign([47],'선택적 fit 대조 미실행','이번 세션에 label-permutation 또는 feature-time-shift learner fit을 추가하지 않았다. causal/prefix·seal admission·nested lineage·scorer 회귀 검증은 이 optional 음성 fit 대조를 대신 수행했다고 표현하지 않는다.',CHRONOLOGY,TESTS)
assign([48,49,50,51],'engineering parity·비용 gate, 최종 raw/API 대기','세 runtime family의 고정 gate와 exact 검증을 보존한다. E003은 cold RSS route를 통과했으며 새 속도 향상을 주장하지 않는다. 최종 새 raw 모델과 실제 public API 30M 검증이 남아 있다.',EVIDENCE,'docs/NEXT3_TREE_STRUCTURAL_AUDIT.md',FINAL)
assign([52,53,54],'중간 회귀·복사 검증, 최종 전체 suite 대기','최신 전체 1,019개와 source 전후 904개가 검증됐다. Windows 경로와 root source 누락도 실제 복사에서 수정했다. source hash나 static signature 검토를 모든 code-quality 속성의 보장으로 쓰지 않는다.',TESTS,'docs/NEXT3_WINDOWS_PAYLOAD_REPAIR.md','artifacts/next3/maintenance/FINAL_EXECUTION_CALLSITE_REVIEW_V2.json')
assign([57],'중대한 위반 발견 없음, 최종 재확인 대기','현재 기록에서 seal 개봉·quota 증가·과학 gate 완화는 없었다. 별도의 최종 guard/raw/source/재현 검사는 계속 필수다.',PREFLIGHT,CHRONOLOGY,FINAL)
assign([66,67],'미개봉·미제출 권고 초안, 최종 발행 대기','새 scientific promotion이 없으므로 seal을 열거나 결과 확인용 제출을 하지 않는 방향이다. 실제 개봉·제출을 승인하지 않는다.','artifacts/next3/maintenance/report_previews/review_v5_payload_and_support/SEAL_OPEN_RECOMMENDATION_NEXT3.md','artifacts/next3/maintenance/report_previews/review_v5_payload_and_support/PUBLIC_SUBMISSION_RECOMMENDATION_NEXT3.md')
assign([71,72,73,74,79],'최종 보고·실제 배포 증거 대기','44개 한국어 답변과 245개 수치 join 초안은 통과했다. 최종 root 표·board·audit·ZIP·실제 압축 해제 재생·정확히 20개 파일 및 종료 상태는 실제 final 단계에서만 확정한다.',CLAIMS,'docs/NEXT3_FINAL_DELIVERY_FILE_MAP.md',FINAL)


def main():
    session=read_json(BASE/'SESSION.json')
    if session.get('audit_started_utc'):raise RuntimeError('This is explicitly a pre-audit traceability snapshot')
    specification=ROOT/'docs/RESEARCH_PROTOCOL_NEXT3.md'
    headings=re.findall(r'^([0-9]{1,2})\. ([^\n]+)\n=+',specification.read_text(encoding='utf-8'),re.M)
    assert [int(n) for n,_ in headings]==list(range(81))
    assert set(RULES)==set(range(81)), sorted(set(range(81))-set(RULES))
    out=BASE/'maintenance/requirements_traceability_pre_audit_v1';out.mkdir(parents=True,exist_ok=False)
    snapshots={}
    sources=sorted({specification.relative_to(ROOT).as_posix(), 'artifacts/next3/SESSION.json',
        *[name for row in RULES.values() for name in row['evidence']]})
    for index,name in enumerate(sources):
        source=ROOT/name
        assert source.is_file() and not source.is_symlink() and not source.is_junction(),name
        target=out/'inputs'/f'{index:03d}_{source.name}';target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
        snapshots[name]={'snapshot':target.relative_to(ROOT).as_posix(),'sha256':digest(target),'bytes':target.stat().st_size}
        assert digest(source)==snapshots[name]['sha256']
    rows=[{'section':int(number),'requirement':title,**RULES[int(number)]} for number,title in headings]
    with (out/'TRACEABILITY.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['section','requirement','status','interpretation','evidence'])
        writer.writeheader();writer.writerows({**row,'evidence':';'.join(row['evidence'])} for row in rows)
    created=utc_now()
    text=f'''# NEXT3 사용자 요구사항의 사전 감사 교차표

기준 시각: {created}. 사용자 원문의 0–80번 **81개 절**을 실제 증거 또는 남은 작업에 연결했다. 이 표의 PASS는 교차표의 누락 없는 구성과 입력 복사를 뜻하며, 최종 감사 완료나 모든 요구의 충족을 뜻하지 않는다. 최종 권한은 root selection·FINAL_AUDIT_NEXT3 및 실제 ZIP 이후 외부 검증·SESSION_COMPLETION 기록에 있다.

원문과 참조 입력 {len(snapshots)}개를 당시 bytes로 별도 보존했다. 진행 중 board·resource log·SESSION의 이후 변화는 과거 교차표를 소급해 바꾸지 않는다. 새 scientific fit·품질 평가·seal 접근·bootstrap은 0회다.

| 절 | 사용자 요구 | 현재 범위 | 판단 및 남은 작업 | 당시 증거 사본 |
|---:|---|---|---|---|
'''
    for row in rows:
        links=' · '.join(f'[{i+1}](../{snapshots[name]["snapshot"]})' for i,name in enumerate(row['evidence']))
        text+=f'| {row["section"]} | {row["requirement"].replace("|","/")} | {row["status"]} | {row["interpretation"]} | {links} |\n'
    text+='''
조건부로 생략한 항목을 완료라고 바꾸지 않는다. 새 generation 후보가 없어 learning curve를 수행하지 않았고, 선택적 label-permutation/time-shift learner fit도 수행하지 않았다. 기존 ALT/H는 새 독립 증거가 아니다. 정기 tmp/resource 기록과 이번 전체 저장량 metadata walk도 서로 다른 범위다.

최종 단계에서는 시간 경계, 새 환경 raw 재학습 3회, 전체 suite의 실제 collection/execution 일치, lineage, 새 raw 모델의 트리 구조, public API 30M, 원본 보존, 최종 표·44개 답변, 현재 source 전체가 포함된 새 payload, 실제 ZIP SHA/CRC, 실제 압축 해제와 두 종류의 98,372점 재생, 바탕화면 20개 복사를 순서대로 검증해야 한다. 준비된 계획이나 이 교차표를 그 실행 결과로 사용하지 않는다.
'''
    document=ROOT/'docs/NEXT3_REQUIREMENTS_TRACEABILITY_PREAUDIT.md'
    with document.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    atomic_json(out/'RESULTS.json',{'status':'PASS_SPECIFICATION_TRACEABILITY_MAP_NOT_FINAL_AUDIT','created_utc':created,
        'user_sections':81,'sections_exactly_0_through_80':True,'rows':rows,'input_snapshots':snapshots,
        'document':document.relative_to(ROOT).as_posix(),'document_sha256':digest(document),
        'table_sha256':digest(out/'TRACEABILITY.csv'),'source_sha256':digest(Path(__file__)),
        'new_fits':0,'new_quality_scores':0,'new_Monte_Carlo_draws':0,
        'learning_curve_fits_performed':False,'label_permutation_or_time_shift_learner_fits_performed':False,
        'all_requirements_claimed_complete':False,'actual_final_audit_claimed_complete':False},immutable=True)
    progress('User requirement traceability reviewed','All 81 sections linked to observed inputs or explicit pending/optional scope; no final execution or completeness claim')
    print('NEXT3 REQUIREMENT TRACEABILITY PASS',len(rows),len(snapshots),flush=True)


if __name__ == '__main__':main()

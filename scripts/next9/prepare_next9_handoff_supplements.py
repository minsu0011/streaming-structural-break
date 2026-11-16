"""Repair review-detected Korean text and include new evidence in exact20 files."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now,activity

with activity('Review Korean handoff encoding and consolidate auxiliary/numerical evidence into exact20 files',
              'HANDOFF_PREPARATION','B9_NORMALIZATION_STATE_AVAILABILITY'):
    old=ROOT/'scripts/next9/finalize_next9_reports_v11.py'
    source=old.read_text(encoding='utf-8');lines=source.splitlines(keepends=True)
    matches=[i for i,line in enumerate(lines) if "lead+='??" in line];assert len(matches)==1
    original=lines[matches[0]]
    lines[matches[0]]="        lead+='추가 검증 근거: 전체 학습 모델의 유한값·정규화·수렴 검사는 FITTED_MODEL_PARAMETER_AUDIT_NEXT9.json, 실패 작업과 성공한 복구의 대응은 FAILED_ATTEMPT_RECONCILIATION_NEXT9.json에 있다. EARLY 상태 빈도와 점수 성분의 변화는 STATE_TRANSPORT_FINDING_NEXT9_KO.md에 기록했으며 성능 저하의 인과 원인으로 단정하지 않았다. 독립 TS-AUC 교차 검증은 INDEPENDENT_TS_AUC_AUDIT_NEXT9.json을 참조한다.\\n\\n'\n"
    source=''.join(lines)
    assert '??' not in source and '\ufffd' not in source
    compile(source,'finalize_next9_reports_v12','exec');atomic(ROOT/'scripts/next9/finalize_next9_reports_v12.py',source)
    original_desktop=ROOT/'scripts/next9/deliver_next9_desktop.py'
    source=original_desktop.read_text(encoding='utf-8')
    source=source.replace("'EXECUTION_AUDIT_NEXT9.json'],", "'EXECUTION_AUDIT_NEXT9.json','CURRENT_CONTROLLERS_NEXT9.json'],")
    source=source.replace("'STATE_OBSERVABILITY_NEXT9.json'],", "'STATE_OBSERVABILITY_NEXT9.json','SELF_CADENCE_DONE_NEXT9.json','SELF_CADENCE_FIVE_FOLD_NEXT9.csv','SELF_CADENCE_FINDING_NEXT9_KO.md'],")
    source=source.replace("'H_GENERALIZATION_NEXT9.csv','CV_RESULTS_NEXT9.csv'],", "'H_GENERALIZATION_NEXT9.csv','CV_RESULTS_NEXT9.csv','STATE_TRANSPORT_SUMMARY_NEXT9.csv','STATE_TRANSPORT_FINDING_NEXT9_KO.md','SAMPLING_PHASE_SUMMARY_NEXT9.csv','SAMPLING_PHASE_FINDING_NEXT9_KO.md'],")
    source=source.replace("'DETERMINISM_NEXT9.json','REPRODUCTION_NEXT9.json']", "'DETERMINISM_NEXT9.json','REPRODUCTION_NEXT9.json','INDEPENDENT_TS_AUC_AUDIT_NEXT9.json','LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json','FITTED_MODEL_PARAMETER_AUDIT_NEXT9.json','FAILED_ATTEMPT_RECONCILIATION_NEXT9.json','EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json']")
    source=source.replace("                else:\n                    with path.open(encoding='utf-8-sig',newline='') as f:value=list(csv.DictReader(f))", "                elif path.suffix=='.csv':\n                    with path.open(encoding='utf-8-sig',newline='') as f:value=list(csv.DictReader(f))\n                else:\n                    assert path.suffix in ('.md','.txt')\n                    value=path.read_text(encoding='utf-8-sig')")
    compile(source,'deliver_next9_desktop_v2','exec');atomic(ROOT/'scripts/next9/deliver_next9_desktop_v2.py',source)
    atomic(BASE/'repairs/HANDOFF_SUPPLEMENT_REVIEW_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
        original_finalizer_sha256=digest(old),original_invalid_text=original,
        finalizer_sha256=digest(ROOT/'scripts/next9/finalize_next9_reports_v12.py'),
        original_desktop_script_sha256=digest(original_desktop),desktop_script_sha256=digest(ROOT/'scripts/next9/deliver_next9_desktop_v2.py'),
        reason='A previously generated Korean sentence had replacement question marks. Corrected from known audit meanings before any final report was produced. New numerical and cadence evidence consolidated into existing20-file slots, preserving exact count.',
        no_result_or_model_changes=True))
    current=read_json(BASE/'CURRENT_CONTROLLERS_NEXT9.json')
    current.update(timestamp=utc_now(),final_reports='finalize_next9_reports_v12.py --core-tag c0001',
                   desktop='deliver_next9_desktop_v2.py --archive-tag f0001')
    atomic(BASE/'CURRENT_CONTROLLERS_NEXT9.json',current,immutable=False)
    print(dict(status='PASS',exact_desktop_files=20))

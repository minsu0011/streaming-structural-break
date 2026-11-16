"""Supplement frozen final audit entry points with solver-recovery evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,digest,utc_now

old=ROOT/'scripts/next9/admit_next9_final_audit.py'
source=old.read_text(encoding='utf-8').replace("'SCIENTIFIC_RESULTS_NEXT9.json','SCIENTIFIC_REPORTS_DONE_NEXT9.json'", "'JOINT_SOLVER_AUDIT_NEXT9.json','SCIENTIFIC_RESULTS_NEXT9.json','SCIENTIFIC_REPORTS_DONE_NEXT9.json'")
source=source.replace("    visual=read_json", "    assert read_json(BASE/'JOINT_SOLVER_AUDIT_NEXT9.json')['status']=='PASS'\n    visual=read_json")
admit=ROOT/'scripts/next9/admit_next9_final_audit_v3.py';compile(source,str(admit),'exec');atomic(admit,source)
old_report=ROOT/'scripts/next9/finalize_next9_reports_v3.py'
source=old_report.read_text(encoding='utf-8').replace("required=['EXECUTION_AUDIT_NEXT9.json'", "required=['JOINT_SOLVER_AUDIT_NEXT9.json','EXECUTION_AUDIT_NEXT9.json'")
source=source.replace("            numerical_repair=", "            optimizer_repair_sha256=digest(BASE/'repairs/JOINT_SOLVER_CONTINUATION_REPAIR_NEXT9.json'),\n            numerical_repair=")
source=source.replace('`repairs/`와 실패 로그가 authority다.', '`repairs/`와 실패 로그가 authority다. M2/M3의 첫250회 최적화는 미수렴이었다. outer 점수를 열기 전에 TRAIN 경사 잔차로 진단하고 C=.1·동일 입력·동일 목적함수에서 한 차례 최대5,000회 추가 최적화를 허용했다. 수렴한 기존 모델은 재학습하지 않았고, 이전 모델·예측은 보존했다. 최종 전체 수렴 증거는 JOINT_SOLVER_AUDIT_NEXT9.json에 있다.')
report=ROOT/'scripts/next9/finalize_next9_reports_v4.py';compile(source,str(report),'exec');atomic(report,source)
atomic(BASE/'repairs/SOLVER_FINAL_AUDIT_GUIDANCE_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
    preferred_admission='scripts/next9/admit_next9_final_audit_v3.py',preferred_finalizer='scripts/next9/finalize_next9_reports_v4.py',
    source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [admit,report]},
    original_sources_preserved=True,reason='Require allfinaljointestimators convergence and discloseboundedoptimizercontinuation inhumanhandoff.'))

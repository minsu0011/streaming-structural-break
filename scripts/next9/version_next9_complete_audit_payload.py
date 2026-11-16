"""Include auditable clock/job pairs and preserved failed test sources in ZIP."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

old=ROOT/'src/next9/delivery_payload.py';source=old.read_text(encoding='utf-8')
source=source.replace("            elif tree=='portable_fixture'", "            elif tree=='repairs' and path.suffix=='.py':add(path)\n            elif tree=='portable_fixture'")
source=source.replace("    if final:\n", "    for path in (BASE/'runtime/jobs').glob('*.json'):add(path)\n    for name in ['pyproject.toml','pytest.ini']:\n        if (ROOT/name).is_file():add(ROOT/name)\n    if final:\n")
new=ROOT/'src/next9/delivery_payload_v2.py';compile(source,str(new),'exec');atomic(new,source)
source=(ROOT/'scripts/next9/reproduce_next9_archive_v3.py').read_text(encoding='utf-8').replace('from src.next9.delivery_payload import','from src.next9.delivery_payload_v2 import')
archive=ROOT/'scripts/next9/reproduce_next9_archive_v4.py';compile(source,str(archive),'exec');atomic(archive,source)
source=(ROOT/'scripts/next9/finalize_next9_reports_v4.py').read_text(encoding='utf-8').replace('reproduce_next9_archive_v3','reproduce_next9_archive_v4')
source=source.replace('## 재현과 제한','## 재현과 제한\n\n실제 사용한 Python·라이브러리 버전은 REPRODUCTION_ENVIRONMENT_NEXT9.json과 REPRODUCTION_REQUIREMENTS_NEXT9.txt에 기록했다. 패키지를 새로 설치하지 않았으며 외부 배포 가능 여부까지 검증한 것은 아니다.')
report=ROOT/'scripts/next9/finalize_next9_reports_v5.py';compile(source,str(report),'exec');atomic(report,source)
atomic(BASE/'repairs/COMPLETE_AUDIT_PAYLOAD_V2_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
    reason='Include both CPU-backed ledger and identical runtimejob receipts so exported work_time snapshot is auditable; include preserved failedtest.py outside active testcollection.',
    old_sources_preserved=True,new_data_or_fitting=False,source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [new,archive,report]},
    preferred_archive_script='scripts/next9/reproduce_next9_archive_v4.py',preferred_finalizer='scripts/next9/finalize_next9_reports_v5.py',
    use_short_five_character_archive_tags=True))

"""Keep immutable v1 report implementation and name tested v3 entry points."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,digest,utc_now

old=ROOT/'scripts/next9/finalize_next9_reports.py'
source=old.read_text(encoding='utf-8').replace('reproduce_next9_archive --final','reproduce_next9_archive_v3 --final')
source=source.replace('## 재현과 제한','## 재현과 제한\n\n현재 추론 진입점은 `scripts/next9/infer_next9_v3.py`, loader는 `src/next9/runtime_loading_v3.py`다. 전체 ZIP 재현은 `scripts/next9/reproduce_next9_archive_v3.py`를 사용한다. 과거 버전은 실패와 수정의 증거로 함께 보존했다.')
source=source.replace('로그손실 상한 반올림 문제를 구분해 기록했다.','로그손실 상한 반올림, M7 shared/dedicated 경로의 M5 입력 정밀도 문제를 구분해 기록했다.')
path=ROOT/'scripts/next9/finalize_next9_reports_v3.py';compile(source,str(path),'exec');atomic(path,source)
atomic(BASE/'repairs/FINAL_RUNTIME_GUIDANCE_V3_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
    original_source_sha256=digest(old),preferred_finalizer='scripts/next9/finalize_next9_reports_v3.py',
    preferred_source_sha256=digest(path),reason='Human handoff must name the testedv3runtime,not preservedpreviousversions. Noresultorcriteriamodification.'))

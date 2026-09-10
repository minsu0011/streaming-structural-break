from datetime import datetime,timezone,timedelta
import json
import pytest
from src.next.research_clock import ResearchClock,ResearchTimeError
from src.utils.artifacts import write_json


def _session(root,elapsed):
    path = root/'artifacts/next/SESSION_NEXT.json'
    path.parent.mkdir(parents=True)
    write_json(path,{'research_start_utc':(datetime.now(timezone.utc)-timedelta(seconds=elapsed)).isoformat(),
        'minimum_research_seconds':36000,'additional_final_audit_min_seconds':2700})


def test_early_final_audit_and_handoff_are_rejected(tmp_path):
    _session(tmp_path,8000)
    clock = ResearchClock(tmp_path)
    with pytest.raises(ResearchTimeError):
        clock.begin_final_audit()
    with pytest.raises(ResearchTimeError):
        clock.assert_handoff_ready()
    resumed = ResearchClock(tmp_path)
    assert resumed.snapshot()['research_elapsed_seconds']>=8000
    assert clock.anchor==resumed.anchor


def test_ten_hours_does_not_include_the_additional_audit(tmp_path):
    _session(tmp_path,36001)
    clock = ResearchClock(tmp_path)
    clock.begin_final_audit()
    with pytest.raises(ResearchTimeError):
        clock.assert_handoff_ready()


def test_original_start_cannot_silently_reset_on_resume(tmp_path):
    _session(tmp_path,8000)
    clock = ResearchClock(tmp_path)
    path = clock.session_path
    session = json.loads(path.read_text(encoding='utf-8'))
    session['research_start_utc']=datetime.now(timezone.utc).isoformat()
    write_json(path,session)
    with pytest.raises(ResearchTimeError):
        ResearchClock(tmp_path)

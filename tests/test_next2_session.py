import concurrent.futures
import datetime
import json
import time
import pytest
from src.next2 import session


def setup(tmp_path,seconds=120):
    out=tmp_path/'artifacts/next2'
    out.mkdir(parents=True)
    (tmp_path/'checkpoints').mkdir()
    now=datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=seconds)
    (out/'SESSION.json').write_text(json.dumps({'started_utc':now.isoformat(),
        'monotonic_start':time.monotonic()-seconds,'segments':[]}))
    return tmp_path


def test_six_hours_is_minimum_not_maximum(tmp_path):
    root=setup(tmp_path)
    result=session.clock(root)
    assert 120<=result['elapsed_research_seconds']<125
    with pytest.raises(RuntimeError,match='Six substantive'):
        session.require_research_complete(root)


def test_reboot_or_clock_discontinuity_fails_closed(tmp_path):
    root=setup(tmp_path)
    p=root/'artifacts/next2/SESSION.json'
    r=json.loads(p.read_text()); r['monotonic_start']-=600
    p.write_text(json.dumps(r))
    with pytest.raises(RuntimeError,match='discontinuity'):
        session.clock(root)


def test_concurrent_checkpoints_never_replace_each_other(tmp_path,monkeypatch):
    root=setup(tmp_path)
    monkeypatch.setattr(session,'commit_id',lambda:None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(session.checkpoint,'task '+str(j),['next'],root=root) for j in range(12)]
        for f in futures:
            f.result()
    files=list((root/'checkpoints').glob('NEXT2_*.json'))
    assert len(files)==12
    assert {json.loads(p.read_text())['task'] for p in files}=={'task '+str(j) for j in range(12)}
    assert json.loads((root/'artifacts/next2/LATEST_CHECKPOINT.json').read_text())['task'].startswith('task ')


def test_audit_time_is_separate_and_cannot_be_reset(tmp_path):
    root=setup(tmp_path,seconds=21601)
    result=session.begin_final_audit(root)
    assert result['minimum_research_met'] and result['audit_started']
    assert result['elapsed_audit_seconds']<1
    with pytest.raises(RuntimeError,match='45 separate'):
        session.require_handoff_time(root)
    with pytest.raises(RuntimeError,match='cannot be reset'):
        session.begin_final_audit(root)


def test_recovery_credits_only_last_checkpoint(tmp_path,monkeypatch):
    root=setup(tmp_path)
    monkeypatch.setattr(session,'commit_id',lambda:None)
    record=session.checkpoint('before interruption',['resume'],root=root)
    result=session.recover_research_from_checkpoint(root)
    assert 0<=result['elapsed_research_seconds']-record['elapsed_research_seconds']<1
    stored=json.loads((root/'artifacts/next2/SESSION.json').read_text())
    assert stored['segments'][0]['active_seconds']==record['elapsed_research_seconds']
    assert stored['original_research_started_utc']

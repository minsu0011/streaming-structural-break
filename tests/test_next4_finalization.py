from datetime import datetime,timezone
import copy
import pytest
from src.next4.finalization import require_timing


def fixture():
    stamp=100000.
    account={'productive_research_seconds':21600.,'audit_productive_seconds':2700.,
        'unobserved_seconds':600.,'wall_seconds':24900.,'wall_timestamp':stamp,
        'timestamp':datetime.fromtimestamp(stamp,timezone.utc).isoformat()}
    session={'audit_started_utc':datetime.fromtimestamp(stamp-2800,timezone.utc).isoformat()}
    return account,session,stamp+10


def test_real_research_then_separate_audit_passes():
    account,session,now=fixture();result=require_timing(account,session,now_wall=now)
    assert result['productive_research_seconds']==21600 and result['audit_productive_seconds']==2700


def test_wall_time_cannot_replace_productive_research_or_audit():
    account,session,now=fixture()
    for field in ('productive_research_seconds','audit_productive_seconds'):
        changed=copy.deepcopy(account);changed[field]-=1
        with pytest.raises(RuntimeError):require_timing(changed,session,now_wall=now)


def test_old_or_future_accounting_is_rejected():
    account,session,now=fixture()
    for current in (account['wall_timestamp']+61,account['wall_timestamp']-1):
        with pytest.raises(RuntimeError):require_timing(account,session,now_wall=current)


def test_fabricated_overlap_and_nonfinite_values_are_rejected():
    account,session,now=fixture()
    with pytest.raises(RuntimeError):require_timing(account,{},now_wall=now)
    changed=copy.deepcopy(account);changed['audit_productive_seconds']=2900
    with pytest.raises(RuntimeError):require_timing(changed,session,now_wall=now)
    changed=copy.deepcopy(account);changed['productive_research_seconds']=float('nan')
    with pytest.raises(ValueError):require_timing(changed,session,now_wall=now)

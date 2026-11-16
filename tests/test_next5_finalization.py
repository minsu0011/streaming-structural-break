from datetime import datetime, timezone
import pytest
from src.next5.finalization import require_timing


def evidence(research=36000, audit=3000, audit_wall=3000):
    now = 1789120000.0
    return ({'productive_research_seconds': research, 'audit_productive_seconds': audit,
        'unobserved_seconds': 600, 'wall_seconds': research + audit + 600,
        'wall_timestamp': now, 'timestamp': datetime.fromtimestamp(now, timezone.utc).isoformat()},
        {'audit_started_utc': datetime.fromtimestamp(now - audit_wall, timezone.utc).isoformat()}, now)


def test_research_and_separate_observed_audit_both_required():
    account, session, now = evidence()
    assert require_timing(account, session, now_wall=now)['separate_audit_minimum_pass']
    account['productive_research_seconds'] = 35999.99
    with pytest.raises(RuntimeError, match='36000'):
        require_timing(account, session, now_wall=now)
    account['productive_research_seconds'] = 36000
    account['audit_productive_seconds'] = 2999.99
    with pytest.raises(RuntimeError, match='3000'):
        require_timing(account, session, now_wall=now)


def test_stale_clock_and_backdated_audit_cannot_pass():
    account, session, now = evidence()
    with pytest.raises(RuntimeError, match='Fresh'):
        require_timing(account, session, now_wall=now + 61)
    account['audit_productive_seconds'] = 3100
    with pytest.raises(RuntimeError, match='predates'):
        require_timing(account, session, now_wall=now)

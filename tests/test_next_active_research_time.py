from datetime import datetime, timedelta, timezone
import pytest
from src.next.active_research_time import adjusted_seconds
from src.next.research_clock import ResearchTimeError


def test_interrupted_wall_time_cannot_satisfy_ten_hours():
    start = datetime(2026, 9, 8, 17, 10, 1, tzinfo=timezone.utc)
    result = adjusted_seconds(start, start+timedelta(hours=12),
        [(start+timedelta(hours=9), start+timedelta(hours=11, minutes=30))])
    assert result['wall_seconds'] == 43200
    assert result['substantive_research_seconds'] == 34200 < 36000
    assert result['excluded_seconds'] == 9000


def test_multiple_disjoint_interruptions_are_counted_once():
    start = datetime(2026, 9, 8, tzinfo=timezone.utc)
    result = adjusted_seconds(start, start+timedelta(hours=13), [
        (start+timedelta(hours=8), start+timedelta(hours=9)),
        (start+timedelta(hours=4), start+timedelta(hours=5))])
    assert result['substantive_research_seconds'] == 39600


def test_overlapping_or_out_of_window_exclusions_fail_closed():
    start = datetime(2026, 9, 8, tzinfo=timezone.utc)
    with pytest.raises(ResearchTimeError):
        adjusted_seconds(start, start+timedelta(hours=12), [
            (start+timedelta(hours=4), start+timedelta(hours=6)),
            (start+timedelta(hours=5), start+timedelta(hours=7))])
    with pytest.raises(ResearchTimeError):
        adjusted_seconds(start, start+timedelta(hours=12), [(start, start+timedelta(hours=13))])


def test_timezone_ambiguity_is_rejected():
    with pytest.raises(ResearchTimeError):
        adjusted_seconds('2026-09-08T17:10:01', '2026-09-09T05:10:01', [])

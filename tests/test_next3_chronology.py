from datetime import datetime, timezone
import pytest
from src.next3.chronology import exact_count, ordered_events, recorded_utc


@pytest.mark.parametrize('value', [None, 0, True, '', 'yesterday', '2026-09-10T01:00:00',
                                  '2026-09-10T10:00:00+09:00'])
def test_invalid_recorded_time_is_not_inferred(value):
    with pytest.raises(ValueError):
        recorded_utc(value)


def test_utc_spellings_and_subsecond_order():
    assert recorded_utc('2026-09-10T01:00:00Z') == datetime(2026, 9, 10, 1, tzinfo=timezone.utc)
    gaps = ordered_events([('registered', '2026-09-10T01:00:00.123456+00:00'),
                           ('fit_start', '2026-09-10T01:00:00.123457Z'),
                           ('stage_complete', '2026-09-10T01:00:02.123457Z')])
    assert [row['seconds'] for row in gaps] == [0.000001, 2]


def test_equal_recorded_timestamps_do_not_fabricate_strict_precedence():
    assert ordered_events([('first', '2026-09-10T01:00:00Z'),
                           ('second', '2026-09-10T01:00:00Z')])[0]['seconds'] == 0


@pytest.mark.parametrize('events', [[], [('a', '2026-09-10T01:00:00Z')],
    [('a', '2026-09-10T01:00:01Z'), ('b', '2026-09-10T01:00:00Z')],
    [('a', '2026-09-10T01:00:00Z'), ('a', '2026-09-10T01:00:01Z')],
    [('', '2026-09-10T01:00:00Z'), ('b', '2026-09-10T01:00:01Z')],
    [('a', '2026-09-10T01:00:00Z'), ('b', None)]])
def test_bad_event_chain_rejected(events):
    with pytest.raises(ValueError):
        ordered_events(events)


@pytest.mark.parametrize('value', [True, 1.0, '1', 0, 2, None])
def test_counter_does_not_accept_boolean_or_coerced_values(value):
    with pytest.raises(ValueError):
        exact_count(value, 1)


def test_exact_fit_counter():
    assert exact_count(1, 1) == 1

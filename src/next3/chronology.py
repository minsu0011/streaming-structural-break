"""Validate recorded UTC event order without inventing absent event times."""
from datetime import datetime, timedelta


def recorded_utc(value):
    if not isinstance(value, str) or not value:
        raise ValueError('A recorded UTC timestamp string is required')
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError('Invalid recorded UTC timestamp') from error
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError('Timestamp must explicitly use UTC')
    return result


def ordered_events(events):
    """Return gaps for named recorded events, rejecting absent or reversed times."""
    if len(events) < 2:
        raise ValueError('At least two recorded events are required')
    names = [name for name, _ in events]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError('Distinct nonempty event names are required')
    parsed = [recorded_utc(value) for _, value in events]
    gaps = []
    for index in range(1, len(events)):
        seconds = (parsed[index] - parsed[index - 1]).total_seconds()
        if seconds < 0:
            raise ValueError(f'Recorded chronology reversed: {names[index - 1]} -> {names[index]}')
        gaps.append({'earlier': names[index - 1], 'later': names[index], 'seconds': seconds})
    return gaps


def exact_count(value, expected):
    if type(value) is not int or value != expected:
        raise ValueError(f'Expected exact integer count {expected}')
    return value

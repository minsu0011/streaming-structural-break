"""Complete, coordinate-bound synthetic golden hashes; no prediction arrays."""
from pathlib import Path
import csv
import re

FIELDS = {'scenario', 'kind', 'synthetic_row_index', 'historical_points', 'online_points',
          'input_sha256', 'float32_prediction_sha256', 'matches_original_saved_output', 'retained_numeric_bytes'}


def exact_integer(value, *, minimum=0):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Canonical integer text required') from error
    if not isinstance(value, str) or str(number) != value or number < minimum:
        raise ValueError('Canonical integer text required')
    return number


def read_golden_cases(path, specifications):
    """Reject truncated, duplicate, misaligned or non-exact prior result rows."""
    expected = {}
    for kind, scenario, count, horizon, partition in specifications:
        if (type(count) is not int or count <= 0 or type(horizon) is not int or horizon <= 0
                or type(partition) is not int or partition < 0):
            raise ValueError('Fixed positive scenario dimensions required')
        for index in range(count):
            key = (kind, scenario, index)
            if key in expected: raise ValueError('Duplicate scenario specification')
            expected[key] = horizon
    if not expected: raise ValueError('Nonempty fixed scenario specifications required')
    records = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS) or set(reader.fieldnames) != FIELDS:
            raise ValueError('Exact golden case columns required')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed golden CSV row')
            key = (row['kind'], row['scenario'], exact_integer(row['synthetic_row_index']))
            if key not in expected or key in records: raise ValueError('Duplicate or unexpected golden coordinate')
            if (exact_integer(row['historical_points']) != 2048
                    or exact_integer(row['online_points']) != expected[key]
                    or exact_integer(row['retained_numeric_bytes']) != 16925):
                raise ValueError('Golden dimensions or numeric-state contract differs')
            if row['matches_original_saved_output'] != 'True': raise ValueError('Golden output was not exact')
            if any(re.fullmatch('[0-9a-f]{64}', row[name]) is None
                   for name in ('input_sha256', 'float32_prediction_sha256')):
                raise ValueError('Canonical SHA256 fields required')
            records[key] = row
    if set(records) != set(expected): raise ValueError('Golden case coverage is incomplete')
    return records

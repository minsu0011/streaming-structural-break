import csv
import pytest
from src.next3.long_horizon_contract import FIELDS, exact_integer, read_golden_cases

PLAN = [('long_null', 'N0', 2, 16384, 2)]


def records():
    return [dict(scenario='N0', kind='long_null', synthetic_row_index=str(index),
        historical_points='2048', online_points='16384', input_sha256='1'*64,
        float32_prediction_sha256='2'*64, matches_original_saved_output='True',
        retained_numeric_bytes='16925') for index in range(2)]


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(FIELDS)); writer.writeheader(); writer.writerows(rows)


def test_complete_coordinate_map_accepts_physical_csv_reordering(tmp_path):
    path = tmp_path/'golden.csv'; write(path, list(reversed(records())))
    result = read_golden_cases(path, PLAN)
    assert set(result) == {('long_null','N0',0),('long_null','N0',1)}


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_scenario', 'wrong_horizon',
                                  'state_growth', 'not_exact', 'bad_hash', 'padded_index'])
def test_tampered_or_incomplete_golden_rejected(tmp_path, change):
    rows = records()
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows[1] = dict(rows[0])
    elif change == 'wrong_scenario': rows[1]['scenario'] = 'N1'
    elif change == 'wrong_horizon': rows[1]['online_points'] = '1024'
    elif change == 'state_growth': rows[1]['retained_numeric_bytes'] = '16926'
    elif change == 'not_exact': rows[1]['matches_original_saved_output'] = 'False'
    elif change == 'bad_hash': rows[1]['input_sha256'] = 'a'*63
    elif change == 'padded_index': rows[1]['synthetic_row_index'] = '01'
    path = tmp_path/'golden.csv'; write(path, rows)
    with pytest.raises(ValueError): read_golden_cases(path, PLAN)


@pytest.mark.parametrize('value', [True, 1, 1.0, '1.0', '-1', '+1', ' 1', None])
def test_noncanonical_integer_fields_are_rejected(value):
    with pytest.raises(ValueError): exact_integer(value)


def test_duplicate_plan_does_not_hide_expected_cases(tmp_path):
    path = tmp_path/'golden.csv'; write(path, records())
    with pytest.raises(ValueError, match='Duplicate scenario'):
        read_golden_cases(path, PLAN+PLAN)


def test_extra_csv_field_is_rejected(tmp_path):
    path = tmp_path/'golden.csv'; write(path, records())
    with path.open('a', encoding='utf-8') as stream: stream.write('extra,fields\n')
    with pytest.raises(ValueError): read_golden_cases(path, PLAN)

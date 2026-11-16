"""Original byte history and final evidence dates cannot be weakened by the auditor."""
from pathlib import Path
import datetime
import hashlib
import pytest
from src.next3.artifacts import ROOT, read_json
from src.next3.final_core_contract import literal_policy, protected_files, registered_hypothesis, require_fresh


@pytest.mark.parametrize('name,structured', [('S1_LOWER_SEMISCALE', True),
    ('S2_LOWER_QUANTILE', True), ('R2_LOGISTIC_GATE', False)])
def test_actual_locked_hypothesis_schemas(name, structured):
    hypothesis = read_json(ROOT/'artifacts/next3/hypotheses'/f'{name}.json')
    result = registered_hypothesis(hypothesis)
    assert result == {'required_keys_checked': 8, 'structured_change': structured}


@pytest.mark.parametrize('key,value', [('HYPOTHESIS', ''), ('BOTTLENECK', {}),
    ('CHANGE', {}), ('CHANGE', {'kind': ''}), ('CHANGE', {'kind': 'test', 'value': float('nan')}),
    ('CHANGE', []), ('STOP_RULE', None), ('PRIMARY_METRIC', True)])
def test_empty_or_malformed_hypothesis_does_not_pass(key, value):
    hypothesis = read_json(ROOT/'artifacts/next3/hypotheses/S1_LOWER_SEMISCALE.json')
    hypothesis[key] = value
    with pytest.raises(RuntimeError):
        registered_hypothesis(hypothesis)


def originals(tmp_path):
    ledger = b'name,value\r\nold,1\r\n'; model = b'frozen model bytes'
    (tmp_path/'EXPERIMENT_LEDGER.csv').write_bytes(ledger)
    (tmp_path/'model.bin').write_bytes(model)
    mapping = {name: hashlib.sha256(value).hexdigest() for name, value in
        [('EXPERIMENT_LEDGER.csv', ledger), ('model.bin', model)]}
    return ledger, mapping


def test_exact_old_files_and_ledger_pass(tmp_path):
    ledger, mapping = originals(tmp_path); result = protected_files(tmp_path, mapping, ledger)
    assert result['protected_files_checked'] == 2 and result['append_only_exceptions'] == []


def test_only_exact_byte_prefix_append_is_permitted(tmp_path):
    ledger, mapping = originals(tmp_path); (tmp_path/'EXPERIMENT_LEDGER.csv').write_bytes(ledger+b'new,2\r\n')
    result = protected_files(tmp_path, mapping, ledger)
    assert len(result['append_only_exceptions']) == 1


def test_newline_normalization_of_old_history_is_rejected(tmp_path):
    ledger, mapping = originals(tmp_path); (tmp_path/'EXPERIMENT_LEDGER.csv').write_bytes(ledger.replace(b'\r\n', b'\n'))
    with pytest.raises(RuntimeError, match='prefix'): protected_files(tmp_path, mapping, ledger)


def test_forged_shorter_prefix_does_not_pass(tmp_path):
    ledger, mapping = originals(tmp_path)
    with pytest.raises(ValueError, match='frozen checksum'): protected_files(tmp_path, mapping, ledger[:3])


def test_nonledger_change_is_rejected(tmp_path):
    ledger, mapping = originals(tmp_path); (tmp_path/'model.bin').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='changed'): protected_files(tmp_path, mapping, ledger)


def test_removed_protected_file_is_rejected(tmp_path):
    ledger, mapping = originals(tmp_path); (tmp_path/'model.bin').unlink()
    with pytest.raises(RuntimeError, match='missing'): protected_files(tmp_path, mapping, ledger)


def test_linked_protected_file_is_rejected(tmp_path, monkeypatch):
    ledger, mapping = originals(tmp_path); target = tmp_path/'model.bin'; original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda path: path == target or original(path))
    with pytest.raises(RuntimeError, match='link'): protected_files(tmp_path, mapping, ledger)


def test_current_preregistered_policy_passes_literal_review():
    policy = read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    assert literal_policy(policy)['literal_numeric_or_boolean_gate_fields_checked'] == 30


@pytest.mark.parametrize('group,key,value', [
    ('performance','mean_delta_min',.0014), ('generation','mean_delta_min',.002),
    ('robustness','ARCH_scale_down_delta_min_each',.009), ('engineering','private_state_ratio_max',2.1),
    ('performance','ci_lower_positive_required',1), ('performance','median_delta_min',False)])
def test_relaxed_or_mistyped_gate_is_rejected(group, key, value):
    policy = read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'); policy[group][key] = value
    with pytest.raises(RuntimeError, match='policy changed'): literal_policy(policy)


def test_relabeling_conditional_uncertainty_is_rejected():
    policy = read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'); policy['bootstrap']['selection_adjusted'] = True
    with pytest.raises(RuntimeError, match='selection-adjusted'): literal_policy(policy)


def test_outer_validation_training_is_rejected():
    policy = read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'); policy['nested_cross_fit']['outer_validation_used_for_training'] = True
    with pytest.raises(RuntimeError, match='Outer validation'): literal_policy(policy)


def test_freshness_accepts_only_actual_audit_interval():
    now = datetime.datetime(2026,9,10,12,0,tzinfo=datetime.timezone.utc)
    require_fresh({'created_utc':'2026-09-10T11:55:00+00:00'}, '2026-09-10T11:41:00+00:00', now=now)


@pytest.mark.parametrize('created', ['2026-09-10T11:40:00+00:00', '2026-09-10T12:01:00+00:00', '2026-09-10T11:55:00'])
def test_preaudit_future_or_naive_evidence_is_rejected(created):
    now = datetime.datetime(2026,9,10,12,0,tzinfo=datetime.timezone.utc)
    with pytest.raises(RuntimeError): require_fresh({'created_utc':created}, '2026-09-10T11:41:00+00:00', now=now)

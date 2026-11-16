from pathlib import Path
import pytest
from src.next3.path_budget import check_windows_copy_path_budget, utf16_units


def test_actual_failure_member_is_preserved_under_shorter_staging_root(tmp_path):
    name = 'artifacts/next3/maintenance/tree_certificate_scalar_oracle_correction_001/original_sources/artifacts/next3/maintenance/TREE_STRUCTURAL_AUDIT_IMPLEMENTATION_LOCK.json'
    # A concrete root depth reproduces the observed limit independently of temp depth.
    root = Path(tmp_path.anchor)/('x'*88)
    with pytest.raises(ValueError,match='before copying'):
        check_windows_copy_path_budget(root/'maintenance_dependency_coverage_v1/payload',[name])
    short = Path(tmp_path.anchor)/'sb3'
    proof = check_windows_copy_path_budget(short,[name])
    assert proof['members_checked'] == 1 and proof['longest_member'] == name
    assert proof['maximum_file_UTF16_units'] < 259


def test_supplementary_unicode_counts_as_two_windows_code_units():
    assert utf16_units('abc') == 3
    assert utf16_units('\uD55C') == 1
    assert utf16_units('\U0001f332') == 2


def test_file_budget_includes_the_absolute_root_and_is_checked_before_io(tmp_path):
    root = tmp_path/'pending'
    available = 259-utf16_units(root)-1
    assert available > 10
    proof = check_windows_copy_path_budget(root,['a'*available])
    assert proof['maximum_file_UTF16_units'] == 259
    with pytest.raises(ValueError,match='file_UTF16_units=260'):
        check_windows_copy_path_budget(root,['a'*(available+1)])
    assert not root.exists()


def test_directory_limit_is_checked_even_with_a_short_filename(tmp_path):
    root = tmp_path/'pending'
    component = 'a'*(248-utf16_units(root)-1)
    with pytest.raises(ValueError,match='directory_UTF16_units=248'):
        check_windows_copy_path_budget(root,[component+'/x'])
    assert not root.exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a\\b', 'NUL.txt', 'a:stream'])
def test_budget_does_not_relax_member_name_safety(tmp_path,name):
    with pytest.raises(ValueError): check_windows_copy_path_budget(tmp_path,[name])

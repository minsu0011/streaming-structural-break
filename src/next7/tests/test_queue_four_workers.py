"""A fourth slot preserves live ownership and high-memory admission limits."""
import pytest
from src.next7.tests.test_queue_three_workers import (
    test_ready_tasks_before_live_owners_never_exceed_available_slots as _check_owners,
)


@pytest.mark.parametrize('existing,capacity,new_expected', [(3,4,1),(4,4,0),(4,1,0)])
def test_four_slots_preserve_existing_owners_and_repeat_dispatch(
        tmp_path, monkeypatch, existing, capacity, new_expected):
    _check_owners(tmp_path, monkeypatch, existing, capacity, new_expected)

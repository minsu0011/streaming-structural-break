import pytest
from src.next.nested_menu import unique_training_tasks, select_inner_candidate


def test_each_outer_inner_pair_has_one_disjoint_shared_fit():
    tasks = unique_training_tasks()
    assert len(tasks) == 10
    covered = set()
    for task in tasks:
        held, train = set(task['held_out_folds']), set(task['training_folds'])
        assert len(held) == 2 and len(train) == 3
        assert not held & train and held | train == set(range(5))
        left, right = task['held_out_folds']
        covered.update(((left, right), (right, left)))
    assert covered == {(outer, inner) for outer in range(5) for inner in range(5) if outer != inner}


def test_inner_choice_requires_complete_menu_and_prefers_stability_on_tie():
    assert select_inner_candidate({'volatile': [.3, .7, .5, .5], 'stable': [.5, .5, .5, .5]}) == 'stable'
    assert select_inner_candidate({'z': [.6]*4, 'a': [.6]*4}) == 'a'
    with pytest.raises(ValueError, match='four'):
        select_inner_candidate({'incomplete': [.7]*3})

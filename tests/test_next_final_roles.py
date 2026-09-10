from scripts.freeze_next_roles import select_roles


def row(name, mean, median, worst, point, *, promoted=True, marginal=True, std=.01):
    return {'id': name, 'mean': mean, 'median': median, 'worst': worst, 'point_us': point,
        'amortized_us': point+10, 'std': std, 'decision': {'promoted': promoted}, 'marginal_complexity_gate_pass': marginal}


def test_frozen_role_order_and_distinctness():
    rows = [row('highest_mean', .630, .629, .620, 8),
        row('highest_worst', .627, .627, .623, 7),
        row('fast', .625, .625, .619, 4),
        row('too_weak_fast', .619, .618, .609, 1),
        row('missing_engineering', .650, .650, .640, 3, promoted=False),
        row('marginally_redundant', .640, .640, .630, 5, marginal=False)]
    assert select_roles(rows) == {'PRIMARY_DEV_CANDIDATE': 'highest_mean',
        'ROBUST_DEV_CANDIDATE': 'highest_worst', 'FAST_DEV_CANDIDATE': 'fast'}


def test_robust_median_floor_and_no_invented_backups():
    rows = [row('primary', .630, .630, .620, 8), row('median_fails', .628, .624, .622, 5)]
    assert select_roles(rows) == {'PRIMARY_DEV_CANDIDATE': 'primary',
        'ROBUST_DEV_CANDIDATE': None, 'FAST_DEV_CANDIDATE': 'median_fails'}
    assert select_roles(rows[:1]) == {'PRIMARY_DEV_CANDIDATE': 'primary',
        'ROBUST_DEV_CANDIDATE': None, 'FAST_DEV_CANDIDATE': None}


def test_primary_tie_uses_worst_then_median_then_lexical():
    rows = [row('b', .630, .631, .620, 8), row('a', .630, .631, .620, 8),
        row('higher_median_lower_worst', .630, .632, .619, 8)]
    assert select_roles(rows)['PRIMARY_DEV_CANDIDATE'] == 'a'

"""Evidence-linked priorities and immutable decisions for a small research queue."""
from src.next3.artifacts import BASE, ROOT, atomic_json, read_json, utc_now, digest
from src.next3.maintenance import append_csv


def priority(item):
    numerator = (item['estimated_metric_impact'] * item['confidence'] *
                 item['unresolved_severity'] * item['achievable_improvement'])
    denominator = (1 + item['overfit_risk']) * (1 + item['engineering_cost']) * (1 + item['runtime_cost'])
    return numerator/denominator


def record_board(items, reason, *, initial=False):
    ordered = sorted(items, key=priority, reverse=True)
    for index, item in enumerate(ordered):
        item['priority'] = priority(item)
        item['rank'] = index+1
    assert sum(item['status'] == 'ACTIVE' for item in ordered) <= 2
    report = {'created_utc': utc_now(), 'reason': reason, 'bottlenecks': ordered,
        'priority_units': 'Decision heuristic, not a measured attainable TS-AUC or posterior probability',
        'primary_objective': 'Official competition TS-AUC; alarm FPR is secondary',
        'maximum_active_bottlenecks': 2}
    directory = BASE/'bottlenecks/history'
    number = len(list(directory.glob('*.json')))+1 if directory.exists() else 1
    atomic_json(directory/f'{number:03d}.json', report, immutable=True)
    atomic_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json', report)
    if initial:
        atomic_json(ROOT/'BOTTLENECK_BOARD_INITIAL.json', report, immutable=True)
    for item in ordered:
        append_csv(BASE/'bottlenecks/BOTTLENECK_HISTORY.csv', {
            'timestamp': report['created_utc'], 'review': number, 'id': item['id'],
            'rank': item['rank'], 'priority': item['priority'], 'status': item['status'],
            'reason': reason, 'next_test': item['next_test']})
    return report

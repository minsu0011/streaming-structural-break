"""Immutable decisions with at most two active research priorities."""
from src.next4.artifacts import BASE,ROOT,read_json,atomic_json,utc_now
from src.next4.scheduler import append_csv


def record_board(items,reason,*,initial=False):
    if len({x['id'] for x in items})!=len(items):raise ValueError('Unique bottleneck IDs required')
    if sum(x['status']=='ACTIVE' for x in items)>2:raise ValueError('At most two active bottlenecks')
    for item in items:
        item['priority']=(item['expected_impact']*item['evidence_strength']*item['solvability']/
            ((1+item['overfit_risk'])*(1+item['compute'])*(1+item['complexity'])))
    ordered=sorted(items,key=lambda x:x['priority'],reverse=True)
    for rank,item in enumerate(ordered,1):item['rank']=rank
    directory=BASE/'bottlenecks/history';number=len(list(directory.glob('*.json')))+1
    result={'created_utc':utc_now(),'reason':reason,'maximum_active':2,'bottlenecks':ordered,
        'priority_scope':'Subjective decision heuristic, not a measured or attainable TS-AUC gain'}
    atomic_json(directory/f'{number:03d}.json',result,immutable=True)
    atomic_json(BASE/'bottlenecks/BOARD.json',result)
    if initial:atomic_json(ROOT/'BOTTLENECK_BOARD_INITIAL_NEXT4.json',result,immutable=True)
    for item in ordered:append_csv(BASE/'bottlenecks/HISTORY.csv',{'timestamp':result['created_utc'],
        'review':number,'id':item['id'],'status':item['status'],'priority':item['priority'],'reason':reason,'next_test':item['next_test']})
    return result

"""Conservative NEXT5 descriptive saturation and hard menu limits."""
from src.next5.artifacts import BASE, read_json


def assert_budget():
    registrations = [read_json(p) for p in (BASE/'registrations').glob('*.json')]
    families = {r['family'] for r in registrations}
    screens = list((BASE/'experiments').glob('*/screen/START.json'))
    full = list((BASE/'experiments').glob('*/full/START.json'))
    shadow = BASE/'shadow/FINALIST_MANIFEST_NEXT5.json'
    finalists = read_json(shadow).get('finalists', []) if shadow.exists() else []
    if len(families)>7 or len(screens)>35 or len(full)>20 or len(finalists)>3:
        raise RuntimeError('NEXT5 hard scientific budget exceeded')
    return {'family_count': len(families), 'screen_count': len(screens),
            'full_count': len(full), 'shadow_candidate_count': len(finalists)}


def assess(full_results, screen_results, bootstrap_results, policy):
    full = {r['candidate']: r for r in full_results if r.get('status') == 'COMPLETE' and r.get('scope') == 'full'}
    failures = {r['candidate'] for r in screen_results if r.get('status') == 'COMPLETE' and r.get('screen_pass') is False}
    finished = set(full)|failures
    best = max((r['mean_delta'] for r in full.values()), default=None)
    gate = policy['saturation']
    exhausted = len(finished)>=gate['structural_hypotheses_min'] and (best is None or best<gate['best_gain_strict_below'])
    return {'completed_structural_hypotheses': len(finished), 'best_full_gain': best,
            'risk': 'HIGH' if exhausted else 'MONITOR', 'case_H': exhausted,
            'stop_new_scientific_fits': exhausted, 'budget': assert_budget(),
            'scope': 'ResearchPool only. Existing DEV history and prospective Shadow limits remain explicit.'}

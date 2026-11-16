"""Recover omitted cohort identity from frozen loop order and scalar receipts."""
import csv
from src.next5.artifacts import BASE, ROOT, read_json, digest
from src.next5.synthetic import cohort_menu, cohort_key


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def annotate_paired_alarms(path, original_rows):
    if path.name != 'PAIRED_ALARMS.csv':
        raise ValueError('Only the ambiguous paired alarm table needs cohort annotation')
    analysis = path.parent
    receipt = read_json(analysis / 'RESULTS.json')
    assert digest(path) == receipt['files_sha256'][path.name]
    menu = read_json(BASE / 'synthetic/MENU_LOCK.json')
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    assert digest(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json') == menu['original_generator_policy_sha256']
    point_tables = {}
    for kind, name in [('counterexample', 'COUNTEREXAMPLE_ALARMS.csv'), ('long_null', 'LONG_NULL_RESULTS.csv')]:
        source = analysis / name
        assert digest(source) == receipt['files_sha256'][name]
        point_tables[kind] = rows(source)
    proofs = {r['cohort']: r for r in receipt['cohort_sources']}
    roles = ('BL80', 'B2', 'D2') if receipt['scope'] == 'full' else ('BL80',)
    expected = []
    for spec in cohort_menu(menu, lock):
        kind = spec['kind']
        if kind == 'calibration':
            continue
        key = cohort_key(spec)
        source_receipt = read_json(analysis.parent / key / 'RESULTS.json')
        assert source_receipt['identity']['cohort'] == spec
        assert source_receipt['prediction_file_sha256'] == proofs[key]['candidate_sha256']
        assert source_receipt['identity']['reference_prediction_sha256'] == proofs[key]['reference_sha256']
        horizons = menu['long_null_horizons'] if kind == 'long_null' else [menu['counterexample_horizon']]
        for horizon in horizons:
            for alpha in menu['threshold_alphas']:
                matching = [r for r in point_tables[kind] if r['model'] == 'candidate'
                    and r['scenario'] == spec['scenario'] and int(r['horizon']) == horizon and float(r['alpha']) == alpha]
                if len(matching) != 1:
                    raise ValueError('Ambiguous scalar source inside an explicitly identified cohort')
                point = matching[0]
                events = (['null_eventual'] if int(point['null_series']) > 0 else [])
                if kind == 'counterexample' and int(point['break_series']) > 0:
                    events += ['prebreak', 'postbreak_detection']
                for event in events:
                    for role in roles:
                        expected.append({'candidate': receipt['candidate'], 'scope': receipt['scope'],
                            'fold': receipt['fold'], 'scenario': spec['scenario'], 'reference': role,
                            'horizon': horizon, 'alpha': alpha, 'event': event,
                            'cohort_kind': kind, 'cohort_key': key,
                            'candidate_cohort_prediction_sha256': proofs[key]['candidate_sha256'],
                            'reference_cohort_prediction_sha256': proofs[key]['reference_sha256']})
    if len(expected) != len(original_rows):
        raise ValueError('Original paired alarm rows do not match the frozen loop coverage')
    output = []
    for actual, identity in zip(original_rows, expected):
        for name in ('candidate', 'scope', 'scenario', 'reference', 'event'):
            if actual[name] != identity[name]:
                raise ValueError('Original paired-alarm row order does not match frozen cohort loop: ' + name)
        for name in ('fold', 'horizon', 'alpha'):
            if float(actual[name]) != identity[name]:
                raise ValueError('Original paired-alarm coordinate differs: ' + name)
        output.append({**actual, **{k: identity[k] for k in ('cohort_kind', 'cohort_key',
            'candidate_cohort_prediction_sha256', 'reference_cohort_prediction_sha256')}})
    keys = [tuple(r[k] for k in ('candidate', 'scope', 'fold', 'cohort_kind', 'scenario', 'reference', 'horizon', 'alpha', 'event')) for r in output]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate identity remains after cohort annotation')
    return output

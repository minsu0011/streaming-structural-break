"""Late genuine breaks for every fixed P-family PRIMARY learner artifact."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from numba import set_num_threads
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.synthetic import cohort_menu, cohort_key, scenario_background, background_mapping, alarm_summary
from src.next5.synthetic_metrics import load_cohort, score_draws
from src.next5.late_break import onset_mask, stratum_event_draws
from src.next5.bootstrap import paired_summary
from src.next5.scheduler import register_job

CANDIDATES = ('P1_TWO_TIMESCALE_CONFIRMATION', 'P2_SOFT_PERSISTENCE_COUNT', 'P3_SHOCK_RECOVERY')


def write_table(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--candidate', choices=CANDIDATES)
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    parent = BASE / 'synthetic_late_break'
    if args.register:
        sources = ('scripts/analyze_next5_late_break.py', 'src/next5/late_break.py',
                   'tests/test_next5_late_break.py', 'src/next5/synthetic_metrics.py', 'src/next5/synthetic.py')
        atomic_json(parent / 'POLICY_LOCK.json', {
            'created_utc': utc_now(), 'candidates': CANDIDATES, 'folds': list(range(5)),
            'onset': 512, 'horizon': 1024, 'scenarios': 'All 15 genuine-change scenarios already in the fixed menu',
            'reason': 'User-required P-family late-break check: lower outlier persistence must not conceal loss of late genuine changes',
            'seeds': 'Existing generating-seed indices whose frozen tau-cycle assignment is 512; retain both matched paths',
            'thresholds': 'Unchanged model-specific independent 512-series calibration from the existing complete fold analysis',
            'bootstrap': 'Original 2000 global generating-seed multiplicities, restricted to the same seed columns; no new draw or normalization',
            'support': '85 generating seeds per late stratum; below the descriptive 100-seed flag, no promotion authority',
            'timing': 'Endpoint stratum specified after initial aggregate P1 fold-0 stress was observed; prior user request required late-break diagnostics',
            'no_score_selected_scenario_omission': True, 'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
            'menu_sha256': digest(BASE / 'synthetic/MENU_LOCK.json'),
            'seed_counts_lock_sha256': digest(BASE / 'synthetic/SEED_COUNTS_LOCK.json'),
            'sources_sha256': {name: digest(ROOT / name) for name in sources},
        }, immutable=True)
        register_job({'id': 'LATE_BREAK_CONTRACTS', 'kind': 'validation', 'priority': 35,
            'argv': ['-m', 'pytest', 'tests/test_next5_late_break.py', '-q',
                '--junitxml=artifacts/next5/synthetic_late_break/CONTRACTS.xml'],
            'description': 'Late-onset paired seed selection and variable eligible bootstrap denominator', 'retry_safe': False})
        for cid in CANDIDATES:
            for fold in range(5):
                register_job({'id': f'LATE_{cid}_{fold}', 'kind': 'validation', 'priority': 91,
                    'depends_on_success': ['LATE_BREAK_CONTRACTS', f'SYN_ANALYZE_{cid}_full_{fold}'],
                    'argv': ['scripts/analyze_next5_late_break.py', '--candidate', cid, '--fold', str(fold)],
                    'description': 'Every predeclared genuine-change scenario at existing onset 512; fixed P-family artifact and original paired counts',
                    'retry_safe': False})
        progress('P-family late-break diagnostic scope frozen', 'Existing tau512 cases, all three P candidates and all five fitted folds; no new seeds or thresholds')
        return
    cid, fold = args.candidate, args.fold
    if cid is None or fold is None:
        raise ValueError('Existing P-family candidate and PRIMARY fitted fold required')
    policy = read_json(parent / 'POLICY_LOCK.json')
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(BASE / 'synthetic/MENU_LOCK.json') == policy['menu_sha256']
    assert digest(BASE / 'synthetic/SEED_COUNTS_LOCK.json') == policy['seed_counts_lock_sha256']
    set_num_threads(2)
    source = BASE / 'synthetic/candidates' / cid / 'full' / f'fold_{fold}'
    refroot = BASE / 'synthetic/references/full' / f'fold_{fold}'
    analysis = source / 'ANALYSIS'
    receipt = read_json(analysis / 'RESULTS.json')
    assert receipt['status'] == 'PASS'
    assert digest(analysis / 'CALIBRATION.csv') == receipt['files_sha256']['CALIBRATION.csv']
    with (analysis / 'CALIBRATION.csv').open(encoding='utf-8', newline='') as stream:
        thresholds = {(r['model'], r['background'], float(r['alpha'])): float(r['threshold']) for r in csv.DictReader(stream)}
    count_record = read_json(BASE / 'synthetic/SEED_COUNTS_LOCK.json')['sizes']['256']
    count_path = ROOT / count_record['path']
    assert digest(count_path) == count_record['sha256']
    with np.load(count_path, allow_pickle=False) as record:
        counts = record['counts']
        np.testing.assert_array_equal(record['synthetic_seed_index'], np.arange(256))
    menu = read_json(BASE / 'synthetic/MENU_LOCK.json')
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    backgrounds = background_mapping(lock)
    out = parent / cid / f'fold_{fold}'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'candidate': cid, 'fold': fold,
        'policy_sha256': digest(parent / 'POLICY_LOCK.json'), 'source_analysis_sha256': digest(analysis / 'RESULTS.json')}, immutable=True)
    score_rows, alarm_rows, paired_alarms, proofs = [], [], [], []
    for spec in cohort_menu(menu, lock):
        if spec['kind'] != 'counterexample' or spec['scenario'] in ('N0', 'N1', 'N2', 'N3', 'O1', 'OB16'):
            continue
        key = cohort_key(spec)
        candidate, cp = load_cohort(source, key)
        reference, rp = load_cohort(refroot, key)
        assert cp['input_sha256'] == rp['input_sha256']
        mask = onset_mask(candidate['synthetic_seed_index'], lock['tau_cycle'], 512)
        ids, tau = candidate['synthetic_seed_index'][mask], candidate['tau'][mask]
        assert len(np.unique(ids)) == 85 and np.sum(tau == 512) == 85 and np.sum(tau < 0) == 85
        for name in ('synthetic_seed_index', 'tau', 'control'):
            np.testing.assert_array_equal(candidate[name], reference[name])
        models = {'candidate': candidate['prediction'][mask, :1024]}
        models.update({name: reference[name][mask, :1024] for name in ('BL80', 'B2', 'D2')})
        base = {'candidate': cid, 'scope': 'full', 'fold': fold, 'scenario': spec['scenario'],
                'onset': 512, 'horizon': 1024, 'paired_generating_seeds': 85, 'supported_100_seed_threshold': False}
        new = score_draws(models['candidate'], tau, ids, 1024, counts)
        for role in ('BL80', 'B2', 'D2'):
            cache = parent / 'reference_draws' / f'fold_{fold}'
            cache.mkdir(parents=True, exist_ok=True)
            name = spec['scenario'] + '_' + role
            path, record_path = cache / (name + '.npy'), cache / (name + '.json')
            identity = {'prediction_file_sha256': rp['prediction_file_sha256'], 'counts_sha256': count_record['sha256'],
                        'policy_sha256': digest(parent / 'POLICY_LOCK.json'), 'onset': 512, 'horizon': 1024, 'role': role}
            if record_path.exists():
                saved = read_json(record_path)
                assert saved['identity'] == identity and digest(path) == saved['sha256']
                old = np.load(path, allow_pickle=False)
            else:
                assert not path.exists()
                old = score_draws(models[role], tau, ids, 1024, counts)
                np.save(path, old, allow_pickle=False)
                atomic_json(record_path, {'created_utc': utc_now(), 'identity': identity, 'sha256': digest(path)}, immutable=True)
            score_rows.append({**base, 'reference': role, 'candidate_TS_AUC': float(new[0]),
                               'reference_TS_AUC': float(old[0]), **paired_summary(old[:, None], new[:, None])})
        background = backgrounds[scenario_background(spec['scenario'], lock)]
        for alpha in menu['threshold_alphas']:
            for role, prediction in models.items():
                alarm_rows.append({**base, 'model': role, 'alpha': alpha,
                    **alarm_summary(prediction, tau, thresholds[(role, background, alpha)])})
            for event in ('prebreak', 'postbreak_detection'):
                new_event = stratum_event_draws(models['candidate'], tau, ids,
                    thresholds[('candidate', background, alpha)], event, counts)
                for role in ('BL80', 'B2', 'D2'):
                    old_event = stratum_event_draws(models[role], tau, ids, thresholds[(role, background, alpha)], event, counts)
                    paired_alarms.append({**base, 'reference': role, 'alpha': alpha, 'event': event,
                        'positive_delta_is_better': event == 'postbreak_detection',
                        **paired_summary(old_event[:, None], new_event[:, None])})
        proofs.append({'scenario': spec['scenario'], 'input_sha256': cp['input_sha256'],
            'candidate_prediction_sha256': cp['prediction_file_sha256'], 'reference_prediction_sha256': rp['prediction_file_sha256']})
        progress('Fixed P-family late-break stratum', cid + ' fold ' + str(fold) + ' ' + spec['scenario'] + '; same 85 existing late-onset seeds')
    assert len(proofs) == 15
    tables = {'LATE_BREAK_TS_AUC.csv': score_rows, 'LATE_BREAK_ALARMS.csv': alarm_rows, 'PAIRED_LATE_ALARMS.csv': paired_alarms}
    for name, values in tables.items():
        write_table(out / name, values)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS', 'candidate': cid, 'fold': fold,
        'scenarios': 15, 'existing_onset': 512, 'generating_seeds_per_scenario': 85,
        'new_fits': 0, 'new_predictions': 0, 'new_random_draws': 0, 'source_proofs': proofs,
        'sources_sha256': policy['sources_sha256'], 'policy_sha256': digest(parent / 'POLICY_LOCK.json'),
        'seed_counts_sha256': count_record['sha256'], 'files_sha256': {n: digest(out / n) for n in tables},
        'conditional_fixed_artifact_and_calibration': True, 'promotion_authority': False}, immutable=True)
    progress('P-family late-break artifact complete', cid + ' fold ' + str(fold) + '; all 15 original genuine-change cases')


if __name__ == '__main__':
    main()

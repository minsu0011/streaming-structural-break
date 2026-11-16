"""Read only metadata to connect registrations, actual fit starts and completions."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.chronology import exact_count, ordered_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='v1')
    args = parser.parse_args()
    if not args.label.isidentifier():
        raise ValueError('A unique simple output label is required')
    out = BASE/'maintenance/registration_chronology'/args.label
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}; chains = []; folds = []; missing = []

    def read(path):
        path = Path(path)
        evidence[path.relative_to(ROOT).as_posix()] = digest(path)
        return read_json(path)

    def chain(name, events):
        gaps = ordered_events(events)
        chains.append({'chain': name, 'events': dict(events), 'gaps': gaps})

    policy_path = ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'
    policy = read(policy_path)
    session = read(BASE/'SESSION.json')
    gate = read(BASE/'selection/NECESSARY_GATES.json')
    index = read(BASE/'ledger/failure_index_v2_policy_schema/RESULTS.json')
    assert index['new_scientific_recipes'] == 13
    jobs = []
    for path in sorted((BASE/'queue').glob('*.json')):
        job = read_json(path)
        state_path = BASE/'jobs'/job['id']/'STATE.json'
        if state_path.exists() and read_json(state_path)['status'] == 'COMPLETE':
            jobs.append((path, state_path, job))

    def execution(hyp_path, script, scope=None):
        matches = [(q, s) for q, s, j in jobs if j['argv'][0] == script
                   and j.get('hypothesis') == hyp_path.relative_to(ROOT).as_posix()
                   and (scope is None or ('--scope' in j['argv'] and
                        j['argv'][j['argv'].index('--scope') + 1] == scope))]
        assert len(matches) == 1, (hyp_path, script, scope, matches)
        job = read(matches[0][0]); state = read(matches[0][1])
        assert state['status'] == 'COMPLETE'
        exact_count(state['exit_code'], 0)
        assert job['hypothesis_sha256'] == state['hypothesis_sha256'] == digest(hyp_path)
        assert job['registered_utc'] == state['registered_utc']
        return job, state

    stages = 0; fit_count = 0; heuristic_folds = 0
    for recipe in index['recipes']:
        name = recipe['experiment_id']; hyp_path = BASE/'hypotheses'/f'{name}.json'
        hyp = read(hyp_path)
        assert hyp['policy_sha256'] == digest(policy_path)
        for scope in ('screen', 'full'):
            directory = BASE/'experiments'/name/scope
            if not (directory/'RESULTS.json').is_file():
                continue
            stages += 1
            local_policy = read(directory/'POLICY.json'); result = read(directory/'RESULTS.json')
            assert result['status'] == 'COMPLETE'
            assert local_policy['hypothesis_sha256'] == digest(hyp_path)
            if 'policy_sha256' in result:
                assert result['policy_sha256'] == digest(directory/'POLICY.json')
            else:
                assert name == 'O1_LINEAR_RESIDUAL'
            source_matches = [p for p in (ROOT/'scripts').glob('evaluate_next3_*.py')
                              if digest(p) == local_policy['script_sha256']]
            assert len(source_matches) == 1
            script = source_matches[0].relative_to(ROOT).as_posix(); read_source = source_matches[0]
            evidence[script] = digest(read_source)
            job, state = execution(hyp_path, script, scope)
            initial = [('session_start', session['started_utc']),
                       ('hypothesis', hyp['created_utc']), ('job_registered', job['registered_utc']),
                       ('job_started', state['started_utc']), ('stage_policy', local_policy['created_utc'])]
            tail = [('stage_complete', result['created_utc']), ('job_finished', state['finished_utc']),
                    ('necessary_gate_export', gate['created_utc'])]
            chain(f'{name}/{scope}', initial + tail)
            expected_folds = [0, 4] if scope == 'screen' else list(range(5))
            actual_folds = sorted(int(p.stem.split('_')[1]) for p in directory.glob('fold_[0-9].json'))
            assert actual_folds == expected_folds
            if scope == 'full':
                screen = read(directory.parent/'screen/RESULTS.json')
                chain(f'{name}/screen_before_full', [('screen_complete', screen['created_utc']),
                                                    ('full_job_registered', job['registered_utc'])])
            for fold in expected_folds:
                receipt = read(directory/f'fold_{fold}.json')
                assert receipt['status'] == 'PASS'
                start_path = directory/f'fold_{fold}_FIT_START.json'
                if name in ('R1_SCORE_GAP', 'R1_EVIDENCE_RATIO'):
                    exact_count(receipt['new_fits'], 0); assert not start_path.exists()
                    heuristic_folds += 1; count = 0; fit_start = ''
                else:
                    start = read(start_path)
                    keys = [key for key in ('intended_new_learner_fits', 'intended_gate_fits',
                                            'intended_residual_fits') if key in start]
                    assert len(keys) == 1
                    exact_count(start[keys[0]], 1)
                    completed_key = next(key for key in ('new_fits', 'new_gate_fits', 'new_residual_fits')
                                         if key in receipt)
                    exact_count(receipt[completed_key], 1)
                    fit_start = start['created_utc']; count = 1; fit_count += 1
                    chain(f'{name}/{scope}/fold_{fold}', initial + [('fit_start', fit_start)] + tail)
                assert 'created_utc' not in receipt
                missing.append({'path': (directory/f'fold_{fold}.json').relative_to(ROOT).as_posix(),
                                'absent': 'per_fold_completion_utc',
                                'available_upper_bound': result['created_utc']})
                folds.append({'candidate': name, 'scope': scope, 'fold': fold, 'new_fits': count,
                              'hypothesis_utc': hyp['created_utc'], 'fit_start_utc': fit_start,
                              'completion_upper_bound_utc': result['created_utc'],
                              'completion_bound_scope': 'stage completion, not individual fit end'})
            exact_count(result['new_fits'], 0 if name.startswith('R1_') else len(expected_folds))
    exact_count(stages, 25); exact_count(len(folds), 86)
    exact_count(fit_count, 72); exact_count(heuristic_folds, 14)

    # Shared inner models have their own registration. Later downstream hypotheses
    # are required before downstream fits, not before already registered base fits.
    nested_hyp_path = BASE/'hypotheses/N001_NESTED_BASE_OOF.json'; nested_hyp = read(nested_hyp_path)
    assert nested_hyp['policy_sha256'] == digest(policy_path)
    nested_fits = 0
    for scope, expected in [('screen', 16), ('full', 40)]:
        directory = BASE/'nested'/scope; local_policy = read(directory/'POLICY.json')
        result = read(directory/'RESULTS.json')
        assert local_policy['hypothesis_sha256'] == digest(nested_hyp_path)
        assert 'created_utc' not in local_policy
        missing.append({'path': (directory/'POLICY.json').relative_to(ROOT).as_posix(),
                        'absent': 'policy_created_utc', 'available_upper_bound': '',
                        'note': 'Policy hash binding checked; no separate policy timestamp inferred.'})
        job, state = execution(nested_hyp_path, 'scripts/build_next3_nested_oof.py', scope)
        starts = sorted(directory.glob('outer_*/inner_*/attempt_1/START.json'))
        exact_count(len(starts), expected)
        for start_path in starts:
            start = read(start_path); complete = read(start_path.with_name('COMPLETE.json'))
            published = read(start_path.parent.parent/'RESULTS.json')
            outer = read(start_path.parents[2]/'RESULTS.json')
            assert complete == published and complete['status'] == 'PASS'
            assert start['binding'] == complete['binding']
            assert outer['scope_policy_sha256'] == digest(directory/'POLICY.json')
            exact_count(start['intended_new_learner_fits'], 1)
            exact_count(complete['new_learner_fits'], 1)
            chain(start_path.relative_to(BASE).as_posix(), [
                ('shared_hypothesis', nested_hyp['created_utc']), ('job_registered', job['registered_utc']),
                ('job_started', state['started_utc']), ('fit_start', start['created_utc']),
                ('fit_complete', complete['created_utc']), ('outer_complete', outer['created_utc']),
                ('scope_complete', result['created_utc']), ('job_finished', state['finished_utc'])])
            nested_fits += 1
        exact_count(result['new_learner_fits_this_invocation'], expected)
        for name in ('R2_LOGISTIC_GATE', 'R3_DEPTH2_GATE', 'O1_LINEAR_RESIDUAL'):
            downstream = read(BASE/'experiments'/name/scope/'POLICY.json')
            key = 'nested_sha256' if name == 'O1_LINEAR_RESIDUAL' else 'nested_scope_sha256'
            assert downstream[key] == digest(directory/'RESULTS.json')
            chain(f'{name}/{scope}/base_reuse', [('shared_scope_complete', result['created_utc']),
                                               ('downstream_policy', downstream['created_utc'])])
    exact_count(nested_fits, 56)

    refit = BASE/'refits/R3_DEPTH2_GATE/research_v1'
    refit_hyp_path = BASE/'hypotheses/R3_DEPTH2_GATE_SYNTHETIC.json'; hyp = read(refit_hyp_path)
    job, state = execution(refit_hyp_path, 'scripts/refit_next3_depth2_gate.py')
    start = read(refit/'FIT_START.json'); result = read(refit/'RESULTS.json')
    assert start['hypothesis_sha256'] == result['hypothesis_sha256'] == digest(refit_hyp_path)
    exact_count(start['intended_gate_fits'], 1); exact_count(result['new_gate_fits'], 1)
    chain('R3/research_all_DEV_refit', [('hypothesis', hyp['created_utc']),
        ('job_registered', job['registered_utc']), ('job_started', state['started_utc']),
        ('fit_start', start['created_utc']), ('fit_complete', result['created_utc']),
        ('job_finished', state['finished_utc'])])
    with (out/'CANDIDATE_FOLD_CHRONOLOGY.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(folds[0])); writer.writeheader(); writer.writerows(folds)
    atomic_json(out/'EVENT_CHAINS.json', chains, immutable=True)
    atomic_json(out/'RECORDED_TIMESTAMP_LIMITATIONS.json', missing, immutable=True)
    # SESSION is live. Retain its actual audited bytes rather than treating the
    # future status change as modification of an immutable historical source.
    session_key = (BASE/'SESSION.json').relative_to(ROOT).as_posix()
    (out/'SESSION_INPUT.json').write_bytes((BASE/'SESSION.json').read_bytes())
    assert digest(out/'SESSION_INPUT.json') == evidence.pop(session_key)
    report = {'status': 'PASS_RECORDED_REGISTRATION_CHRONOLOGY', 'created_utc': utc_now(),
        'new_scientific_recipes': 13, 'candidate_stages': stages, 'candidate_fold_receipts': len(folds),
        'candidate_fits': fit_count, 'heuristic_zero_fit_fold_outputs': heuristic_folds,
        'shared_nested_fits': nested_fits, 'research_all_DEV_gate_refits': 1,
        'science_fits_accounted': fit_count + nested_fits + 1, 'new_fits_in_this_audit': 0,
        'event_chains_checked': len(chains), 'chronology_reversals': 0,
        'missing_per_fold_completion_times': 86, 'missing_nested_policy_times': 2,
        'observations_labels_tau_features_predictions_read': False,
        'scope': 'Existing application UTC receipts and hash bindings only. No external timestamp notarization, OS clock attestation, or proof of unrecorded activity. Individual candidate fit end times are not present; stage completion is an upper bound. Shared nested fits use their own preregistration.',
        'evidence_sha256': evidence,
        'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in
            [Path(__file__), ROOT/'src/next3/chronology.py', ROOT/'tests/test_next3_chronology.py']},
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    atomic_json(out/'RESULTS.json', report, immutable=True)
    progress('Recorded experiment chronology audited',
             f'{len(chains)} UTC chains; 25 stages, 86 fold outputs and all 129 scientific fits; missing timestamps explicitly retained')
    print('NEXT3 REGISTRATION CHRONOLOGY PASS', len(chains), flush=True)


if __name__ == '__main__':
    main()

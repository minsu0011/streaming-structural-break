"""Real frozen model paths on synthetic inputs, with no RP/cache reader."""
import copy
import hashlib
import importlib
import pickle
from pathlib import Path
import numpy as np
from src.next8.common import ROOT, BASE, read_json, digest
from src.next8.streaming_runtime import load_prepared, IntegratedState
from src.next8.synthetic_controls import matched_path, interventions
from src.next5.model import FeatureModel
from src.next5.runtime import SharedResearchBaseline, ScalarDetector

REPLAY_CASES = (('Gaussian', 0, 512, 0), ('Gaussian', 0, 2048, 1),
                ('AR08', 1, 512, 2), ('AR08', 1, 2048, 0),
                ('ARCH1', 2, 512, 3), ('ARCH1', 2, 2048, 4),
                ('Student_t5', 3, 512, 0), ('Student_t5', 3, 2048, 1))
HISTORICAL_CASES = (('N0', 11), ('V4', 1), ('RETURN_NORMAL', 0))


def bits(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def clone_and_pickle(state):
    if isinstance(state, ScalarDetector):
        args, prepared = pickle.loads(pickle.dumps((copy.deepcopy(state.args), state.prepared), protocol=5))
        return ScalarDetector(state.kernel, args, prepared)
    return pickle.loads(pickle.dumps(state.snapshot(), protocol=5))


def run(factory, historical, online):
    state = factory(historical)
    prediction = np.asarray([state.predict_one(x) for x in online], dtype=np.float64)
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    resumed = factory(historical)
    left = np.asarray([resumed.predict_one(x) for x in online[:73]], dtype=np.float64)
    restored = clone_and_pickle(resumed)
    right = np.asarray([restored.predict_one(x) for x in online[73:]], dtype=np.float64)
    np.testing.assert_array_equal(prediction.view(np.uint64), np.r_[left, right].view(np.uint64))
    changed = online.copy()
    changed[73:] = np.float32(-7.) * changed[73:] + np.float32(.375)
    alternative = factory(historical)
    altered = np.asarray([alternative.predict_one(x) for x in changed], dtype=np.float64)
    np.testing.assert_array_equal(prediction[:73].view(np.uint64), altered[:73].view(np.uint64))
    chunk = factory(historical)
    pieces = []
    for start, stop in ((0, 31), (31, 73), (73, 257), (257, len(online))):
        values = online[start:stop]
        pieces.append(np.asarray(chunk.replay(values), dtype=np.float64) if hasattr(chunk, 'replay')
                      else np.asarray([chunk.predict_one(x) for x in values], dtype=np.float64))
    np.testing.assert_array_equal(prediction.view(np.uint64), np.concatenate(pieces).view(np.uint64))
    return prediction, {'points': len(online), 'fresh_scalar_chunk_bitwise': True,
                        'pickle_resume_at73_bitwise': True, 'future_suffix_prefix73_bitwise': True,
                        'prediction_sha256': bits(prediction), 'only_H_and_current_scalar_given_to_state': True}


def required_runtime_paths():
    """Minimal fitted artifacts; full real-point OOF/cache is deliberately absent."""
    paths = {ROOT / 'artifacts/next8/CURRENT_RUN.json', BASE / 'SESSION_NEXT8.json'}
    scopes = []
    for early in (False, True):
        for fold in range(5):
            scope = f'outer_{fold}' + ('_EARLY_T199' if early else '')
            directory = BASE / 'models' / scope
            done = read_json(directory / 'DONE.json')
            paths.add(directory / 'DONE.json')
            for method in sorted(set(done['model_aliases'].values())):
                path = directory / (method + '.joblib')
                assert digest(path) == done['files_sha256'][path.name]
                paths.add(path)
            selection = BASE / 'h_context_selection' / scope
            paths.update((selection / 'SELECTION.json', selection / 'DONE.json'))
            assert digest(selection / 'SELECTION.json') == read_json(selection / 'DONE.json')['selection_sha256']
            graph = BASE / 'graph_states' / scope
            gd = read_json(graph / 'DONE.json')
            paths.add(graph / 'DONE.json')
            for model in gd['models']:
                if model['status'] == 'COMPLETE':
                    path = graph / f"F{model['feature']:03d}_T3.joblib"
                    assert digest(path) == model['model_sha256']
                    paths.add(path)
            integration = BASE / ('early_integration' if early else 'integration') / scope / 'DONE.json'
            paths.add(integration)
            references = []
            for kind in ('J102', 'C205'):
                path = (BASE / 'early_baseline' / scope / kind / 'model.joblib' if early else
                        ROOT / f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}/model.joblib')
                paths.add(path)
                references.append(path.relative_to(ROOT).as_posix())
            scopes.append({'scope': scope, 'early_only': early, 'fold': fold, 'reference_models': references,
                           'M6_eligible': read_json(integration)['eligible']})
    historical = ROOT / 'artifacts/next7/FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json'
    old = read_json(historical)
    paths.add(historical)
    for name, sha in old['files_sha256'].items():
        assert digest(ROOT / name) == sha
        paths.add(ROOT / name)
    paths.update((ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json',
                  ROOT / 'artifacts/next7/frozen_reference_fixture/FIXTURE.json',
                  ROOT / 'artifacts/next7/frozen_reference_fixture/PREDICTIONS.npz'))
    return sorted(paths), scopes


def calculate_research(manifest):
    predictions, checks, inputs = {}, [], {}
    for scope_record in manifest['scopes']:
        scope = scope_record['scope']
        heads = {method: load_prepared(BASE / 'models' / scope,
                 BASE / 'h_context_selection' / scope / 'SELECTION.json', BASE / 'graph_states' / scope, method)
                 for method in ('M1', 'M2', 'M3', 'M4', 'M5')}
        baseline = SharedResearchBaseline(*(FeatureModel.load(ROOT / p) for p in scope_record['reference_models']))
        factories = {'M0': baseline.new_state, **{m: p.new_state for m, p in heads.items()},
                     'M6': lambda h: IntegratedState(baseline, heads['M5'], h, scope_record['M6_eligible'])}
        for background, index, length, intervention in REPLAY_CASES:
            h, o = matched_path(background, index, length)
            o = interventions(o, index)[intervention]
            case = f'{background}_{index}_H{length}_I{intervention}'
            inputs[case] = {'H_sha256': bits(h), 'Online_sha256': bits(o), 'H_length': len(h), 'Online_length': len(o)}
            for method, factory in factories.items():
                key = f'{scope}_{case}_{method}'
                values, check = run(factory, h, o)
                predictions[key] = values
                checks.append({'scope': scope, 'method': method, 'case': case, **check,
                               'actual_alias_method': heads[method].method if method in heads else method,
                               'M6_eligible': scope_record['M6_eligible'] if method == 'M6' else None})
        h, o = matched_path('Gaussian', 0, 512)
        for length in (0, 159):
            for method, prepared in heads.items():
                values, check = run(prepared.new_state, h[:length], o)
                state = prepared.new_state(h[:length])
                assert state.short_H and state.zero_features
                key = f'{scope}_SHORT_H{length}_{method}'
                predictions[key] = values
                checks.append({'scope': scope, 'method': method, 'case': f'SHORT_H{length}', **check,
                               'short_H_zero_feature_UNKNOWN_fallback': True,
                               'canonical_BL80_short_H_requirement_not_changed': True})
        print('Raw-free actual research paths', scope, flush=True)
    return predictions, checks, inputs


def calculate_canonical():
    """Use public generators and compare against the existing NEXT7 fixture."""
    from src.next6_extension.synthetic import generated
    lock = read_json(ROOT / 'artifacts/next7/FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json')
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    directory = ROOT / 'artifacts/next7/frozen_reference_fixture'
    fixture = read_json(directory / 'FIXTURE.json')
    assert digest(directory / 'PREDICTIONS.npz') == fixture['predictions_sha256']
    assert digest(ROOT / 'artifacts/next7/FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json') == fixture['reference_lock_sha256']
    result, checks = {}, []
    with np.load(directory / 'PREDICTIONS.npz', allow_pickle=False) as expected:
        for role in lock['references']:
            entry = importlib.import_module(Path(role['entry']).stem)
            for case, seed in HISTORICAL_CASES:
                h, o, _, _ = generated(case, seed, partition=31, horizon=1024, control=False, lock=suite)
                name = role['id'] + '_' + case + '_seed' + str(seed)
                assert {'H': bits(h), 'O': bits(o)} == fixture['synthetic_input_sha256'][case + '_seed' + str(seed)]
                stream = entry.infer(iter([(h, iter(o))]), str(ROOT / role['model_directory']))
                assert next(stream) is None
                values = np.fromiter(stream, dtype=np.float64)
                np.testing.assert_array_equal(values.view(np.uint64), expected[name].view(np.uint64))
                result['CANONICAL_' + name] = values
                checks.append({'role': role['id'], 'case': case, 'points': len(values),
                               'historical_NEXT7_fixture_bitwise_equal': True, 'public_infer_generator': True})
    assert len(checks) == 9
    return result, checks


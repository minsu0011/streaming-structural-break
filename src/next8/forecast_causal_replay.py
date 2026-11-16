"""Inference-only D/T replay on fixed RP cases; labels are never inputs."""
import hashlib
import numpy as np
import joblib
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity
from src.next8.features import DETAILED, HORIZONS, UNKNOWN, movement_direction
from src.next8.protocol import H_BUCKETS
from src.next8.cache import FeatureCache
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.runtime_audit import observations
from src.next8.full_replay_audit_v2 import FreshSeriesCache
from src.next8.forecasting import forecast_designs, predict_bundle, _persistence_distribution
from src.next8.transitions import transition_inputs, transition_tabular_design


def bits_equal(a, b):
    a = np.asarray(a); b = np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), 'Future suffix changed inference bits'


def saved_parity(actual, expected):
    a = np.asarray(actual); b = np.asarray(expected)
    assert a.shape == b.shape and a.size > 0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error = float(np.max(np.abs(a.astype(float) - b.astype(float))))
    assert error <= np.finfo(np.float32).eps, 'Saved forecast prediction mismatch'
    return {'maximum_absolute_error': error, 'saved_bitwise_equal': a.dtype == b.dtype and a.tobytes() == b.tobytes(),
            'fresh_prediction_sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def fixed_cases(cache, fold):
    scope = outer_scope(fold, cache.data)
    h = np.diff(cache.data.series['historical_offsets'])
    o = np.diff(cache.data.series['online_offsets'])
    records = []
    for k, (lo, hi) in enumerate(H_BUCKETS):
        eligible = sorted(int(sid) for sid, nh, no in zip(cache.data.ids, h, o)
                          if int(sid) in scope.validation_ids and nh >= lo and
                          (hi is None or nh <= hi) and no >= 465)
        records.append({'H_bin': k, 'eligible_series': len(eligible), 'selected_ids': eligible[:1]})
    return records


def fresh_variants(sid, historical, online, prefix=449):
    assert len(online) > prefix
    changed = np.asarray(online, dtype=np.float32).copy()
    changed[prefix:] = -7 * changed[prefix:] + .375
    variants = [FreshSeriesCache(sid, historical, v) for v in (online, changed, online[:prefix])]
    for fresh in variants[1:]:
        bits_equal(variants[0].values[:prefix], fresh.values[:prefix])
        # Whole-Online diagnostics (online_bound_count) intentionally differ.
        for name in ('seed_values', 'low', 'high', 'valid', 'epsilon', 'context', 'center', 'spread'):
            bits_equal(variants[0].series[name], fresh.series[name])
    return variants


def designs_and_predictions(fresh, rows, j, horizon, selection, done, models, transition_done, t3, t4):
    parents = done['parents']
    designs = forecast_designs(fresh, rows, j, horizon, selection['stable_features'], parents, selection['variable_features'])
    current = fresh.current(rows, [j])[:, 0]
    valid = fresh.series['valid'][fresh.series_index[rows], j]
    output = {}
    for method, donor in done['method_aliases'].items():
        # Re-run aliases with their own requested design to check schema parity.
        for target, prediction in predict_bundle(models[donor], designs[method], current, valid).items():
            output[method + '_' + target] = prediction
    td = transition_tabular_design(fresh, rows, j, parents, designs['D5_CROSS'])
    designs['T2'] = td
    tree = models.get('T2_state')
    state = np.zeros((len(rows), 4), dtype=np.float32)
    if tree is None:
        state[:, UNKNOWN] = 1
    else:
        p = tree.predict_proba(td)
        state[valid, :p.shape[1]] = p[valid]
        state[~valid, UNKNOWN] = 1
    output['T2_state'] = state
    own, other = transition_inputs(fresh, rows, j, parents)
    output['T1_state'] = t3.predict(own, other, use_parents=False)[0]
    output['T3_state'], details = t3.predict(own, other)
    output['T3_active_factors'] = details['active_parent_factors']
    output['own_supported'] = details['own_context_supported']
    if transition_done['T4_alias_T3_identical_parents']:
        output['T4_state'] = output['T3_state']
        output['T4_active_factors'] = output['T3_active_factors']
    else:
        ro, rp = transition_inputs(fresh, rows, j, transition_done['random_parents'])
        output['T4_state'], rd = t4.predict(ro, rp)
        output['T4_active_factors'] = rd['active_parent_factors']
    policy = done['D0_policy']
    direction = movement_direction(current - fresh.lag(rows, [j], 1)[:, 0], fresh.epsilon(rows, [j], 1)[:, 0])
    output['D0_delta'] = np.zeros(len(rows), dtype=np.float32)
    output['D0_movement_persistence'] = _persistence_distribution(direction, policy['movement_persistence_train_probability'], 3)
    state = np.zeros((len(rows), 4), dtype=np.float32)
    known = own < UNKNOWN
    state[known, :3] = _persistence_distribution(own[known], policy['state_persistence_train_probability'], 3)
    state[~known, UNKNOWN] = 1
    output['D0_state_persistence'] = state
    return designs, output


def saved_rows(path, global_rows):
    with np.load(path, allow_pickle=False) as z:
        rows = z['global_row']
        assert np.all(np.diff(rows) > 0)
        at = np.searchsorted(rows, global_rows)
        assert np.all(at < len(rows))
        np.testing.assert_array_equal(rows[at], global_rows)
        return {name: z[name][at] for name in z.files if name not in ('global_row', 'dataset_id')}


def audit_fold(fold):
    release = read_json(BASE / 'PREDICTION_FREEZE_NEXT8.json')
    assert release['status'] == 'FROZEN'
    checked = {}
    def verify(path):
        key = path.relative_to(BASE).as_posix()
        sha = checked.get(key) or digest(path)
        assert release['files_sha256'][key] == sha, key
        checked[key] = sha
        return path
    out = BASE / 'forecast_causal_replay' / f'fold_{fold}'
    out.mkdir(parents=True, exist_ok=False)
    cache = FeatureCache(); cases = fixed_cases(cache, fold); prepared = []
    with activity(f'D/T fresh causal feature replay fold{fold}', 'FORECAST_CAUSAL_REPLAY', 'B7_NORMALIZATION_AND_LINEAGE'):
        for case in cases:
            for sid in case['selected_ids']:
                h, o, global_rows = observations(cache, sid)
                variants = fresh_variants(sid, h, o)
                bits_equal(variants[0].values[:449], np.asarray(cache.values[global_rows[:449]], dtype=np.float32))
                prepared.append({'sid': sid, 'H_bin': case['H_bin'], 'H_length': len(h),
                                 'online_length_audit_only': len(o), 'variants': variants, 'global_rows': global_rows})
    records = []
    for early in (False, True):
        scope = early_scope(outer_scope(fold, cache.data), early)
        sp = BASE / 'h_context_selection' / scope.name
        selection = read_json(verify(sp / 'SELECTION.json'))
        assert selection['scope'] == scope.provenance()
        for j in DETAILED:
            for horizon in HORIZONS:
                with activity(f'D/T fitted prediction prefix replay {scope.name} F{j:03d} h{horizon}', 'FORECAST_CAUSAL_REPLAY', 'B7_NORMALIZATION_AND_LINEAGE'):
                    fp = BASE / 'forecasts' / scope.name / f'F{j:03d}_h{horizon}'
                    tp = BASE / 'transitions' / scope.name / f'F{j:03d}_h{horizon}'
                    done = read_json(verify(fp / 'DONE.json')); trans = read_json(verify(tp / 'DONE.json'))
                    assert done['scope'] == trans['scope'] == scope.provenance()
                    assert done['parents'] == trans['parents']
                    for directory, receipt in ((fp, done), (tp, trans)):
                        for name, sha in receipt['files_sha256'].items():
                            verify(directory / name)
                            assert checked[(directory / name).relative_to(BASE).as_posix()] == sha
                    models = {donor: joblib.load(fp / (donor + '.joblib')) for donor in set(done['method_aliases'].values())}
                    if 'T2_state.joblib' in done['files_sha256']:
                        models['T2_state'] = joblib.load(fp / 'T2_state.joblib')
                    t3 = joblib.load(tp / 'T3.joblib')
                    t4 = None if trans['T4_alias_T3_identical_parents'] else joblib.load(tp / 'T4.joblib')
                    for case in prepared:
                        rows = np.arange(400 if early else 0, 449, 8, dtype=np.int64)
                        va = case['variants']; outputs = []
                        reference_designs = None
                        for fresh in va:
                            designs, prediction = designs_and_predictions(fresh, rows, j, horizon, selection, done, models, trans, t3, t4)
                            if reference_designs is None:
                                reference_designs = designs
                            else:
                                assert designs.keys() == reference_designs.keys()
                                for method, design in designs.items():
                                    assert design.names == reference_designs[method].names
                                    bits_equal(design.values, reference_designs[method].values)
                            outputs.append(prediction)
                        for output in outputs[1:]:
                            assert output.keys() == outputs[0].keys()
                            for name, prediction in output.items():
                                bits_equal(prediction, outputs[0][name])
                        gr = case['global_rows'][rows]
                        expected = {**saved_rows(fp / 'PREDICTIONS.npz', gr), **saved_rows(tp / 'PREDICTIONS.npz', gr)}
                        parity = {}
                        for name, actual in outputs[0].items():
                            donor_name = name
                            for method, donor in done['method_aliases'].items():
                                if name.startswith(method + '_') and name[len(method) + 1:] in ('delta', 'movement', 'reversal', 'state'):
                                    donor_name = donor + name[len(method):]
                                    break
                            parity[name] = saved_parity(actual, expected[donor_name])
                        records.append({'scope': scope.name, 'feature': j, 'horizon': horizon,
                            **{k: v for k, v in case.items() if k not in ('variants', 'global_rows')},
                            'origins': rows.tolist(), 'origins_with_future_absent_after_truncation': int(np.sum(rows + horizon >= 449)),
                            'all_prefix_feature_design_prediction_bits_equal': True, 'saved_predictions': parity})
    result = {'status': 'PASS', 'timestamp': utc_now(), 'fold': fold, 'case_policy': cases,
        'series': len(prepared), 'feature_horizon_scope_case_records': len(records), 'records': records,
        'frozen_source_artifact_sha256': checked, 'release_sha256': digest(BASE / 'PREDICTION_FREEZE_NEXT8.json'),
        'policy_sha256': digest(BASE / 'FORECAST_CAUSAL_REPLAY_POLICY_NEXT8.json'),
        'new_model_fits': 0, 'new_quality_scores': 0, 'future_targets_and_break_labels_consumed': False,
        'scope_limit': 'Fixed RP H-bin audit sample; whole RP D/T output parity is not claimed. Majority D0 priors are not refitted or replayed.'}
    assert len(records) == len(prepared) * 2 * len(DETAILED) * len(HORIZONS)
    atomic(out / 'DONE.json', result)
    return result


def summarize():
    records = []; hashes = {}
    for fold in range(5):
        path = BASE / 'forecast_causal_replay' / f'fold_{fold}' / 'DONE.json'
        r = read_json(path)
        assert r['status'] == 'PASS' and r['fold'] == fold
        assert r['policy_sha256'] == digest(BASE / 'FORECAST_CAUSAL_REPLAY_POLICY_NEXT8.json')
        assert r['release_sha256'] == digest(BASE / 'PREDICTION_FREEZE_NEXT8.json')
        hashes[path.relative_to(BASE).as_posix()] = digest(path)
        records.append(r)
    atomic(BASE / 'FORECAST_CAUSAL_AUDIT_NEXT8.json', {'status': 'PASS', 'timestamp': utc_now(),
        'folds': records, 'input_receipts_sha256': hashes, 'new_model_fits': 0, 'new_quality_scores': 0,
        'fixed_cases_not_representative_quality_evaluation': True})

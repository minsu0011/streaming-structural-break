"""Independent envelope checks from original membership locks, without outcomes."""
from types import SimpleNamespace
import numpy as np
import joblib
from src.next8.common import ROOT, BASE, read_json, digest, ids_hash
from src.next8.data import FitScope
from src.next8.screening import procedure_scope, selection_splits
from src.next8.nested_auxiliary import early_scope, subdivide_scope, selected_cross_branch_needed
from src.next5.guard import ResearchGuard

SCOPE_FIELDS = ('scope', 'training_ids_sha256', 'validation_ids_sha256', 'outer_excluded_ids_sha256',
                'training_series', 'validation_series', 'selection_training_only')


def expected_scopes():
    guard = ResearchGuard(ROOT)
    # Original IDs and group assignments only. No ResearchArrays/FeatureCache,
    # raw Parquet, historical/Online arrays or target arrays are loaded here.
    cache = SimpleNamespace(data=SimpleNamespace(guard=guard, ids=np.asarray(sorted(guard.research_ids), dtype=np.int32)))
    scopes = {}

    def add(scope):
        if scope.name in scopes:
            assert scopes[scope.name] == scope
        scopes[scope.name] = scope

    for fold in range(5):
        for inner in (None, 0, 1, 2):
            parent = procedure_scope(cache, fold, inner)
            splits = selection_splits(cache, fold, inner)
            for early in (False, True):
                add(early_scope(parent, early))
                for split in splits:
                    scope = early_scope(split, early)
                    add(scope)
                    for leaf in subdivide_scope(scope, guard.split.groups):
                        add(leaf)
    for scope in scopes.values():
        guard.partition(scope.training_ids, scope.validation_ids | scope.outer_excluded_ids)
    return scopes


def validate_training_record(training, scopes):
    name = training.get('scope')
    if name not in scopes:
        raise ValueError('Unregistered fitting scope: ' + str(name))
    expected = scopes[name].provenance()
    for key in SCOPE_FIELDS:
        if training.get(key) != expected[key]:
            raise ValueError('Training envelope differs at ' + name + ':' + key)
    if training.get('validation_labels_used') or training.get('validation_or_late_training_labels_used'):
        raise ValueError('Validation/late fitting label use claimed')
    if 'EARLY_T199' in name:
        if 'training_max_online_time' in training and training['training_max_online_time'] > 199:
            raise ValueError('Early-only reference trained beyond199')
        if training.get('early_training_cutoff', 199) != 199:
            raise ValueError('Early-only head cutoff mismatch')
    return expected


def audit_lineage():
    scopes = expected_scopes()
    release = read_json(BASE / 'PREDICTION_FREEZE_NEXT8.json')
    assert release['status'] == 'FROZEN'
    models, selections, splits, state_inputs, maps = [], [], [], [], []
    for name, sha in release['files_sha256'].items():
        path = BASE / name
        if path.suffix == '.joblib':
            assert digest(path) == sha
            obj = joblib.load(path)
            objects = obj.items() if isinstance(obj, dict) else [('model', obj)]
            for component, model in objects:
                if model is None:
                    continue
                training = model.training
                expected = validate_training_record(training, scopes)
                models.append({'artifact': name, 'component': component, 'model_sha256': sha,
                               'scope': expected['scope'], 'training_ids_sha256': expected['training_ids_sha256'],
                               'training_series_envelope': expected['training_series'],
                               'early_only': 'EARLY_T199' in expected['scope'],
                               'recorded_actual_training_rows': training.get('actual_training_rows', training.get('training_rows')),
                               'model_class': type(model).__name__, 'scope_matches_original_group_locks': True})
        elif path.name == 'SELECTION.json' and path.parent.parent.name in ('selection', 'h_context_selection'):
            assert digest(path) == sha
            record = read_json(path)
            expected = validate_training_record(record['scope'], scopes)
            assert set(record['selection_ids']) == scopes[expected['scope']].training_ids
            assert ids_hash(record['selection_ids']) == expected['training_ids_sha256']
            assert len(record['stable_features']) <= 6 and len(record['variable_features']) <= 6
            assert not record.get('outer_or_meta_validation_labels_used', False)
            selections.append({'artifact': name, 'sha256': sha, 'scope': expected['scope'],
                               'stable_count': len(record['stable_features']), 'variable_count': len(record['variable_features']),
                               'exact_original_TRAIN_membership_verified': True})
        elif path.name == 'DONE.json' and path.parent.name.startswith('split_') and 'scope' in read_json(path):
            assert digest(path) == sha
            record = read_json(path)
            expected = validate_training_record(record['scope'], scopes)
            splits.append({'artifact': name, 'sha256': sha, 'scope': expected['scope'], 'original_group_exclusions_verified': True})
        elif path.name == 'STATE_INPUT_PROVENANCE.json':
            assert digest(path) == sha
            record = read_json(path)
            expected = validate_training_record(record['scope'], scopes)
            scope_name = expected['scope']
            selection = read_json(BASE / 'h_context_selection' / scope_name / 'SELECTION.json')
            graph = read_json(BASE / 'parent_selection' / scope_name / 'GRAPH.json')
            required = selected_cross_branch_needed(selection, graph)
            assert record['cross_branch_needed'] == required
            if required:
                assert record['all_state_fit_and_graph_selection_exclude_prediction_group']
                for part in record['training_state_parts']:
                    child = validate_training_record(part['scope'], scopes)
                    assert scopes[child['scope']].training_ids <= scopes[scope_name].training_ids
                    assert scopes[child['scope']].validation_ids <= scopes[scope_name].training_ids
                    assert part['graph_selection_ids_sha256'] == child['training_ids_sha256']
            else:
                assert record['no_in_sample_state_inputs'] and record['no_unused_leaf_fits_required']
            state_inputs.append({'artifact': name, 'sha256': sha, 'scope': scope_name,
                                 'complete_crossfit_branch_needed': required, 'selection_aware_skip_verified': not required})
        elif path.name == 'H_STATE_ESTIMATORS.json':
            assert digest(path) == sha
            record = read_json(path)
            for specification in record.values():
                training = specification['training']
                validate_training_record(training, scopes)
                assert training['H_series_count'] == training['training_series']
                assert training['H_quantiles_training_only'] and training['equal_weight_per_H_series']
                assert not training['Online_values_or_labels_used']
            maps.append({'artifact': name, 'sha256': sha, 'descriptor': record})
    assert len(selections) >= 50 and len(state_inputs) >= 25
    assert models and splits
    return {'status': 'PASS', 'models': models, 'feature_selections': selections, 'inner_splits': splits,
            'crossfit_state_inputs': state_inputs, 'H_population_maps': maps,
            'counts': {'serialized_estimator_components_checked': len(models), 'selections_checked': len(selections),
                       'split_receipts_checked': len(splits), 'crossfit_state_input_procedures': len(state_inputs),
                       'H_map_descriptors': len(maps)},
            'expected_scopes_reconstructed_from_original_RP_and_NEXT7_inner_group_locks': True,
            'no_ResearchArrays_FeatureCache_Parquet_or_target_arrays_loaded': True,
            'model_byte_hashes_checked_before_deserialization': True,
            'source_prediction_freeze_sha256': digest(BASE / 'PREDICTION_FREEZE_NEXT8.json'),
            'normal_full_DEV_deployment_models_are_not_among_new_research_fit_paths': True,
            'validation_truth_used_for_any_new_selection': False,
            'scope_limit': 'Independent recorded envelope/exact selection membership check plus frozen producer contracts and tests; metadata is not an OS information-flow proof',
            'conditional_or_known_state_fits_may_use_admitted_TRAIN_subsets': True,
            'H_normalization_is_per_series_and_excluded_from_population_fit_counts': True}

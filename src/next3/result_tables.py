"""Explicit record types for closed science, inherited validation and zero change."""
import csv
import io
import math
import statistics


def csv_bytes(rows):
    if not rows:
        raise ValueError('A scientific table cannot silently lose all rows')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def close_recipe(row, recipe):
    if row['experiment_id'] != recipe['experiment_id']:
        raise ValueError('Recipe/result identity mismatch')
    if recipe['scientific_role_selected'] is not False:
        raise ValueError('Closed NEXT3 science contains an unexpected promotion')
    disposition = recipe['disposition']
    if disposition not in ('FULL_CV_NO_PROMOTION', 'SCREEN_FAIL_STOP'):
        raise ValueError('Unknown closed recipe disposition')
    full = recipe['stages'].get('full')
    if disposition == 'SCREEN_FAIL_STOP':
        if full is not None or row.get('full_mean') not in (None, ''):
            raise ValueError('A failed screen cannot carry a full-CV result')
    else:
        if not full or len(full['fold_scores']) != 5:
            raise ValueError('A completed full recipe requires five original folds')
        if not math.isclose(float(row['full_mean']), full['mean'], rel_tol=0, abs_tol=1e-14):
            raise ValueError('Full mean differs from frozen receipt')
    return {**row, 'original_stage_status': row['status'], 'status': disposition,
        'final_disposition': disposition, 'record_type': 'NEXT3_NEW_SCIENTIFIC_RECIPE',
        'final_failure_reason': recipe['reason'], 'scientific_role_selected': False,
        'promotion': False, 'selection_adjusted': False,
        'recipe_evidence_binding_sha256': recipe['recipe_evidence_binding_sha256'],
        'not_run_after_rejection': str(recipe['not_evaluated_after_rejection']),
        'new_scientific_hypothesis_count': 1, 'reproduction_fits_included': False}


def inherited_validation(rows, *, role_names, expected_kind, source, sha):
    selected = []
    for row in rows:
        if row.get('candidate') not in role_names or row.get('record_type') != expected_kind:
            continue
        selected.append({**row, 'record_type': 'INHERITED_'+expected_kind,
            'validation_kind': expected_kind, 'evidence_origin': 'NEXT2_FROZEN_NO_NEXT3_REFIT',
            'source_table': source, 'source_table_sha256': sha,
            'new_fits_in_NEXT3': 0, 'new_independent_validation': False,
            'selected_runtime_quality_delta': 0.0,
            'scope_note': 'Historical fitted-model evidence; NEXT3 runtime equality is checked separately.'})
    expected_splits = {'ALT_1', 'ALT_2', 'ALT_3'} if expected_kind == 'REFITTED_ALTERNATE_SPLIT' else {'LEAVE_ONE_H_REGIME_OUT'}
    for name in role_names:
        for split in expected_splits:
            fold_rows = [r for r in selected if r['candidate'] == name and r['split'] == split]
            if len(fold_rows) != 5 or {int(float(r['fold'])) for r in fold_rows} != set(range(5)):
                raise ValueError('Inherited validation lacks exactly five unique folds')
            scores = [float(r['ts_auc']) for r in fold_rows]
            if any(not math.isclose(float(r['mean']), statistics.mean(scores), rel_tol=0, abs_tol=1e-14)
                   or float(r['worst']) != min(scores) for r in fold_rows):
                raise ValueError('Inherited summary and fold scores disagree')
    return selected


def selected_identity_pair(reference):
    name = reference['experiment_id']; score = reference['full']['mean']
    return {'candidate': name, 'reference': name, 'scope': 'all_time',
        'record_type': 'SELECTED_PERFORMANCE_VS_SESSION_START_IDENTICAL_SCIENCE',
        'candidate_mean': score, 'reference_mean': score, 'estimate_mean_delta': 0.0,
        'bootstrap_mean_delta': 0.0, 'bootstrap_median_delta': 0.0,
        'ci_2_5': 0.0, 'ci_97_5': 0.0, 'probability_delta_positive': 0.0,
        'replicates': 0, 'new_draws': 0, 'analytic_identity': True,
        'selection_adjusted': False, 'validation_kind': 'IDENTICAL_FROZEN_PREDICTIONS',
        'scope_note': 'Exact same predictions imply zero contrast for every possible draw. P(delta > 0) uses strict positivity; no new bootstrap was run.'}


def append_history(original, rows):
    if not original.endswith((b'\n', b'\r')):
        raise ValueError('Original CSV must end with a line boundary')
    reader = csv.DictReader(io.StringIO(original.decode('utf-8-sig')))
    fields = reader.fieldnames; prior = list(reader)
    existing = {row['experiment_id'] for row in prior}
    ids = [row['experiment_id'] for row in rows]
    if len(ids) != len(set(ids)) or existing.intersection(ids):
        raise ValueError('Appended experiment identities collide with preserved history')
    if any(set(row)-set(fields) for row in rows):
        raise ValueError('Append would require changing the old CSV header')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writerows(rows)
    result = original+stream.getvalue().encode('utf-8')
    parsed = list(csv.DictReader(io.StringIO(result.decode('utf-8-sig'))))
    if parsed[:len(prior)] != prior or len(parsed) != len(prior)+len(rows):
        raise ValueError('Append failed to preserve every old field')
    return result

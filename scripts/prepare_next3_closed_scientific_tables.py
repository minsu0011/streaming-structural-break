"""Prepare reviewable closed-science tables without accepting final delivery."""
from pathlib import Path
import argparse
import csv
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.result_tables import csv_bytes, close_recipe, inherited_validation, selected_identity_pair, append_history

INDEX = 'artifacts/next3/ledger/failure_index_v2_policy_schema/RESULTS.json'


class Evidence:
    def __init__(self): self.files = {}

    def bind(self, relative):
        path = ROOT/relative; self.files[relative] = digest(path); return path

    def json(self, relative): return read_json(self.bind(relative))

    def csv(self, relative):
        with self.bind(relative).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def completed_table(self, relative):
        path = ROOT/relative; receipt_relative = (path.parent/'RESULTS.json').relative_to(ROOT).as_posix()
        receipt = self.json(receipt_relative)
        if receipt['status'] != 'PASS': raise RuntimeError('A partial forensic table cannot be published')
        if receipt['files_sha256'][path.name] != digest(path): raise RuntimeError('Committed table bytes changed')
        return [{**row, 'source_receipt': receipt_relative, 'source_receipt_sha256': self.files[receipt_relative]}
                for row in self.csv(relative)]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--label', default='v1'); args = parser.parse_args()
    if not args.label.isidentifier(): raise ValueError('Safe one-shot preview label required')
    out = BASE/'final_tables'/('prepared_'+args.label); out.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    references = evidence.json('artifacts/next3/preflight/FROZEN_REFERENCES.json')
    frozen_roles = evidence.json('configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
    assert frozen_roles['status'] == 'PRE_AUDIT_SCIENTIFIC_ROLES_FROZEN'
    assert frozen_roles['scientific_role_count'] == 2 and not frozen_roles['new_scientific_search_authorized']
    roles = {name: references['roles'][name] for name in ('FROZEN_DISTILLED', 'FROZEN_PERFORMANCE')}
    performance = roles['FROZEN_PERFORMANCE']; names = {role['experiment_id'] for role in roles.values()}
    index = evidence.json(INDEX); assert index['status'] == 'FROZEN_CLOSED_SCIENTIFIC_FAILURE_INDEX'
    recipes = {r['experiment_id']: r for r in index['recipes']}; assert len(recipes) == 13
    for recipe in recipes.values():
        for path, sha in recipe['recipe_evidence_binding']['stage_evidence_sha256'].items():
            assert digest(evidence.bind(path)) == sha
        relative = 'artifacts/next3/experiments/'+recipe['experiment_id']+'/RESULTS.json'
        assert digest(evidence.bind(relative)) == recipe['current_result_sha256']
    ongoing = evidence.csv('MODEL_RESULTS_NEXT3.csv')
    assert {r['experiment_id'] for r in ongoing} == set(recipes)
    models = [close_recipe(row, recipes[row['experiment_id']]) for row in ongoing]
    for name, role in roles.items():
        full = role['full']
        models.append({'experiment_id': role['experiment_id'], 'generation': 'NEXT2_INHERITED',
            'current_stage': 'full', 'status': 'FROZEN_SCIENTIFIC_ROLE', 'final_disposition': 'FROZEN_SCIENTIFIC_ROLE',
            'record_type': 'INHERITED_FROZEN_SCIENTIFIC_ROLE', 'full_mean': full['mean'],
            'full_median': full['median'], 'full_worst': full['worst_fold'],
            'delta_vs_frozen_performance': full['mean']-performance['full']['mean'],
            'selected_role': name, 'scientific_role_selected': True, 'new_scientific_hypothesis_count': 0,
            'new_learner_fits': 0, 'new_gate_fits': 0, 'new_residual_fits': 0, 'promoted': False,
            'source_receipt': 'artifacts/next3/preflight/FROZEN_REFERENCES.json',
            'source_receipt_sha256': evidence.files['artifacts/next3/preflight/FROZEN_REFERENCES.json'],
            'seal_rows': 0, 'scope_note': 'Scientific selection fixed; final runtime delivery requires fresh raw reproduction and audit.'})
    for name, disposition in (('I1_SIGN_PRODUCTS', 'EXACT_PRIOR_FAILURE_REUSED_ZERO_FITS'),
                              ('S4_CONDITIONAL_RATIO_DOWN', 'DEFERRED_WITHOUT_FIT')):
        # Neither unexecuted menu item has an eight-key new-fit hypothesis.
        # Preserve the real reuse/closure receipts instead of fabricating one.
        if name == 'I1_SIGN_PRODUCTS':
            evidence.json('artifacts/next3/reuse/I1_SIGN_PRODUCTS.json')
            description = 'Exact prior C211 sign-product failure reused; no new fit or new score.'
        else:
            evidence.json('artifacts/next3/bottlenecks/FINITE_MENU_CLOSURE_REVIEW.json')
            description = 'Conditional-ratio-down menu item deferred without fitting; no empirical result.'
        models.append({'experiment_id': name, 'generation': 'NEXT3', 'status': disposition,
            'final_disposition': disposition, 'record_type': 'PRIOR_FAILURE_REUSE_OR_DEFERRED',
            'hypothesis': 'UNEXECUTED_MENU_ITEM', 'change': description,
            'new_scientific_hypothesis_count': 0, 'new_learner_fits': 0, 'new_gate_fits': 0, 'new_residual_fits': 0,
            'promoted': False, 'scientific_role_selected': False, 'seal_rows': 0,
            'source_receipt': INDEX, 'source_receipt_sha256': evidence.files[INDEX]})
    cv = evidence.csv('CV_RESULTS_NEXT3.csv')
    for row in cv:
        row.update(record_type='NEXT3_NEW_SCIENTIFIC_RECIPE', final_disposition=recipes[row['experiment_id']]['disposition'],
                   scientific_role_selected=False, validation_kind='LOCKED_ORIGINAL_GROUP_FOLDS')
    for name, role in roles.items():
        for fold, score in enumerate(role['full']['fold_scores']):
            baseline = performance['full']['fold_scores'][fold]
            cv.append({'experiment_id': role['experiment_id'], 'scope': 'full', 'fold': fold, 'ts_auc': score,
                'reference_ts_auc': baseline, 'delta': score-baseline, 'seal_rows': 0,
                'record_type': 'INHERITED_FROZEN_SCIENTIFIC_ROLE', 'selected_role': name,
                'scientific_role_selected': True, 'validation_kind': 'NEXT2_ORIGINAL_GROUP_FOLDS_NO_NEXT3_REFIT'})
    paired = []
    for path in sorted((BASE/'forensic').glob('*/PAIRED_FRACTURES.csv')):
        for row in evidence.completed_table(path.relative_to(ROOT).as_posix()):
            row.update(record_type='NEXT3_FIXED_OOF_CONDITIONAL_DIAGNOSTIC', selection_adjusted=False,
                validation_kind='ORIGINAL_OOF_H_SLICE_NOT_LEAVE_H_OUT' if row['scope'].startswith('regime_')
                    else 'ORIGINAL_GROUP_5FOLD_OOF_DIAGNOSTIC_SLICE')
            paired.append(row)
    audit_rows = evidence.completed_table('artifacts/next3/pair_matrix_audit/run_v1/CONDITIONAL_SUMMARY_RECONSTRUCTED.csv')
    for summary in audit_rows:
        candidate = summary['candidate']; full = recipes[candidate]['stages']['full']
        matches = [row for row in paired if row['candidate'] == candidate and row['scope'] == 'all_time']
        assert len(matches) <= 1
        if matches:
            row = matches[0]
            for key, other in (('estimate_mean_delta', 'observed_mean_delta'), ('ci_2_5', 'conditional_ci_low'),
                               ('ci_97_5', 'conditional_ci_high'), ('probability_delta_positive', 'probability_delta_positive')):
                assert abs(float(row[key])-float(summary[other])) <= 1e-14
        else:
            row = {'candidate': candidate, 'reference': performance['experiment_id'], 'scope': 'all_time',
                'candidate_mean': full['mean'], 'reference_mean': performance['full']['mean'],
                'candidate_median': full['median'], 'candidate_worst': full['worst_fold'],
                'estimate_mean_delta': summary['observed_mean_delta'], 'ci_2_5': summary['conditional_ci_low'],
                'ci_97_5': summary['conditional_ci_high'], 'probability_delta_positive': summary['probability_delta_positive'],
                'replicates': 2000, 'selection_adjusted': False,
                'source_receipt': summary['source_receipt'], 'source_receipt_sha256': summary['source_receipt_sha256'],
                'record_type': 'NEXT3_FIXED_OOF_CONDITIONAL_DIAGNOSTIC', 'validation_kind': 'ORIGINAL_GROUP_5FOLD_OOF',
                'scope_note': 'Includes rejected full-admission recipes; independent numerical reconstruction, not new generalization validation.'}
            paired.append(row)
        row.update(joint_conditional_envelope_lower=summary['joint_conditional_envelope_lower'],
            joint_conditional_envelope_upper=summary['joint_conditional_envelope_upper'],
            independent_numerical_audit_receipt=summary['source_receipt'],
            independent_numerical_audit_receipt_sha256=summary['source_receipt_sha256'])
    paired.append({**performance['paired_all_time'], 'record_type': 'INHERITED_NEXT2_BLEND_VS_J102',
                   'selection_adjusted': False, 'new_draws': 0,
                   'scope_note': 'Historical NEXT2 gain; this is not the NEXT3 selected-vs-starting-performance delta.'})
    paired.append(selected_identity_pair(performance))
    old_alt = 'ALT_SPLIT_RESULTS_NEXT2.csv'; old_h = 'REGIME_RESULTS_NEXT2.csv'
    alt = inherited_validation(evidence.csv(old_alt), role_names=names, expected_kind='REFITTED_ALTERNATE_SPLIT',
                              source=old_alt, sha=evidence.files[old_alt])
    h = inherited_validation(evidence.csv(old_h), role_names=names, expected_kind='REFITTED_LEAVE_H_REGIME_OUT',
                            source=old_h, sha=evidence.files[old_h])
    # Keep the baseline in every inherited heldout record so a historical
    # BL-vs-J102 gain cannot be misreported as new E003 scientific improvement.
    for row in alt+h: row['delta_vs_same_frozen_role_at_NEXT3_start'] = 0.0
    for split in performance['alternate_splits']:
        observed = sorted([r for r in alt if r['candidate'] == performance['experiment_id'] and r['split'] == split['split']],
                          key=lambda row: int(row['fold']))
        assert [float(r['ts_auc']) for r in observed] == split['fold_scores']
    assert [float(r['ts_auc']) for r in sorted([r for r in h if r['candidate'] == performance['experiment_id']],
        key=lambda row: int(float(row['fold'])))] == performance['heldout_H']['fold_scores']
    mechanism = [r for r in paired if r['scope'].startswith('mechanism_')]
    old_mechanism = 'MECHANISM_RESULTS_NEXT2.csv'
    for row in evidence.csv(old_mechanism):
        if row['candidate'] == performance['experiment_id'] and row['reference'] == roles['FROZEN_DISTILLED']['experiment_id']:
            mechanism.append({**row, 'record_type': 'INHERITED_NEXT2_BLEND_VS_J102', 'selection_adjusted': False,
                'source_table': old_mechanism, 'source_table_sha256': evidence.files[old_mechanism],
                'validation_kind': 'ORIGINAL_GROUP_5FOLD_OOF_DIAGNOSTIC_SLICE'})
    regime = [r for r in paired if r['scope'].startswith('regime_')]+h
    for row in evidence.csv(old_h):
        if row['candidate'] in names and row['record_type'] == 'ORIGINAL_OOF_H_REGIME_SLICE':
            regime.append({**row, 'record_type': 'INHERITED_NEXT2_ORIGINAL_OOF_H_REGIME_SLICE',
                'source_table': old_h, 'source_table_sha256': evidence.files[old_h],
                'validation_kind': 'ORIGINAL_OOF_H_SLICE_NOT_LEAVE_H_OUT'})
    tables = {'MODEL_RESULTS_NEXT3.csv': models, 'CV_RESULTS_NEXT3.csv': cv,
        'PAIRED_BOOTSTRAP_NEXT3.csv': paired, 'ALT_SPLIT_RESULTS_NEXT3.csv': alt,
        'REGIME_RESULTS_NEXT3.csv': regime, 'MECHANISM_RESULTS_NEXT3.csv': mechanism}
    for name, prefixes in (('ROUTER_RESULTS.csv', ('R',)), ('SCALE_REPAIR_RESULTS.csv', ('S',)),
        ('C205_ABLATION_RESULTS.csv', ('P',)), ('DEPENDENCE_SPECIALIST_RESULTS.csv', ('P', 'I')),
        ('ORTHOGONAL_COMPLEMENT_RESULTS.csv', ('O',))):
        tables[name] = [r for r in models if r['experiment_id'].startswith(prefixes) and r['generation'] == 'NEXT3']
    for name in ('COUNTEREXAMPLE_RESULTS_NEXT3.csv', 'LONG_NULL_RESULTS_NEXT3.csv'):
        rows = evidence.csv(name)
        for row in rows:
            row.update(record_type='EXISTING_FIXED_SYNTHETIC_DIAGNOSTIC', scientific_promotion_authority=False,
                scope_note='Synthetic diagnosis only. Null/outlier-only TS-AUC is undefined; no threshold tuning or new scientific promotion.')
        tables[name] = rows
    fit = evidence.json('artifacts/next3/ledger/FIT_INVENTORY.json')
    assert fit['total_fit_completions'] == 129 and fit['old_fits_recounted_as_new'] == 0
    candidate_fits = sum(int(r.get(key, 0)) for r in models for key in ('new_learner_fits', 'new_gate_fits', 'new_residual_fits'))
    assert candidate_fits == 72 and candidate_fits+fit['nested_base_fit_completions']+fit['complete_DEV_refit_completions'] == 129
    ledger_rows = []
    for row in models:
        if row['record_type'] != 'NEXT3_NEW_SCIENTIFIC_RECIPE': continue
        recipe = recipes[row['experiment_id']]
        ledger_rows.append({'experiment_id': row['experiment_id'], 'generation': 'NEXT3', 'phase': 'CLOSED_SCIENTIFIC_RECIPE',
            'hypothesis': recipe['hypothesis']['HYPOTHESIS'], 'change': recipe['hypothesis']['CHANGE'],
            'expected_effect': recipe['hypothesis']['EXPECTED_EFFECT'], 'risks': recipe['hypothesis']['RISKS'],
            'params': json.dumps({'recipe_evidence_binding_sha256': recipe['recipe_evidence_binding_sha256'],
                'hypothesis_sha256': recipe['recipe_evidence_binding']['hypothesis_sha256']}, sort_keys=True),
            'original_stage_status': row['original_stage_status'], 'final_evidence_status': row['final_disposition'],
            'final_disposition': row['final_disposition'], 'status': row['final_disposition'], 'reason': recipe['reason'],
            'full_mean': row.get('full_mean', ''), 'median': row.get('full_median', ''), 'worst_fold': row.get('full_worst', ''),
            'delta': row.get('delta_vs_frozen_performance', ''), 'screen_mean': row.get('screen_mean', ''),
            'screen_delta': row.get('screen_delta', ''),
            'new_fits': sum(int(row.get(key, 0)) for key in ('new_learner_fits', 'new_gate_fits', 'new_residual_fits')),
            'seal_rows': 0, 'reduced_new_uses': 0, 'new_generation_seal_worthy': False,
            'evidence_origin': INDEX, 'experiment_report_sha256': recipe['current_result_sha256'],
            'ledger_section': 'NEXT3_SCIENTIFIC_RECIPES', 'selected_role': 'NONE'})
    for name, count, reason in (('NEXT3_SHARED_NESTED_BASE_OOF', 56,
        'Shared strict inner-OOF base fits, counted once across routers/residual; not 56 scientific candidates.'),
        ('NEXT3_R3_ALL_DEV_GATE_REFIT', 1,
        'One all-DEV frozen depth2 gate refit for synthetic diagnostics; rejected scientific recipe, no promoted role.')):
        ledger_rows.append({'experiment_id': name, 'generation': 'NEXT3', 'phase': 'SHARED_OR_DIAGNOSTIC_FIT_ACCOUNTING',
            'status': 'COMPLETE_NOT_SCIENTIFIC_PROMOTION', 'reason': reason, 'new_fits': count,
            'seal_rows': 0, 'reduced_new_uses': 0, 'evidence_origin': 'artifacts/next3/ledger/FIT_INVENTORY.json',
            'experiment_report_sha256': evidence.files['artifacts/next3/ledger/FIT_INVENTORY.json'],
            'ledger_section': 'NEXT3_SHARED_FIT_ACCOUNTING', 'selected_role': 'NONE'})
    original = evidence.bind('artifacts/next3/preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv').read_bytes()
    ledger_content = append_history(original, ledger_rows)
    assert sum(int(row['new_fits']) for row in ledger_rows) == 129
    files = {}
    for name, rows in tables.items():
        with (out/name).open('xb') as stream: stream.write(csv_bytes(rows))
        files[name] = {'sha256': digest(out/name), 'rows': len(rows)}
    with (out/'EXPERIMENT_LEDGER.csv').open('xb') as stream: stream.write(ledger_content)
    files['EXPERIMENT_LEDGER.csv'] = {'sha256': digest(out/'EXPERIMENT_LEDGER.csv'), 'rows': 213+len(ledger_rows)}
    # No roots are modified here. The final-audit publisher verifies this
    # frozen table snapshot, then copies it and appends the exact old ledger.
    result = {'status': 'PREPARED_CLOSED_SCIENTIFIC_TABLES_NOT_FINAL_DELIVERY', 'created_utc': utc_now(),
        'new_scientific_recipes': 13, 'full_CV_recipes': 12, 'failed_screen_recipes': 1,
        'inherited_frozen_scientific_roles': 2, 'exact_prior_failure_reuses': 1, 'deferred_without_fit': 1,
        'scientific_table_rows': len(models), 'new_scientific_promotions': 0,
        'scientific_fits': 129, 'candidate_direct_fits': candidate_fits, 'new_reproduction_fits': 0,
        'old_ledger_rows_preserved': 213, 'appended_scientific_or_shared_fit_rows': len(ledger_rows),
        'other_diagnostics_and_engineering_history': 'Immutable NEXT3 hypotheses, queue, attempt receipts and maintenance ledgers; no learner fits.',
        'source_evidence_sha256': evidence.files, 'files': files,
        'source_sha256': {path: digest(ROOT/path) for path in ('src/next3/result_tables.py', 'scripts/prepare_next3_closed_scientific_tables.py')},
        'root_files_modified': False, 'final_audit_completed': False, 'raw_data_or_point_predictions_read': False,
        'limitations': ['All inferred quality remains conditional on reused DEV.',
            'Inherited ALT/H are not new independent data or NEXT3 fits.',
            'The analytic selected-versus-start contrast is zero; old BL-versus-J102 gain is historical.',
            'Final raw reproduction and actual archive proof are separate, still mandatory.']}
    atomic_json(out/'RESULTS.json', result, immutable=True)
    progress('Closed-science table preview prepared', '17 role/recipe/reuse/deferred rows; explicit inherited ALT/H, analytic zero selection change, and byte-preserving old ledger preview')
    print('NEXT3 CLOSED SCIENTIFIC TABLES PREPARED', len(models), 'rows;', len(files), 'files;', candidate_fits, 'direct candidate fits', flush=True)


if __name__ == '__main__': main()

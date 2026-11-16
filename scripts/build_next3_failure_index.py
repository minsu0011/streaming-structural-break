"""Bind every closed scientific recipe to its real rejection evidence and protocol."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now

HYPOTHESIS_KEYS = ('HYPOTHESIS', 'BOTTLENECK', 'CHANGE', 'EXPECTED_EFFECT', 'RISKS',
                   'PRIMARY_METRIC', 'SECONDARY_DIAGNOSTIC', 'STOP_RULE')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='v1')
    args = parser.parse_args()
    if not args.label.isidentifier(): raise ValueError('Simple preserved-output label required')
    closure = read_json(BASE/'bottlenecks/FINITE_MENU_CLOSURE_REVIEW.json')
    assert closure['status'] == 'FINITE_SCIENTIFIC_MENU_CLOSED_MAINTENANCE_CONTINUES'
    gate_path = BASE/'selection/NECESSARY_GATES.json'; gates = read_json(gate_path)
    for name, sha in gates['files_sha256'].items(): assert digest(gate_path.parent/name) == sha
    checks = list(csv.DictReader((gate_path.parent/'NECESSARY_GATE_CHECKS.csv').open(encoding='utf-8', newline='')))
    decisions = {row['candidate']: row for row in gates['decisions']}
    assert len(decisions) == 13 and all(not row['promotion'] for row in decisions.values())
    out = BASE/'ledger'/('failure_index_'+args.label); out.mkdir(parents=True, exist_ok=False)
    recipes = []; flat = []
    for name, decision in sorted(decisions.items()):
        directory = BASE/'experiments'/name
        result_path = directory/'RESULTS.json'; result = read_json(result_path)
        hypothesis_path = BASE/'hypotheses'/f'{name}.json'; hypothesis = read_json(hypothesis_path)
        assert all(hypothesis.get(key) for key in HYPOTHESIS_KEYS)
        evidence = {}; stages = {}
        for stage in ('screen', 'full'):
            receipt_path = directory/stage/'RESULTS.json'
            if not receipt_path.is_file(): continue
            receipt = read_json(receipt_path); assert receipt['status'] == 'COMPLETE'
            policy_path = directory/stage/'POLICY.json'
            stage_policy = read_json(policy_path)
            assert stage_policy['hypothesis_sha256'] == digest(hypothesis_path)
            embedded_policy_hash = receipt.get('policy_sha256')
            if embedded_policy_hash is not None:
                assert digest(policy_path) == embedded_policy_hash
            else:
                # Preserve the actual O1 receipt schema; do not invent a missing
                # reverse hash or weaken other candidates' explicit bindings.
                assert name == receipt['kind'] == 'O1_LINEAR_RESIDUAL'
                assert set(stage_policy) == {'created_utc', 'hypothesis_sha256', 'script_sha256', 'nested_sha256'}
                assert stage_policy['script_sha256'] == digest(ROOT/'scripts/evaluate_next3_orthogonal.py')
                assert stage_policy['nested_sha256'] == digest(BASE/'nested'/stage/'RESULTS.json')
            evidence[receipt_path.relative_to(ROOT).as_posix()] = digest(receipt_path)
            evidence[policy_path.relative_to(ROOT).as_posix()] = digest(policy_path)
            stages[stage] = {key: receipt.get(key) for key in
                ('mean', 'median', 'worst_fold', 'mean_delta', 'median_delta', 'worst_delta',
                 'fold_scores', 'reference_scores', 'new_fits', 'new_gate_fits', 'finalist_admitted')}
            stages[stage]['result_embeds_policy_sha256'] = embedded_policy_hash is not None
        failed = {role: [row['gate'] for row in checks if row['candidate'] == name and row['role'] == role and row['state'] == 'FAIL']
                  for role in ('PERFORMANCE', 'ROBUSTNESS')}
        missing = {role: [row['gate'] for row in checks if row['candidate'] == name and row['role'] == role and row['state'] == 'PENDING']
                   for role in ('PERFORMANCE', 'ROBUSTNESS')}
        if decision['current_scope'] == 'screen':
            assert decision['performance_necessary_gates'] == 'SCREEN_FAIL_STOP' and 'full' not in stages
            disposition = 'SCREEN_FAIL_STOP'; reason = 'Fixed screen failed; full OOF was not generated.'
        else:
            assert failed['PERFORMANCE'] and failed['ROBUSTNESS']
            disposition = 'FULL_CV_NO_PROMOTION'
            reason = 'Performance: '+', '.join(failed['PERFORMANCE'])+'; robustness: '+', '.join(failed['ROBUSTNESS'])
        binding = {'candidate_configuration': result.get('candidate'),
                   'hypothesis_sha256': digest(hypothesis_path), 'stage_evidence_sha256': evidence,
                   'research_policy_sha256': digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')}
        recipe = {'experiment_id': name, 'hypothesis': {key: hypothesis[key] for key in HYPOTHESIS_KEYS},
            'disposition': disposition, 'reason': reason, 'stages': stages,
            'failed_necessary_gates': failed, 'not_evaluated_after_rejection': missing,
            'recipe_evidence_binding': binding, 'recipe_evidence_binding_sha256': canonical_hash(binding),
            'current_result_sha256': digest(result_path), 'candidate_configuration': result.get('candidate'),
            'scientific_role_selected': False, 'new_hypothesis_origin': 'Immutable NEXT3 registration; parent template prose is not authority'}
        recipes.append(recipe)
        flat.append({'experiment_id': name, 'bottleneck': hypothesis['BOTTLENECK'],
            'hypothesis': hypothesis['HYPOTHESIS'], 'change': hypothesis['CHANGE'],
            'disposition': disposition, 'failure_reason': reason,
            'screen_delta': stages.get('screen', {}).get('mean_delta'),
            'full_mean': stages.get('full', {}).get('mean'), 'full_delta': stages.get('full', {}).get('mean_delta'),
            'full_median_delta': stages.get('full', {}).get('median_delta'),
            'full_worst_delta': stages.get('full', {}).get('worst_delta'),
            'hypothesis_sha256': digest(hypothesis_path), 'recipe_evidence_binding_sha256': canonical_hash(binding),
            'result_path': result_path.relative_to(ROOT).as_posix(), 'result_sha256': digest(result_path),
            'do_not_repeat_without_new_preregistered_difference': True})
    with (out/'FAILED_RECIPE_LOOKUP.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    prior_path = BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv'
    original = prior_path.read_bytes(); current = (ROOT/'EXPERIMENT_LEDGER.csv').read_bytes()
    assert current.startswith(original)
    prior = list(csv.DictReader(io.StringIO(original.decode('utf-8-sig'))))
    prior_rows = []
    for number, row in enumerate(prior, start=1):
        prior_rows.append({'original_row_number': number, 'experiment_id': row.get('experiment_id'),
            'generation': row.get('generation'), 'phase': row.get('phase'), 'hypothesis': row.get('hypothesis'),
            'change': row.get('change'), 'original_stage_status': row.get('original_stage_status'),
            'final_evidence_status': row.get('final_evidence_status'), 'final_disposition': row.get('final_disposition'),
            'status': row.get('status'), 'original_reason': row.get('original_reason'), 'reason': row.get('reason'),
            'failed_gates': row.get('failed_gates'), 'failed_performance_checks': row.get('failed_performance_checks'),
            'failed_distilled_checks': row.get('failed_distilled_checks'),
            'original_csv_row_canonical_sha256': canonical_hash(row)})
    with (out/'PREVIOUS_LEDGER_LOOKUP.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prior_rows[0])); writer.writeheader(); writer.writerows(prior_rows)
    report = {'status': 'FROZEN_CLOSED_SCIENTIFIC_FAILURE_INDEX', 'created_utc': utc_now(),
        'recipes': recipes, 'new_scientific_recipes': len(recipes),
        'full_CV_rejected': sum(r['disposition'] == 'FULL_CV_NO_PROMOTION' for r in recipes),
        'screen_failures': sum(r['disposition'] == 'SCREEN_FAIL_STOP' for r in recipes),
        'exact_old_failure_reuse_no_new_fit': closure['exact_prior_failure_reuse'],
        'deferred_no_fit': closure['deferred_no_fit'], 'previous_ledger_rows_preserved': len(prior),
        'prior_ledger_sha256': digest(prior_path), 'necessary_gate_receipt_sha256': digest(gate_path),
        'finite_menu_closure_sha256': digest(BASE/'bottlenecks/FINITE_MENU_CLOSURE_REVIEW.json'),
        'source_sha256': digest(Path(__file__)), 'root_experiment_ledger_modified': False,
        'scope': 'Exact evidence lookup, not a new model search. Configuration/protocol hashes bind executed evidence; semantic equivalence across changed prose still requires human review. Old row statuses are retained separately without treating old intermediate FAIL or PASS as a new final verdict.',
        'files_sha256': {name: digest(out/name) for name in ('FAILED_RECIPE_LOOKUP.csv', 'PREVIOUS_LEDGER_LOOKUP.csv')}}
    atomic_json(out/'RESULTS.json', report, immutable=True)
    progress('Failure deduplication index built', f'{len(recipes)} rejected new recipes with actual gates; {len(prior)} old ledger rows indexed without rewriting')
    print('NEXT3 FAILURE INDEX COMPLETE', len(recipes), len(prior), flush=True)


if __name__ == '__main__': main()

"""Expose the probability-floor limit of the registered absolute shock gate."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity


def rows(name):
    with (BASE/name).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def lock():
    path = BASE/'ROLE_GATE_FEASIBILITY_LOCK_NEXT7.json'
    protocol = read_json(BASE/'NEXT7_PROTOCOL_LOCK.json')
    if path.exists():
        assert read_json(path)['source_sha256'] == digest(__file__)
        return
    save(path.name, {'status':'FROZEN', 'created_utc':utc_now(), 'source_sha256':digest(__file__),
        'protocol_sha256':digest(BASE/'NEXT7_PROTOCOL_LOCK.json'), 'unchanged_role_gate':protocol['robustness_role'],
        'origin':'Posthoc design-feasibility audit of the originally agent-quantified absolute0.10 shock reduction. The user requested a large reduction without specifying0.10.',
        'identity':'For a nonnegative candidate probability p, baseline b minus p is at most b. If b is below the required absolute reduction, that empirical endpoint/reference cannot meet the gate even at p=0.',
        'state_references':['NEXT6_HARD4_EVENT'], 'classifier_references':['M0','M3'],
        'cases':['O1','REPEATED_SHOCK'], 'horizons':[1024,4096],
        'scope':'Show every listed endpoint/reference without retroactively choosing one. The concise robustness field did not itself disambiguate state/classifier, overall/treatment-only endpoint or horizon.',
        'no_gate_relaxation_or_new_candidate':True, 'population_probability_claim':False})


def main(lock_only):
    lock()
    if lock_only:
        return
    complete = read_json(BASE/'COMPLETE_DIAGNOSTICS_CLASSIFIERS_NEXT7.json')
    assert complete['status']=='COMPLETE' and not complete['partial_cohorts_included']
    required = float(read_json(BASE/'NEXT7_PROTOCOL_LOCK.json')['robustness_role']['single_and_repeated_shock_false_latch_absolute_reduction_min'])
    inputs = ['SHOCK_EVENT_CONTAMINATION_NEXT7.csv','CLASSIFIER_STRESS_RESULTS_NEXT7.csv',
              'MODEL_COMPARISON_NEXT7.csv','NEXT7_PROTOCOL_LOCK.json']
    result = []
    with activity('Audit attainable absolute shock-reduction gates and independent fold failure',
                  'PROTOCOL_FEASIBILITY_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for kind, filename, references, metrics, value in (
            ('STATE_EVENT_MEMORY','SHOCK_EVENT_CONTAMINATION_NEXT7.csv',('NEXT6_HARD4_EVENT',),
             ('any_latch','excess_latch_vs_paired_control'),'estimate'),
            ('OWN_CALIBRATED_CLASSIFIER','CLASSIFIER_STRESS_RESULTS_NEXT7.csv',('M0','M3'),
             ('any_alarm','treatment_alarm_control_never'),'mean_across_fixed_folds')):
            for r in rows(filename):
                if r['case'] not in ('O1','REPEATED_SHOCK') or r['method'] not in references or r['metric'] not in metrics or r['horizon'] not in ('1024','4096'):
                    continue
                b = None if r[value]=='' else float(r[value])
                if b is not None:
                    assert 0 <= b <= 1
                result.append({'record_type':kind, 'case':r['case'], 'horizon':r['horizon'],
                    'reference':r['method'], 'endpoint':r['metric'], 'empirical_reference_probability':b,
                    'registered_required_absolute_reduction':required,
                    'maximum_possible_reduction_at_zero_candidate_probability':b,
                    'empirically_attainable_under_probability_floor':b>=required if b is not None else None,
                    'missing_endpoint_not_imputed':b is None,
                    'no_new_reference_or_endpoint_selected_for_promotion':True})
        # M3 intentionally lacks some historical paired-control endpoint fields.
        assert len(result)==24 and sum(r['missing_endpoint_not_imputed'] for r in result)==4
        candidates = [r for r in rows('MODEL_COMPARISON_NEXT7.csv') if r['new_NEXT7_scientific_candidate']=='True']
        assert len(candidates)==15 and all(float(r['worst_matched_fold_delta_M0'])<0 for r in candidates)
        csv_write('ROLE_GATE_EMPIRICAL_FEASIBILITY_NEXT7.csv', result)
        save('ROLE_GATE_FEASIBILITY_DONE_NEXT7.json', {'status':'COMPLETE', 'created_utc':utc_now(),
            'lock_sha256':digest(BASE/'ROLE_GATE_FEASIBILITY_LOCK_NEXT7.json'),
            'inputs_sha256':{name:digest(BASE/name) for name in inputs}, 'rows':len(result),
            'all15_candidates_fail_user_required_worst_fold_nonworse_independently':True,
            'registered_absolute_reduction_is_agent_quantification_not_user_numeric_requirement':True,
            'gate_changed':False, 'new_model_or_draw_or_interval':False,
            'reference_or_endpoint_ambiguity_not_resolved_by_outcome_selection':True,
            'output_sha256':digest(BASE/'ROLE_GATE_EMPIRICAL_FEASIBILITY_NEXT7.csv')})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    main(parser.parse_args().lock_only)

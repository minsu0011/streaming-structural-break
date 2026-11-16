"""Conditional fixed validation only, no new scientific recipe generation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    path=BASE/'diagnostic_policy/CONDITIONAL_ROBUSTNESS_VALIDATION.json';policy=read_json(path)
    for name,sha in policy['execution_source_sha256'].items():assert digest(ROOT/name)==sha
    for candidate in policy['candidates']:
        identifier=candidate+'_full_fold0';synthetic=BASE/'synthetic/candidates'/identifier/'RESULTS.json'
        decision=BASE/'robustness_screen'/(candidate+'.json')
        if decision.exists() or not synthetic.exists():continue
        result=read_json(synthetic);assert result['status']=='PASS'
        assert result['training_scope']=='full_fold0'
        admitted=result['ARCH_decrease_each_delta_positive'] and result['heavy_tail_decrease_each_delta_positive']
        atomic_json(decision,{'created_utc':utc_now(),'candidate':candidate,'policy_sha256':digest(path),
            'synthetic_result_sha256':digest(synthetic),'synthetic_training_scope':'full_fold0',
            'conditional_ALT_H_admission':bool(admitted),'disposition':'FIXED_ALT_H_NOMINATED' if admitted else 'STOP_ROBUSTNESS_FIXED_SYNTHETIC_GATE',
            'new_candidate_identity':False,'gate_changed':False},immutable=True)
        if not admitted:
            print('NEXT4 ROBUSTNESS SYNTHETIC STOP',candidate,flush=True);continue
        evidence=policy['original_evidence'][candidate]
        full=BASE/'experiments'/candidate/'full/RESULTS.json';reg=BASE/'registrations'/(candidate+'.json')
        assert digest(full)==evidence['full_result_sha256'] and digest(reg)==evidence['registration_sha256']
        nomination=BASE/'robustness_nominations'/(candidate+'.json')
        atomic_json(nomination,{'created_utc':utc_now(),'candidate':candidate,'role_under_review':'NEXT4_ROBUSTNESS',
            'candidate_registration_sha256':digest(reg),'primary_result_sha256':digest(full),
            'conditional_policy_sha256':digest(path),'decision_sha256':digest(decision),
            'original_evidence':evidence,'synthetic_result_sha256':digest(synthetic),
            'split_menu':['ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT'],'intended_candidate_refits':20,
            'new_frozen_base_fits':0,'new_candidate_identity':False,'primary_performance_gate_override_allowed':False,
            'automatic_promotion_allowed':False,'execution_source_sha256':policy['execution_source_sha256'],
            'stop_rule':'Exactly four original grouped partitions, unchanged base recipe and fixed blend. No alternative parameters or split seeds.'},immutable=True)
        register_job({'id':'RB_'+candidate,'kind':'validation','candidate_id':candidate,'family':read_json(reg)['family'],
            'priority':115,'depends_on_success':['SYN_'+identifier],'argv':['scripts/robustness_next4_base.py','--candidate',candidate],
            'description':'Conditional unchanged ALT/H replication after supported real SCALE and fixed synthetic decrease gates',
            'retry_safe':False,'nomination_sha256':digest(nomination)})
        print('NEXT4 CONDITIONAL ALT/H QUEUED',candidate,flush=True)
    progress('Conditional robustness dispatch checked','Only completed preregistered synthetic gates can trigger existing-recipe ALT/H refits')


if __name__=='__main__':main()

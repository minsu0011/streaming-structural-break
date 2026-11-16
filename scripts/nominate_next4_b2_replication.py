"""Freeze a borderline candidate for independent partition diagnostics only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    candidate='B2_LOG_ENERGY_SEMITAIL';full_path=BASE/'experiments'/candidate/'full/RESULTS.json';full=read_json(full_path)
    boot_path=BASE/'bootstrap'/candidate/'RESULTS.json';boot=read_json(boot_path)
    assert full['performance_numeric_admission'] and not boot['performance_primary_and_bootstrap_admission']
    assert all(s>r for s,r in zip(full['fold_scores'],full['reference_fold_scores']))
    sources=['scripts/robustness_next4_base.py','src/next4/robustness.py','src/next4/cache.py','src/next4/model.py',
        'src/next/bootstrap.py','src/validation/group_bootstrap.py','src/next2/blend.py','src/scoring/ts_auc.py']
    path=BASE/'robustness_nominations'/(candidate+'.json')
    atomic_json(path,{'created_utc':utc_now(),'candidate':candidate,'primary_result_sha256':digest(full_path),
        'candidate_registration_sha256':digest(BASE/'registrations'/(candidate+'.json')),
        'paired_primary_sha256':digest(boot_path),'mean_gain':full['mean_delta'],'primary_ci_lower':boot['paired_all_time']['ci_2_5'],
        'nomination_reason':'All five original folds improve and mean narrowly exceeds 0.0013, but the 95% CI includes zero. Existing ALT/H fits quantify fragility of this unchanged representation.',
        'primary_performance_admission':False,'primary_gate_override_allowed':False,'automatic_promotion_allowed':False,
        'split_menu':['ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT'],'intended_candidate_refits':20,'new_frozen_base_fits':0,
        'new_candidate_identity':False,'fixed_model_feature_and_blend_recipe_unchanged':True,
        'paired_draws_each_split':2000,'bootstrap_seed_policy_sha256':digest(ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json'),
        'execution_source_sha256':{p:digest(ROOT/p) for p in sources},
        'stop_rule':'One unchanged candidate across all four original splits. No selecting split seeds, refit parameters or replacing failed splits. Preserve primary CI failure regardless of replication scores.'},immutable=True)
    register_job({'id':'V001_STRESS_COORDINATES','kind':'validation','priority':45,'argv':['scripts/audit_next4_stress_coordinates.py'],
        'description':'Check original ALT/H grouped partitions and frozen component coordinates before new refits','retry_safe':False})
    register_job({'id':'B401_FIXED_ALT_H_REPLICATION','kind':'validation','candidate_id':candidate,'family':'B','priority':60,
        'depends_on_success':['V001_STRESS_COORDINATES','B202_BOOTSTRAP'],
        'argv':['scripts/robustness_next4_base.py','--candidate',candidate],
        'description':'All 20 fixed ALT/H refits for unchanged B2; primary CI failure retained','retry_safe':False,
        'nomination_sha256':digest(path)})
    progress('B2 fixed partition replication registered','No primary-gate exception; 20 unchanged candidate refits, zero repeated frozen base fits')
    print('NEXT4 B2 FIXED ALT/H REPLICATION QUEUED',flush=True)


if __name__=='__main__':main()

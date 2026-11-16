"""Validate the unchanged B1 SCALE-robustness candidate on all existing splits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    candidate='B1_SPLIT_ENERGY_BRANCHES';fp=BASE/'experiments'/candidate/'full/RESULTS.json'
    diagnostic=BASE/'forensic'/candidate/'RESULTS.json';forensic=read_json(diagnostic)
    assert forensic['robustness_primary_scale_admission']
    assert forensic['paired_scale']['supported'] and forensic['paired_scale']['ci_2_5']>0
    sources=['scripts/robustness_next4_base.py','src/next4/robustness.py','src/next4/cache.py','src/next4/model.py',
        'src/next/bootstrap.py','src/validation/group_bootstrap.py','src/next2/blend.py','src/scoring/ts_auc.py']
    path=BASE/'robustness_nominations'/(candidate+'.json')
    atomic_json(path,{'created_utc':utc_now(),'candidate':candidate,'candidate_registration_sha256':digest(BASE/'registrations'/(candidate+'.json')),
        'primary_result_sha256':digest(fp),'paired_scale_result_sha256':digest(diagnostic),
        'role_under_review':'NEXT4_ROBUSTNESS','primary_performance_admission':False,
        'nomination_reason':'Overall delta +0.000547827 with supported SCALE delta +0.010265491 and CI [0.000641140,0.020910214]. Original other supported mechanism losses exceed neither -0.005 nor the overall -0.0005 gate. Continue fixed replication and synthetic tests; no promotion yet.',
        'split_menu':['ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT'],'intended_candidate_refits':20,
        'new_frozen_base_fits':0,'new_candidate_identity':False,'fixed_model_feature_and_blend_recipe_unchanged':True,
        'remaining_gate_scope':'Fixed matched-fold-0 synthetic ARCH/tail decrease diagnostics and causal/runtime checks. If eventually selected, validate a fresh all-DEV reproduction before claiming deployed robustness.',
        'primary_gate_override_allowed':False,'automatic_promotion_allowed':False,'paired_draws_each_split':2000,
        'execution_source_sha256':{name:digest(ROOT/name) for name in sources},
        'stop_rule':'Exactly four existing partitions and one unchanged B1 recipe. No parameters, alternate split seeds or mechanism-based inference inputs.'},immutable=True)
    register_job({'id':'B601_ROBUSTNESS_ALT_H','kind':'validation','candidate_id':candidate,'family':'B','priority':110,
        'depends_on_success':['V001_STRESS_COORDINATES','B301_FORENSIC'],
        'argv':['scripts/robustness_next4_base.py','--candidate',candidate],
        'description':'20 fixed ALT/H fits for supported B1 SCALE robustness; unchanged candidate, no new frozen fits',
        'retry_safe':False,'nomination_sha256':digest(path)})
    progress('B1 robustness replication nominated','Supported real SCALE improvement, fixed 20 existing partition refits; no performance gate exception')
    print('NEXT4 B1 ROBUSTNESS ALT/H QUEUED',flush=True)


if __name__=='__main__':main()

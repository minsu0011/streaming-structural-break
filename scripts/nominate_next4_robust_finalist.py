"""Follow the frozen SCALE-lower-bound ordering; never invent a quality gate."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review-tag',required=True);args=parser.parse_args()
    path=BASE/'gate_reviews'/args.review_tag/'RESULTS.json';review=read_json(path)
    assert review['status']=='SCIENTIFIC_GATE_REVIEW_NOT_FINAL_SELECTION'
    for name,sha in review['evidence_sha256'].items():assert digest(ROOT/name)==sha
    potential=[r for r in review['rows'] if not r['robustness_failed_gates']]
    potential.sort(key=lambda r:(-r['SCALE_CI_lower'],-r['primary_gain'],r['candidate']))
    if not potential:print('NEXT4 NO ROBUSTNESS NOMINEE',flush=True);return
    while potential:
        chosen=potential[0];cid=chosen['candidate'];prior=BASE/'finalist_nominations'/(cid+'.json')
        if not prior.exists():break
        runtime=BASE/'latency'/(cid+'_DIRECT')/'RESULTS.json';syn=BASE/'synthetic/candidates'/(cid+'_all_dev')/'RESULTS.json'
        if not runtime.exists() or not syn.exists():
            print('NEXT4 NOMINATED FINALIST VALIDATION STILL RUNNING',cid,flush=True);return
        timing=read_json(runtime);stress=read_json(syn);assert timing['status']=='PASS' and stress['status']=='PASS'
        failed=[]
        if not timing['direct_runtime_within_3x']:failed.append('ORIGINAL_DIRECT_RUNTIME_GATE')
        if not stress['ARCH_decrease_each_delta_positive']:failed.append('FINAL_ARTIFACT_ARCH_DECREASE_CONFIRMATION')
        if not stress['heavy_tail_decrease_each_delta_positive']:failed.append('FINAL_ARTIFACT_HEAVY_TAIL_DECREASE_CONFIRMATION')
        if not failed:
            print('NEXT4 HIGHEST VALIDATED ROBUSTNESS FINALIST RETAINED',cid,flush=True);return
        rejection=BASE/'finalist_rejections'/(cid+'.json')
        if not rejection.exists():atomic_json(rejection,{'created_utc':utc_now(),'candidate':cid,'failed_conditions':failed,
            'runtime_sha256':digest(runtime),'synthetic_sha256':digest(syn),'new_recipe_allowed':False},immutable=True)
        potential.pop(0)
    if not potential:print('NEXT4 ALL ROBUSTNESS FINALISTS REJECTED',flush=True);return
    chosen=potential[0];cid=chosen['candidate']
    if not chosen['robustness_fixed_real_and_synthetic_admission']:
        print('NEXT4 WAIT FOR HIGHER PRIORITY ROBUSTNESS EVIDENCE',cid,flush=True);return
    reg=BASE/'registrations'/(cid+'.json')
    sources=['src/next4/robust_finalist.py','next4_robustness_main.py','scripts/materialize_next4_robust_finalist.py',
        'scripts/verify_next4_candidate_api.py','scripts/verify_next4_robust_cold.py','scripts/profile_next4_robust_finalist.py',
        'scripts/register_next4_finalist_stress.py','scripts/diagnose_next4_finalist_memory.py',
        'scripts/profile_next4_b2_direct.py','src/next4/model.py','src/next4/cache.py','src/next2/blend.py',
        'src/next2/shared_blend_runtime.py','src/next2/dependence_runtime.py','src/next/readonly_bootstrap.py']
    nomination=BASE/'finalist_nominations'/(cid+'.json')
    atomic_json(nomination,{'created_utc':utc_now(),'candidate':cid,'role_under_review':'NEXT4_ROBUSTNESS',
        'registration_sha256':digest(reg),'scientific_gate_review_sha256':digest(path),'original_evidence':chosen,
        'ordering':'Previously fixed highest original supported SCALE CI lower bound, then primary gain, then ID.',
        'execution_sources_sha256':{name:digest(ROOT/name) for name in sources},
        'latency_policy_sha256':digest(ROOT/'configs/next_latency_policy.json'),
        'research_paired_run_order':[['BL80','ROBUSTNESS'],['ROBUSTNESS','BL80']],
        'audit_paired_run_order':[['BL80','ROBUSTNESS']],
        'timing_scope':'Two fixed paired actual-10M runs in research, one additional fixed paired repeat in separate final audit. Original 128-series sample; no timing changes or optimizations.',
        'new_candidate_identity':False,'new_frozen_base_fits':0,'all_DEV_fits_intended':1,
        'conditional_fresh_raw_reproduction_fits':1,'promotion':False,
        'actual_artifact_synthetic_confirmation':'Repeat the unchanged full all-DEV synthetic suite and original memory arithmetic on this fitted artifact. Require both original ARCH and heavy-tail decrease conditions to remain positive; no altered seed, scenario or threshold. If confirmation fails, retain the failure and consider only the next originally ranked candidate.',
        'stop_rule':'Nomination is not promotion. Require direct/API/serialization/cold/runtime validation and final audit; retain every failed scientific gate and reported ALT/H limitation.'},immutable=True)
    jobs=[('RF_'+cid,[],['scripts/materialize_next4_robust_finalist.py','--candidate',cid],181),
        ('RN_'+cid,['RF_'+cid],['scripts/register_next4_finalist_stress.py','--candidate',cid],181),
        ('RV_'+cid,['RF_'+cid],['scripts/verify_next4_candidate_api.py','--model',cid,'--tag',cid],184),
        ('RC_'+cid,['RV_'+cid],['scripts/verify_next4_robust_cold.py','--candidate',cid],185),
        ('RL_'+cid,['RC_'+cid],['scripts/profile_next4_robust_finalist.py','--candidate',cid],186)]
    for identifier,deps,argv,priority in jobs:
        register_job({'id':identifier,'kind':'validation','candidate_id':cid,'priority':priority,
            'depends_on_success':deps,'argv':argv,'description':'Unchanged nominated robustness artifact '+identifier,
            'retry_safe':False,'nomination_sha256':digest(nomination)})
    progress('Robustness deployment-validation nominated',cid+' original SCALE-CI ordering; fixed all-DEV fit, 300 prefixes, cold determinism and two paired 10M timing runs')
    print('NEXT4 ROBUSTNESS FINALIST NOMINATED',cid,flush=True)


if __name__=='__main__':main()

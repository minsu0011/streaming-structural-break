"""Freeze expanded prefix/API checks and a finite contemporaneous timing protocol."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    directory=BASE/'refits/B2_LOG_ENERGY_SEMITAIL/diagnostic_all_dev_v1';receipt=read_json(directory/'RESULTS.json')
    assert receipt['status']=='PASS'
    names=['src/next4/b2_inference.py','next4_b2_diagnostic_main.py','scripts/verify_next4_b2_direct.py',
        'scripts/profile_next4_b2_direct.py','src/next4/model.py','src/next4/cache.py','src/next4/synthetic.py',
        'src/next2/blend.py','src/next2/shared_blend_runtime.py','src/next2/dependence_runtime.py',
        'scripts/verify_next2_shared_public_api.py']
    path=BASE/'diagnostic_policy/B2_RUNTIME_POLICY.json'
    atomic_json(path,{'created_utc':utc_now(),'candidate':'B2_LOG_ENERGY_SEMITAIL','model_directory':directory.relative_to(ROOT).as_posix(),
        'model_sha256':digest(directory/'model.joblib'),'model_receipt_sha256':digest(directory/'RESULTS.json'),
        'sources_sha256':{name:digest(ROOT/name) for name in names},'prefix_cases':300,'prefix_seed':20260908,
        'prefix_design':'Changed suffix and shorter horizon, same H/current prefix; H lengths 192/256/512/1024, Student-5 scale/location variation, finite current-point and batch parity.',
        'real_protocol':'All 8000 DEV current-point outputs against cached registered B2 features plus causal canonical C205; first None before dataset access, future length/suffix traps, filesystem write trap, reverse-order 20 series.',
        'latency_policy_sha256':digest(ROOT/'configs/next_latency_policy.json'),
        'paired_run_order':[['BL80','B2'],['B2','BL80'],['BL80','B2'],['B2','BL80'],['BL80','B2']],
        'timing_rule':'Five fixed paired repetitions. Existing fixed sample and actual 10M updates on one state. Every timed run and 1M checkpoint rejects other non-ancestor workspace Python processes.',
        'new_fits':0,'new_optimization':False,'promotion':False,'stop_rule':'Do not vary parameters, sample, repetitions or run order after timing. Preserve slow results and all statistical gate failures.'},immutable=True)
    register_job({'id':'RT001_B2_DIRECT_CAUSALITY','kind':'validation','priority':135,
        'depends_on_success':['B501_ALL_DEV_DIAGNOSTIC_REFIT'],'argv':['scripts/verify_next4_b2_direct.py'],
        'description':'300 expanded causal prefixes and every DEV point through B2 research-only direct entry','retry_safe':False,'policy_sha256':digest(path)})
    register_job({'id':'LT001_B2_DIRECT_TIMING','kind':'validation','priority':145,
        'depends_on_success':['RT001_B2_DIRECT_CAUSALITY'],'argv':['scripts/profile_next4_b2_direct.py'],
        'description':'Five paired direct scalar 10M timing runs, contemporaneous BL80 and B2','retry_safe':False,'policy_sha256':digest(path)})
    progress('Direct causal/runtime verification preregistered','Expanded 300 prefix cases, 4M-point API parity, five fixed paired scalar timings; no engineering optimization')
    print('NEXT4 B2 DIRECT VALIDATION AND TIMING QUEUED',flush=True)


if __name__=='__main__':main()

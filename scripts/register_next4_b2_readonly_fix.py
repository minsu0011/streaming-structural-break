"""Preserve a cold-import failure and register an additive read-only entry."""
from pathlib import Path
import sys
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    failed=BASE/'jobs/CI001_COLD_API_DETERMINISM/attempt_1/RUN.log'
    trace=BASE/'maintenance/COLD_API_FAILURE_001/EXCEPTION.json';r=read_json(trace)
    assert r['status']=='REPRODUCED_EXCEPTION' and 'os.mkdir' in r['exception_message']
    out=trace.parent;preserved={}
    for path in (ROOT/'scripts/verify_next4_cold_api.py',ROOT/'next4_b2_diagnostic_main.py',failed):
        target=out/(path.name+'.preserved');assert not target.exists();shutil.copyfile(path,target)
        assert digest(path)==digest(target);preserved[path.relative_to(ROOT).as_posix()]=digest(path)
    names=['next4_b2_readonly_diagnostic_main.py','scripts/verify_next4_b2_readonly.py','src/next/readonly_bootstrap.py']
    original=read_json(BASE/'diagnostic_policy/B2_RUNTIME_POLICY.json')
    unchanged={name:value for name,value in original['sources_sha256'].items()}
    for name,value in unchanged.items():assert digest(ROOT/name)==value
    path=BASE/'diagnostic_policy/B2_READONLY_ENTRY_V2.json'
    atomic_json(path,{'created_utc':utc_now(),'source_sha256':{name:digest(ROOT/name) for name in names},
        'unchanged_scientific_source_sha256':unchanged,'preserved_failure_source_sha256':preserved,
        'failure_trace_sha256':digest(trace),'original_runtime_policy_sha256':digest(BASE/'diagnostic_policy/B2_RUNTIME_POLICY.json'),
        'cause':'Cold import of src.next4.model reaches a cache=True project Numba decorator before the later read-only cache disabling code. Its cache locator attempts os.makedirs.',
        'fix':'Use the existing pinned read-only bootstrap before importing the original B2 infer function, through a new entry file. No source or fitted parameters of the scientific model are edited.',
        'new_fits':0,'new_candidate_identity':False,'promotion':False,'quality_or_gate_changes':False,
        'validation':'Two new fresh processes, write audit enabled before entry import, same core function identity, direct prepared equality and repeated fingerprints.'},immutable=True)
    register_job({'id':'CI002_B2_READONLY_COLD_VALIDATION','kind':'validation','priority':110,
        'argv':['scripts/verify_next4_b2_readonly.py'],'description':'Validate additive read-only B2 import repair with original failure preserved',
        'retry_safe':False,'policy_sha256':digest(path)})
    progress('Cold B2 import repair registered','Preserved original failure and exact trace; new additive entry uses existing read-only bootstrap, original model/runtime sources unchanged')
    print('NEXT4 B2 READONLY REPAIR QUEUED',flush=True)


if __name__=='__main__':main()

"""Create a provisional research runtime artifact only after its fixed timing gate."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.exact_inference import write_artifact
from src.next2.canonical_model import load


def main():
    timing=BASE/'engineering/precomputed_shared/timing/RESULTS.json';r=read_json(timing)
    assert r['status']=='PASS' and r['timing_gate_pass'] and all(r['criteria'].values())
    if (timing.parent/'MEASUREMENT_ACTIVITY_CAVEAT.json').exists():
        raise RuntimeError('Known timing overlap requires a documented clean validation before materialization')
    for name,sha in r['files_sha256'].items():assert digest(timing.parent/name)==sha
    original=ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    frozen=read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256']
    assert digest(original)==frozen[original.relative_to(ROOT).as_posix()]
    out=BASE/'canonical_deployment/PERFORMANCE_precomputed_shared_v1'
    descriptor=write_artifact(load(original),out,read_json(ROOT/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json'),
        origin='EXACT_NEXT3_RUNTIME_FROM_UNCHANGED_FROZEN_PERFORMANCE_MODEL')
    assert descriptor['model_sha256']==digest(original)
    atomic_json(out/'ASSEMBLY.json',{'status':'PASS','created_utc':utc_now(),'engineering_timing_receipt_sha256':digest(timing),
        'model_bytes_identical_to_frozen':True,'model_sha256':digest(original),'new_fits':0,'final_promotion':False,
        'pending':['Actual read-only public API on every DEV point','Final fresh pinned raw iterator training reproduction','Final audit decision']},immutable=True)
    progress('Provisional exact deployment assembled','Same canonical model bytes; public API and final raw reproduction still required',experiment='E001_PRECOMPUTED_SHARED')
    print('NEXT3 EXACT RESEARCH DELIVERY ASSEMBLED',descriptor['model_sha256'],flush=True)


if __name__=='__main__':main()

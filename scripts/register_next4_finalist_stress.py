"""Bind the complete unchanged synthetic suite to the actual all-DEV artifact."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job
from src.next4.synthetic import sources


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args();cid=args.candidate
    npth=BASE/'finalist_nominations'/(cid+'.json');finalist=read_json(npth)
    for name,sha in finalist['execution_sources_sha256'].items():assert digest(ROOT/name)==sha
    rp=BASE/'registrations'/(cid+'.json');reg=read_json(rp);directory=BASE/'refits'/cid/'nominated_all_dev_v1'
    receipt=read_json(directory/'RESULTS.json');assert receipt['status']=='PASS'
    suitepath=BASE/'diagnostic_policy/SYNTHETIC_POLICY.json';assert read_json(suitepath)['sources_sha256']==sources()
    identifier=cid+'_all_dev';nompath=BASE/'synthetic_nominations'/(identifier+'.json')
    atomic_json(nompath,{'id':identifier,'created_utc':utc_now(),'candidate':cid,'family':reg['family'],
        'training_scope':'all_dev','model_path':(directory/'model.joblib').relative_to(ROOT).as_posix(),
        'model_sha256':digest(directory/'model.joblib'),'registration_sha256':digest(rp),
        'training_result_sha256':digest(directory/'RESULTS.json'),'suite_sha256':digest(suitepath),
        'finalist_nomination_sha256':digest(npth),'new_fits':0,'new_candidate_identity':False,'promotion':False,
        'scope_extension':'Final-artifact confirmation required by the prospective finalist nomination; same 21 scenarios, 4 null regimes, all-DEV sample sizes and matched frozen reference.'},immutable=True)
    memory_original=BASE/'diagnostic_policy/MEMORY_DIAGNOSTIC_POLICY.json';memory=read_json(memory_original)
    memory={**memory,'created_utc':utc_now(),'status':'FROZEN_BEFORE_FINAL_ARTIFACT_SYNTHETIC_SCORES',
        'original_memory_policy_sha256':digest(memory_original),'nominations':{identifier:digest(nompath)},
        'sources_sha256':{**memory['sources_sha256'],'scripts/diagnose_next4_finalist_memory.py':digest(ROOT/'scripts/diagnose_next4_finalist_memory.py')},
        'extension_scope':'Identical memory arithmetic, unchanged fixed thresholds and existing paired generating-seed counts; actual nominated all-DEV artifact only.'}
    memorypath=BASE/'diagnostic_policy'/('MEMORY_FINALIST_'+identifier+'.json');atomic_json(memorypath,memory,immutable=True)
    register_job({'id':'SYN_'+identifier,'kind':'validation','candidate_id':cid,'family':reg['family'],'priority':182,
        'depends_on_success':['RF_'+cid],'argv':['scripts/stress_next4_candidate.py','--nomination',identifier],
        'description':'Full unchanged matched all-DEV final-artifact synthetic and long-null confirmation','retry_safe':False,
        'nomination_sha256':digest(nompath)})
    register_job({'id':'MD_'+identifier,'kind':'validation','candidate_id':cid,'priority':183,
        'depends_on_success':['SYN_'+identifier],'argv':['scripts/diagnose_next4_finalist_memory.py','--nomination',identifier],
        'description':'Original paired outlier and null-memory arithmetic for the nominated all-DEV artifact','retry_safe':False,
        'policy_sha256':digest(memorypath)})
    register_job({'id':'DN_FINAL_'+cid,'kind':'validation','candidate_id':cid,'priority':187,
        'depends_on_success':['RL_'+cid,'SYN_'+identifier],
        'argv':['scripts/advance_next4_finalist_review.py'],
        'description':'Review actual artifact confirmation and only then retain the top role or validate the next originally ranked existing candidate',
        'retry_safe':False,'nomination_sha256':digest(npth)})
    progress('Actual finalist artifact stress registered',cid+' same all-DEV synthetic cohort sizes, cases, thresholds and memory definitions; no new model or seed')
    print('NEXT4 FINAL ARTIFACT STRESS QUEUED',identifier,flush=True)


if __name__=='__main__':main()

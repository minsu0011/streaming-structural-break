"""One best completed recipe per family, before any new ALT/H result."""
from pathlib import Path
import argparse
import ast
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.scheduler import register_job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contracts',action='store_true');args=parser.parse_args()
    if args.contracts:
        for name in ('src/next5/stability.py','scripts/fit_next5_stability.py','scripts/analyze_next5_stability.py'):
            ast.parse((ROOT/name).read_text(encoding='utf-8'),filename=name)
        register_job({'id':'STAB000_CONTRACTS','kind':'validation','priority':35,'depends_on_success':['DIAG002_BOOTSTRAP_CONTRACT'],
            'argv':['-m','pytest','-q','tests/test_next5_stability.py','--junitxml=artifacts/next5/stability/CONTRACTS.xml'],
            'description':'Immutable ALT and H partitions, duplicate-group separation and literal bootstrap parity using original draws','retry_safe':False})
        return
    state=read_json(BASE/'jobs/STAB000_CONTRACTS/STATE.json');assert state['status']=='COMPLETE' and state['exit_code']==0
    xml=BASE/'stability/CONTRACTS.xml';tree=ET.parse(xml);assert not tree.findall('.//failure') and not tree.findall('.//error')
    assert not list((BASE/'stability').glob('**/FIT_START.json'))
    full=[read_json(p) for p in (BASE/'experiments').glob('*/full/RESULTS.json')];nominees={}
    for family in ('S','P','W','E','R','M','X'):
        best=sorted([r for r in full if r['family']==family],key=lambda r:(-r['mean_delta'],r['candidate']))[0];cid=best['candidate']
        nominees[cid]={'family':family,'primary_mean_delta':best['mean_delta'],'registration_sha256':digest(BASE/'registrations'/(cid+'.json')),
            'primary_result_sha256':digest(BASE/'experiments'/cid/'full/RESULTS.json')}
    files=('src/next5/stability.py','scripts/fit_next5_stability.py','scripts/analyze_next5_stability.py',
        'scripts/train_next5_reference.py','src/next5/model.py','src/next5/cache.py','src/next5/data.py','src/next5/guard.py',
        'src/next5/bootstrap.py','src/next/bootstrap.py','src/scoring/ts_auc.py')
    atomic_json(BASE/'stability/POLICY_LOCK.json',{'created_utc':utc_now(),'status':'FROZEN_BEFORE_ALT_OR_H_EXCLUSION_SCORES',
        'diagnostic_nominees':nominees,'nomination_rule':'Highest completed full PRIMARY mean delta within each of seven families; lexical ID tie-break. No failed-screen rescue.',
        'purpose':'Characterize stability and failure mechanisms of the seven distinct scientific approaches. Every primary admission gate remains binding.',
        'splits':['ALT_A','ALT_B','H_EXCLUSION'],'folds':[0,1,2,3,4],
        'reference_recipes':['REF_J102','REF_C205'],'reference_training':'Exactly identical ResearchPool training IDs for each held-out evaluation.',
        'H_partition':'Unchanged historical H-only regime classifier applied to newly computed ResearchPool H descriptors; no H clustering refit.',
        'H_partition_transductive_history':read_json(BASE/'diagnostics/POLICY.json')['historical_transductive_limitation'],
        'split_lock_sha256':digest(ROOT/'NEXT5_SPLIT_LOCK.json'),'diagnostic_lock_sha256':digest(BASE/'diagnostics/RESEARCH_DIAGNOSTIC_LOCK.json'),
        'original_counts_lock_sha256':digest(BASE/'bootstrap/COUNTS_LOCK.json'),'new_real_random_draws':False,
        'bootstrap_design':'Reuse original PRIMARY-stratified whole-series counts by ID. Do not redraw for ALT or H; report the conditional design and variable held-out totals.',
        'new_diagnostic_learner_fits_planned':135,'new_scientific_candidates':0,'new_scientific_hyperparameters':0,
        'execution_sources_sha256':{name:digest(ROOT/name) for name in files},'contract_tests_sha256':digest(xml),
        'shadow_access':False,'cannot_rescue_failed_primary_gate':True},immutable=True)
    for split in ('ALT_A','ALT_B','H_EXCLUSION'):
        for kind in ('J102','C205'):
            argv=['scripts/fit_next5_stability.py','--reference',kind,'--split',split] if split=='H_EXCLUSION' else ['scripts/train_next5_reference.py','--kind',kind,'--scope','full','--split',split]
            register_job({'id':'STAB_REF_'+kind+'_'+split,'kind':'validation','priority':81,'depends_on_success':['STAB000_CONTRACTS'],
                'argv':argv,'description':'Fixed matched '+kind+' training for '+split+' only; no new recipe','retry_safe':False})
        for cid in nominees:
            fit='STAB_'+cid+'_'+split
            register_job({'id':fit,'kind':'validation','priority':82,'depends_on_success':['STAB_REF_J102_'+split,'STAB_REF_C205_'+split],
                'argv':['scripts/fit_next5_stability.py','--candidate',cid,'--split',split],
                'description':'One unchanged family representative under '+split+'; primary gate remains binding','retry_safe':False})
            register_job({'id':'STAB_BOOT_'+cid+'_'+split,'kind':'validation','priority':84,'depends_on_success':[fit],
                'argv':['scripts/analyze_next5_stability.py','--candidate',cid,'--split',split],
                'description':'Same original 2000 draws for fixed held-out OOF '+cid+' '+split,'retry_safe':False})
    progress('Held-out stability menu frozen','Seven family representatives, two locked ALT splits and five historical H exclusions; 135 matched diagnostic refits, no new candidates or real-data resampling')
    print('NEXT5 STABILITY FROZEN',','.join(nominees),flush=True)


if __name__=='__main__':main()

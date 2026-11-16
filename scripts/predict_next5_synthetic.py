"""One owned fitted-fold replay across the complete immutable synthetic menu."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.guard import ResearchGuard
from src.next5.synthetic import load_reference,load_candidate,cohort_menu,cohort_key
from src.next5.synthetic_cache import reference_cohort,candidate_cohort,cached


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',action='store_true');parser.add_argument('--candidate')
    parser.add_argument('--scope',choices=('full','screen'),required=True);parser.add_argument('--fold',type=int,required=True)
    args=parser.parse_args()
    if args.reference==bool(args.candidate):raise ValueError('Exactly one reference or candidate request required')
    if args.fold not in (range(5) if args.scope=='full' else (0,4)):raise ValueError('Only already fitted fixed folds allowed')
    guard=ResearchGuard();menu=read_json(BASE/'synthetic/MENU_LOCK.json');execution=read_json(BASE/'synthetic/EXECUTION_LOCK.json')
    for name,sha in execution['sources_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Synthetic execution source changed after freeze')
    execution_sha=digest(BASE/'synthetic/EXECUTION_LOCK.json');lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    for name,sha in menu['generator_sources_sha256'].items():assert digest(ROOT/name)==sha
    assert digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')==menu['original_generator_policy_sha256']
    if args.reference:
        models,proof=load_reference(args.scope,args.fold);root=BASE/'synthetic/references'/args.scope/('fold_'+str(args.fold))
    else:
        candidate,registration,proof=load_candidate(args.candidate,args.scope,args.fold)
        models,reference_proof=load_reference(args.scope,args.fold)
        if any(m.training['training_ids_sha256']!=candidate.training['training_ids_sha256'] for m in models.values()):
            raise RuntimeError('Candidate and synthetic baseline training IDs differ')
        root=BASE/'synthetic/candidates'/args.candidate/args.scope/('fold_'+str(args.fold))
    if (root/'REPLAY_RESULTS.json').exists():
        raise RuntimeError('Complete fitted-fold replay is not rerun; use its immutable cohort receipts')
    if not (root/'REPLAY_START.json').exists():
        atomic_json(root/'REPLAY_START.json',{'created_utc':utc_now(),'candidate':args.candidate,'reference':args.reference,'scope':args.scope,'fold':args.fold,
            'menu_sha256':digest(BASE/'synthetic/MENU_LOCK.json'),'execution_sha256':execution_sha,'new_fits':0},immutable=True)
    records=[]
    for spec in cohort_menu(menu,lock):
        if args.reference:record=reference_cohort(models,proof,args.scope,args.fold,spec,menu,lock,execution_sha)
        else:
            reference_root=BASE/'synthetic/references'/args.scope/('fold_'+str(args.fold))/cohort_key(spec)
            identity={'scope':args.scope,'fold':args.fold,'cohort':spec,'model_sources':reference_proof,
                'menu_sha256':digest(BASE/'synthetic/MENU_LOCK.json'),'execution_sha256':execution_sha}
            reference=cached(reference_root,identity)
            if reference is None:raise RuntimeError('Completed matched synthetic reference cohort required')
            record=candidate_cohort(candidate,registration,proof,args.scope,args.fold,spec,menu,lock,execution_sha,reference)
        records.append({'cohort':cohort_key(spec),'prediction_file_sha256':record['prediction_file_sha256'],'input_sha256':record['input_sha256']})
        print('NEXT5 SYNTHETIC COHORT COMPLETE',args.candidate or 'REFERENCES',args.scope,args.fold,cohort_key(spec),flush=True)
    guard.check_state()
    atomic_json(root/'REPLAY_RESULTS.json',{'created_utc':utc_now(),'status':'COMPLETE','candidate':args.candidate,'reference':args.reference,
        'scope':args.scope,'fold':args.fold,'cohorts':records,'new_fits':0,'real_Shadow_rows':0,'LOCAL_SEAL_rows':0,
        'execution_sha256':execution_sha,'menu_sha256':digest(BASE/'synthetic/MENU_LOCK.json')},immutable=True)
    progress('Complete fitted-fold synthetic replay',str(args.candidate or 'REFERENCES')+' '+args.scope+' fold '+str(args.fold)+'; exact matched inputs and immutable cohort files')


if __name__=='__main__':main()

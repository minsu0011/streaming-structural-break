"""Freeze existing H-regime transform, post-hoc rules and exact paired counts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.guard import ResearchGuard
from src.validation.group_bootstrap import draw_group_multiplicities


def main():
    guard=ResearchGuard();randomness=read_json(ROOT/'NEXT5_RANDOMNESS_LOCK.json');split=read_json(ROOT/'NEXT5_SPLIT_LOCK.json')
    original=ROOT/'artifacts/next/current_forensic';original_lock=read_json(original/'REGIME_ASSIGNMENT_LOCK.json')
    transform_path=original/'HISTORICAL_TRANSFORM.json'
    assert digest(transform_path)==original_lock['files_sha256'][transform_path.name]
    # This frozen aggregate H-only classifier preserves the original regimes.
    # Never load the old DEV_H_DESCRIPTORS or DEV_SIGNATURES per-series tables.
    transform=read_json(transform_path);policy=read_json(ROOT/'ROBUSTNESS_SPLIT_LOCK.json')
    atomic_json(BASE/'diagnostics/HISTORICAL_TRANSFORM.json',transform,immutable=True)
    record={'created_utc':utc_now(),'status':'FROZEN','mechanism':policy['mechanism'],
       'original_H_transform_sha256':digest(transform_path),'original_regime_lock_sha256':digest(original/'REGIME_ASSIGNMENT_LOCK.json'),
       'H_regime_method':'Apply the unchanged original H-only mean/std, clip [-8,8], and five canonically ordered centers to newly calculated ResearchPool H descriptors.',
       'historical_transductive_limitation':'Original transform was fitted before NEXT5 on DEV H. It is a fixed historical diagnostic, not a new Shadow observation or independently fitted ResearchPool transform.',
       'old_full_DEV_per_series_tables_loaded':False,'Shadow_descriptors_or_outcomes_loaded':False,
       'feature_training_use_forbidden':True,'new_H_clustering_fits':0,
       'mechanism_source_sha256':digest(ROOT/'src/next/regimes.py'),'policy_sha256':digest(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json')}
    atomic_json(BASE/'diagnostics/POLICY.json',record,immutable=True)
    counts_receipts=[]
    for fold in range(5):
        ids=np.asarray(sorted(int(i) for i,f in split['mappings']['PRIMARY'].items() if f==fold),dtype=np.int32)
        guard.admit(ids,purpose='Freeze paired ResearchPool bootstrap group counts')
        groups=np.asarray([guard.split.groups[int(i)] for i in ids]);assert len(np.unique(groups))==len(ids)
        _,counts=draw_group_multiplicities(ids,randomness['bootstrap_replicates'],randomness['bootstrap_seed_base']+fold)
        path=BASE/'bootstrap/counts'/('PRIMARY_fold_'+str(fold)+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise RuntimeError('Never regenerate frozen real-data bootstrap counts')
        np.savez_compressed(path,dataset_id=ids,counts=counts)
        counts_receipts.append({'fold':fold,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'groups':len(ids),'draws':len(counts)-1})
    atomic_json(BASE/'bootstrap/COUNTS_LOCK.json',{'created_utc':utc_now(),'status':'FROZEN','folds':counts_receipts,
       'randomness_sha256':digest(ROOT/'NEXT5_RANDOMNESS_LOCK.json'),'prediction_dtype':'float64, identical to NEXT5 TS_AUC',
       'source_sha256':{'src/next5/bootstrap.py':digest(ROOT/'src/next5/bootstrap.py'),'src/next/bootstrap.py':digest(ROOT/'src/next/bootstrap.py'),
         'src/validation/group_bootstrap.py':digest(ROOT/'src/validation/group_bootstrap.py')},
       'additional_real_draws_permitted':False,'conditional_fixed_OOF_not_selection_adjusted':True},immutable=True)
    progress('Fixed diagnostic boundaries and paired counts recorded','Unchanged historical H-regimes, no old per-series tables, 2000 paired draws per locked ResearchPool fold')


if __name__=='__main__':main()

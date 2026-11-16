"""Frozen-catalog uncertainty and paired comparisons with matched B2/D2."""
from pathlib import Path
import argparse
import csv
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import set_num_threads
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research,admitted_npz
from src.next5.references import component
from src.next5.bootstrap import PreparedAUC,paired_summary
from src.next5.selection_uncertainty import summarize_catalog
from src.next5.scheduler import register_job
from src.scoring.ts_auc import ts_auc


def table(path,rows):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    root=BASE/'selection_uncertainty';policy_path=root/'POLICY_LOCK.json'
    if args.register:
        candidates=sorted(p.parent.parent.name for p in (BASE/'experiments').glob('*/full/RESULTS.json'))
        atomic_json(policy_path,{'created_utc':utc_now(),'status':'FIXED_CATALOG_DIAGNOSTIC_BEFORE_ANALYSIS','candidates':candidates,
            'source_sha256':{name:digest(ROOT/name) for name in ('scripts/analyze_next5_selection_uncertainty.py','src/next5/selection_uncertainty.py','src/next5/bootstrap.py','src/next/bootstrap.py')},
            'existing_counts_lock_sha256':digest(BASE/'bootstrap/COUNTS_LOCK.json'),'new_random_draws':False,'new_fits':0,
            'warning':'The catalog was adaptively constructed. This descriptive fixed-OOF envelope is not a correction for all historical or within-session selection.',
            'no_primary_gate_change':True},immutable=True)
        register_job({'id':'UNC000_CONTRACTS','kind':'validation','priority':35,'argv':['-m','pytest','-q','tests/test_next5_selection_uncertainty.py',
            '--junitxml=artifacts/next5/selection_uncertainty/CONTRACTS.xml'],'description':'Tie-credit and empirical catalog uncertainty contract','retry_safe':False})
        register_job({'id':'UNC001_MATCHED_CATALOG','kind':'validation','priority':79,
            'depends_on_success':['UNC000_CONTRACTS']+['BOOT_'+cid for cid in candidates],
            'argv':['scripts/analyze_next5_selection_uncertainty.py'],'description':'Same 2000 draws for B2/D2 reference comparisons, winner gap and fixed catalog uncertainty','retry_safe':False})
        return
    policy=read_json(policy_path)
    for name,sha in policy['source_sha256'].items():assert digest(ROOT/name)==sha
    atomic_json(root/'START.json',{'created_utc':utc_now(),'new_fits':0,'new_random_draws':False,'policy_sha256':digest(policy_path)},immutable=True)
    set_num_threads(2);data=prepare_research();metadata=data.rows();predictions={};reference_sources=[]
    for kind in ('J102','C205','B2_augmented_J','D2_expert'):
        predictions[kind],proof=component(kind,'full','PRIMARY',metadata,data.guard);reference_sources.append(proof)
    bl=.8*predictions['J102']+.2*predictions['C205']
    composites={'B2':.8*predictions['B2_augmented_J']+.2*predictions['C205'],'D2':.9*bl+.1*predictions['D2_expert']}
    refs={'BL80':np.column_stack([np.load(BASE/'bootstrap/ResearchPool_BL80'/('fold_'+str(f)+'.npy'),allow_pickle=False) for f in range(5)])}
    counts_lock=read_json(BASE/'bootstrap/COUNTS_LOCK.json')
    for name,prediction in composites.items():
        folds=[]
        for fold in range(5):
            rows=np.flatnonzero(metadata['fold']==fold);record=counts_lock['folds'][fold];path=ROOT/record['path'];assert digest(path)==record['sha256']
            source=admitted_npz(path,data.guard,'Original paired counts for matched '+name+' reference')
            code=np.searchsorted(source['dataset_id'],metadata['dataset_id'][rows]).astype(np.int32)
            np.testing.assert_array_equal(source['dataset_id'][code],metadata['dataset_id'][rows])
            progress('Matched reference uncertainty',name+' ResearchPool fold '+str(fold)+'; unchanged original counts')
            draw=PreparedAUC.create(metadata['target'][rows],prediction[rows],metadata['time_online'][rows],code,source['counts'].shape[1]).evaluate(source['counts'])
            assert abs(draw[0]-ts_auc(metadata['target'][rows],prediction[rows],metadata['time_online'][rows]))<1e-14
            folds.append(draw);print('NEXT5 MATCHED REFERENCE UNCERTAINTY',name,fold,flush=True)
        refs[name]=np.asarray(folds).T
    np.savez_compressed(root/'MATCHED_REFERENCE_DRAWS.npz',**refs)
    deltas=[];families=[];candidate_matrices={};comparisons=[];sources=[]
    for cid in policy['candidates']:
        directory=BASE/'bootstrap'/cid;receipt=read_json(directory/'RESULTS.json');path=directory/'AGGREGATE_DRAWS.npz'
        assert digest(path)==receipt['files_sha256'][path.name]
        with np.load(path,allow_pickle=False) as arrays:
            new=arrays['candidate_fold_scores'];np.testing.assert_array_equal(arrays['reference_fold_scores'],refs['BL80'])
        candidate_matrices[cid]=new;deltas.append(new.mean(axis=1)-refs['BL80'].mean(axis=1));families.append(receipt['family'])
        sources.append({'candidate':cid,'bootstrap_result_sha256':digest(directory/'RESULTS.json'),'draws_sha256':digest(path)})
        for name,old in refs.items():comparisons.append({'candidate':cid,'family':receipt['family'],'reference':name,**paired_summary(old,new)})
    values=np.asarray(deltas).T;catalog=summarize_catalog(values,policy['candidates'],families)
    leave_one=[]
    for omitted in range(5):
        scores={cid:float(np.delete(matrix-refs['BL80'],omitted,axis=1)[0].mean()) for cid,matrix in candidate_matrices.items()}
        order=sorted(scores,key=lambda cid:(-scores[cid],cid));leave_one.append({'omitted_fold':omitted,'winner':order[0],
            'winner_delta':scores[order[0]],'runner_up':order[1],'winner_gap':scores[order[0]]-scores[order[1]],'refits':0})
    table(root/'CATALOG_UNCERTAINTY.csv',catalog.pop('candidates'));table(root/'FAMILY_MULTIPLICITY.csv',catalog.pop('families'))
    table(root/'MATCHED_REFERENCE_COMPARISONS.csv',comparisons);table(root/'OMIT_ONE_FOLD_DIAGNOSTIC.csv',leave_one)
    atomic_json(root/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS',**catalog,'candidate_count':len(policy['candidates']),
        'family_count':len(set(families)),'winner_gap_below_frozen_0_0007':catalog['winner_gap']<.0007,
        'new_fits':0,'new_random_draws':False,'reference_sources':reference_sources,'candidate_sources':sources,
        'shadow_rows':0,'seal_rows':0,'files_sha256':{p.name:digest(p) for p in root.glob('*.csv')},
        'matched_reference_draws_sha256':digest(root/'MATCHED_REFERENCE_DRAWS.npz')},immutable=True)
    data.guard.check_state();progress('Fixed catalog uncertainty complete','Winner gap, reference comparisons and family multiplicity; no selection-adjusted improvement claim')


if __name__=='__main__':main()

"""Executable checkpoint audit of frozen bytes and actual nested/OOF coordinates."""
from pathlib import Path
import sys
import argparse
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.nested import nested_partitions,array_hash
from src.next.data import prepare_dev
from src.scoring.ts_auc import ts_auc


def validate_fold_record(summary,receipt):
    # The two heuristic workers add PASS only to the standalone receipt.
    # Every scientific field must still match exactly; no generic subset rule.
    if summary.get('status','PASS')!='PASS' or receipt.get('status')!='PASS' or receipt!={**summary,'status':'PASS'}:
        raise ValueError('Fold summary and standalone receipt differ beyond the optional PASS field')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    if not args.label.isidentifier():raise ValueError('Simple audit label required')
    out=BASE/'maintenance'/('lineage_'+args.label)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'script_sha256':digest(Path(__file__)),
        'scope':'Checkpoint audit; not the final independent audit phase','new_fits':0},immutable=True)
    frozen=read_json(BASE/'preflight/FROZEN_REFERENCES.json')
    protected=frozen['protected_sha256'];count=0;failures=[]
    for name,expected in protected.items():
        path=ROOT/name
        if name=='EXPERIMENT_LEDGER.csv':
            original=(BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv').read_bytes()
            actual=path.read_bytes()
            same=actual.startswith(original)
        else:same=path.is_file() and digest(path)==expected
        if not same:failures.append(name)
        count+=1
        if count%2000==0:
            progress('Frozen provenance checkpoint',f'{count} protected old files checked; zero allowed overwrite',experiment='LINEAGE_AUDIT')
            print('NEXT3 FROZEN FILES CHECKED',count,flush=True)
    if failures:
        atomic_json(out/'FROZEN_FAILURE.json',{'status':'FAIL','changed_or_missing':failures},immutable=True)
        raise RuntimeError('Protected bytes changed')
    data=prepare_dev(ROOT);metadata=data.rows();nested_checked=[];fold_checked=[]
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    for scope in ('screen','full'):
        folder=BASE/'nested'/scope
        if not (folder/'RESULTS.json').exists():continue
        global_rows=np.flatnonzero(np.isin(metadata['dataset_id'],policy['screen']['ids'])) if scope=='screen' else np.arange(len(metadata['target']))
        local={k:v[global_rows] for k,v in metadata.items()}
        for outer in (policy['screen']['folds'] if scope=='screen' else range(5)):
            directory=folder/f'outer_{outer}';receipt=read_json(directory/'RESULTS.json')
            assert receipt['status']=='PASS' and receipt['outer_validation_never_used_for_inner_training']
            for name,expected in receipt['files_sha256'].items():assert digest(directory/name)==expected
            train=np.flatnonzero(local['fold']!=outer);valid=np.flatnonzero(local['fold']==outer)
            np.testing.assert_array_equal(np.load(directory/'TRAIN_GLOBAL_ROWS.npy',allow_pickle=False),global_rows[train])
            np.testing.assert_array_equal(np.load(directory/'VALID_GLOBAL_ROWS.npy',allow_pickle=False),global_rows[valid])
            partitions=nested_partitions(local,train,valid,data.guard)
            expected={inner:(a,b) for inner,a,b in partitions}
            seen=set()
            for lineage in receipt['lineage']:
                path=ROOT/lineage['receipt'];assert digest(path)==lineage['sha256']
                child=read_json(path);binding=child['binding'];inner=binding['inner_fold'];a,b=expected[inner]
                assert child['status']=='PASS' and binding==lineage['binding']
                assert binding['training_rows_sha256']==array_hash(global_rows[a])
                assert binding['validation_rows_sha256']==array_hash(global_rows[b])
                assert binding['outer_validation_rows_sha256']==array_hash(global_rows[valid])
                assert binding['training_target_sha256']==array_hash(local['target'][a])
                assert binding['training_series']==len(np.unique(local['dataset_id'][a]))
                assert binding['validation_series']==len(np.unique(local['dataset_id'][b]))
                assert binding['outer_validation_intersection']==binding['inner_train_validation_group_intersection']==0
                for name,sha in child['files_sha256'].items():assert digest(path.parent/name)==sha
                key=(inner,binding['role']);assert key not in seen;seen.add(key)
            assert len(seen)==2*len(partitions)
            nested_checked.append({'scope':scope,'outer_fold':outer,'base_fits':len(seen),'outer_validation_rows_excluded':len(valid),
                'outer_receipt_sha256':digest(directory/'RESULTS.json')})
            progress('Actual nested lineage reconstructed',f'{scope} outer {outer}: {len(seen)} base fits; train/validation/target hashes exact',experiment='LINEAGE_AUDIT')
    for path in sorted((BASE/'experiments').glob('*/*/RESULTS.json')):
        result=read_json(path)
        if result.get('status')!='COMPLETE' or result.get('scope') not in ('screen','full'):continue
        scope=result['scope'];global_rows=np.flatnonzero(np.isin(metadata['dataset_id'],policy['screen']['ids'])) if scope=='screen' else np.arange(len(metadata['target']))
        for record in result['fold_records']:
            fold=record['fold'];expected_rows=global_rows[metadata['fold'][global_rows]==fold]
            receipt=read_json(path.parent/f'fold_{fold}.json')
            validate_fold_record(record,receipt)
            for name,sha in receipt['files_sha256'].items():assert digest(path.parent/name)==sha
            rows=np.load(path.parent/f'fold_{fold}_global_rows.npy',allow_pickle=False)
            pred=np.load(path.parent/f'fold_{fold}.npy',allow_pickle=False)
            np.testing.assert_array_equal(rows,expected_rows)
            data.guard.admit(np.unique(metadata['dataset_id'][rows]),purpose='NEXT3 checkpoint OOF coordinate reconstruction')
            assert pred.dtype==np.float32 and len(pred)==len(rows) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
            score=ts_auc(metadata['target'][rows],pred,metadata['time_online'][rows])
            np.testing.assert_allclose(score,record['ts_auc'],rtol=0,atol=1e-14)
            fold_checked.append({'candidate':path.parent.parent.name,'scope':scope,'fold':fold,'rows':len(rows),
                'official_score_exact_to_1e14':True,'receipt_sha256':digest(path.parent/f'fold_{fold}.json')})
        progress('Completed OOF receipts verified',f'{path.parent.parent.name} {scope}: full coordinate, file and scorer match',experiment='LINEAGE_AUDIT')
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'protected_old_files_checked':count,
        'old_ledger_prefix_preserved':True,'nested_base_fits_reconstructed':sum(r['base_fits'] for r in nested_checked),
        'nested_scopes':nested_checked,'candidate_fold_receipts_checked':len(fold_checked),'fold_receipts':fold_checked,
        'seal_rows':0,'new_fits':0,'new_candidate_quality_selection':False,
        'limitations':'Reconstructs actual saved coordinate/target hashes and score bytes; this is executable single-agent auditing, not an independent external reviewer or selection-adjusted validation.'},immutable=True)
    print('NEXT3 LINEAGE CHECKPOINT PASS',count,len(fold_checked),flush=True)


if __name__=='__main__':main()

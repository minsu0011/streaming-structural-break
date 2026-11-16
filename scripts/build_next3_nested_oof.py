"""Fit frozen base recipes within outer training and save reusable honest gate data."""
from pathlib import Path
import sys
import argparse
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next3.frozen import candidate,fold_artifact,J102,C205
from src.next3.router import gate_design
from src.next3.nested import nested_partitions,array_hash
from src.next.data import prepare_dev
from src.next2.io import design,fit,model_class


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['screen','full'],required=True)
    args=parser.parse_args()
    hypothesis_path=BASE/'hypotheses/N001_NESTED_BASE_OOF.json'
    hypothesis=read_json(hypothesis_path)
    for name,expected in hypothesis['sources_sha256'].items():assert digest(ROOT/'src/next3'/name)==expected
    assert read_json(BASE/'preflight/NEXT3_PREFLIGHT.json')['status']=='PASS'
    if args.scope=='full':
        eligible=[p for p in (BASE/'experiments').glob('R[23]_*/screen/RESULTS.json') if read_json(p).get('screen_pass')]
        if not eligible:raise RuntimeError('No learned router passed its screen')
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    data=prepare_dev(ROOT)
    jc,_=candidate(J102);cc,_=candidate(C205)
    jx,metadata,jnames,jsources,global_rows=design(data,jc,args.scope)
    cx,cm,cnames,csources,crows=design(data,cc,args.scope)
    np.testing.assert_array_equal(global_rows,crows)
    for key in metadata:np.testing.assert_array_equal(metadata[key],cm[key])
    scope_dir=BASE/'nested'/args.scope
    scope_dir.mkdir(parents=True,exist_ok=True)
    identity={'scope':args.scope,'hypothesis_sha256':digest(hypothesis_path),
        'script_sha256':digest(Path(__file__)),'primary_feature_sources':jsources,'complement_feature_sources':csources,
        'rows_sha256':array_hash(global_rows),'target_sha256':array_hash(metadata['target'])}
    if (scope_dir/'POLICY.json').exists():assert read_json(scope_dir/'POLICY.json')==identity
    else:atomic_json(scope_dir/'POLICY.json',identity,immutable=True)
    outer_folds=policy['screen']['folds'] if args.scope=='screen' else list(range(5))
    fits_this_run=0
    for outer in outer_folds:
        out=scope_dir/f'outer_{outer}'
        out.mkdir(parents=True,exist_ok=True)
        if (out/'RESULTS.json').exists():
            old=read_json(out/'RESULTS.json')
            assert old['status']=='PASS'
            for name,expected in old['files_sha256'].items():assert digest(out/name)==expected
            print('NEXT3 NESTED OUTER REUSED',args.scope,outer,flush=True)
            continue
        train=np.flatnonzero(metadata['fold']!=outer);valid=np.flatnonzero(metadata['fold']==outer)
        partitions=nested_partitions(metadata,train,valid,data.guard)
        jp=np.full(len(global_rows),np.nan,dtype=np.float32);cp=jp.copy()
        lineage=[]
        for inner,inner_train,inner_valid in partitions:
            for role,spec,matrix,names,prediction in [('J102',jc,jx,jnames,jp),('C205',cc,cx,cnames,cp)]:
                piece=out/f'inner_{inner}_{role}'
                piece.mkdir(parents=True,exist_ok=True)
                receipt_path=piece/'RESULTS.json'
                binding={'outer_fold':outer,'inner_fold':inner,'role':role,
                    'training_rows_sha256':array_hash(global_rows[inner_train]),
                    'validation_rows_sha256':array_hash(global_rows[inner_valid]),
                    'outer_validation_rows_sha256':array_hash(global_rows[valid]),
                    'training_target_sha256':array_hash(metadata['target'][inner_train]),
                    'training_series':len(np.unique(metadata['dataset_id'][inner_train])),
                    'validation_series':len(np.unique(metadata['dataset_id'][inner_valid])),
                    'outer_validation_intersection':0,'inner_train_validation_group_intersection':0}
                if receipt_path.exists():
                    receipt=read_json(receipt_path)
                    assert receipt['status']=='PASS' and receipt['binding']==binding
                    for name,expected in receipt['files_sha256'].items():assert digest(piece/name)==expected
                    values=np.load(piece/'prediction.npy',allow_pickle=False)
                else:
                    attempts=list(piece.glob('attempt_*'))
                    attempt=piece/f'attempt_{len(attempts)+1}'
                    attempt.mkdir(exist_ok=False)
                    atomic_json(attempt/'START.json',{'created_utc':utc_now(),'binding':binding,'intended_new_learner_fits':1},immutable=True)
                    tick=time.monotonic()
                    progress('Nested base fitting',f'{args.scope} outer{outer} inner{inner} {role}; outer validation excluded',experiment='N001_NESTED_BASE_OOF')
                    model,values,detail=fit(data,spec,matrix,metadata,names,inner_train,inner_valid)
                    fits_this_run+=1
                    model.save(attempt/'model.joblib')
                    loaded=model_class(spec).load(attempt/'model.joblib')
                    np.testing.assert_array_equal(np.asarray(loaded.predict(matrix[inner_valid[:1024]]),dtype=np.float32),values[:1024])
                    np.save(piece/'prediction.npy',values,allow_pickle=False)
                    np.save(piece/'validation_global_rows.npy',global_rows[inner_valid],allow_pickle=False)
                    files=['prediction.npy','validation_global_rows.npy',str((attempt/'model.joblib').relative_to(piece)).replace('\\','/')]
                    receipt={'status':'PASS','created_utc':utc_now(),'binding':binding,'new_learner_fits':1,
                        'elapsed_seconds':time.monotonic()-tick,'training_detail':detail,'attempt':attempt.name,
                        'files_sha256':{name:digest(piece/name) for name in files}}
                    atomic_json(attempt/'COMPLETE.json',receipt,immutable=True)
                    atomic_json(receipt_path,receipt,immutable=True)
                np.testing.assert_array_equal(np.load(piece/'validation_global_rows.npy',allow_pickle=False),global_rows[inner_valid])
                assert values.shape==(len(inner_valid),) and np.isfinite(values).all()
                prediction[inner_valid]=values
                lineage.append({'receipt':receipt_path.relative_to(ROOT).as_posix(),'sha256':digest(receipt_path),'binding':binding})
                print('NEXT3 INNER BASE',args.scope,outer,inner,role,flush=True)
        assert np.isfinite(jp[train]).all() and np.isfinite(cp[train]).all()
        assert np.isnan(jp[valid]).all() and np.isnan(cp[valid]).all()
        gate_train=gate_design(jp[train],cp[train],jx[train],cx[train],jnames,cnames)
        _,jvalid,jrows,jproof=fold_artifact(data,J102,outer,args.scope)
        _,cvalid,cvalidrows,cproof=fold_artifact(data,C205,outer,args.scope)
        np.testing.assert_array_equal(global_rows[valid],jrows)
        np.testing.assert_array_equal(jrows,cvalidrows)
        gate_valid=gate_design(jvalid,cvalid,jx[valid],cx[valid],jnames,cnames)
        values={'GATE_TRAIN.npy':gate_train,'GATE_VALID.npy':gate_valid,'TRAIN_LOCAL_ROWS.npy':train,
            'VALID_LOCAL_ROWS.npy':valid,'TRAIN_GLOBAL_ROWS.npy':global_rows[train],'VALID_GLOBAL_ROWS.npy':global_rows[valid]}
        for name,array in values.items():np.save(out/name,array,allow_pickle=False)
        atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'outer_fold':outer,
            'scope_policy_sha256':digest(scope_dir/'POLICY.json'),'lineage':lineage,
            'inner_partitions':len(partitions),'base_learner_fits':2*len(partitions),
            'outer_validation_never_used_for_inner_training':True,'outer_validation_predictions_absent_from_inner_cache':True,
            'guarded_training_groups_disjoint':True,'seal_rows':0,
            'files_sha256':{name:digest(out/name) for name in values}},immutable=True)
        progress('Nested base OOF complete',f'{args.scope} outer fold {outer}: {len(partitions)} inner partitions; exact train coverage')
    atomic_json(scope_dir/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'scope':args.scope,
        'outer_folds':outer_folds,'new_learner_fits_this_invocation':fits_this_run,
        'files_sha256':{f'outer_{f}/RESULTS.json':digest(scope_dir/f'outer_{f}/RESULTS.json') for f in outer_folds}},immutable=True)
    print('NEXT3 NESTED BASE OOF PASS',args.scope,fits_this_run,'new fits',flush=True)


if __name__=='__main__':main()

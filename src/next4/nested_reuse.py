"""Strict read-only reuse of NEXT3 honest inner OOF with original coordinates."""
from pathlib import Path
import numpy as np
from src.next4.artifacts import ROOT,BASE,read_json,digest
from src.next4.frozen import fold_predictions
from src.next3.nested import nested_partitions,array_hash
from src.next3.frozen import candidate as frozen_candidate,J102,C205


def load_outer(data,scope,outer):
    if scope not in ('screen','full') or type(outer) is not int or outer not in range(5):
        raise ValueError('Locked scope and original outer fold required')
    current_policy=read_json(ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json')
    previous_policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    assert current_policy['screen']['ids']==previous_policy['screen']['ids']
    assert current_policy['screen']['folds']==previous_policy['screen']['folds']
    hyp_path=ROOT/'artifacts/next3/hypotheses/N001_NESTED_BASE_OOF.json';hyp=read_json(hyp_path)
    for name,sha in hyp['sources_sha256'].items():
        if digest(ROOT/'src/next3'/name)!=sha:raise RuntimeError('Original nested source identity changed: '+name)
    directory=ROOT/'artifacts/next3/nested'/scope;policy=read_json(directory/'POLICY.json')
    assert policy['scope']==scope and policy['hypothesis_sha256']==digest(hyp_path)
    assert policy['script_sha256']==digest(ROOT/'scripts/build_next3_nested_oof.py')
    # Validate current registered feature implementations against the recipes
    # that produced the original nested cache, without extracting or fitting.
    primary,_=frozen_candidate(J102);complement,_=frozen_candidate(C205)
    assert policy['primary_feature_sources']['extension_hash']==primary['extension_frozen_hash']
    assert policy['complement_feature_sources']['extension_hash']==complement['extension_frozen_hash']
    allmeta=data.rows();global_rows=np.arange(len(allmeta['target']),dtype=np.int64)
    if scope=='screen':global_rows=global_rows[np.isin(allmeta['dataset_id'],current_policy['screen']['ids'])]
    meta={key:value[global_rows] for key,value in allmeta.items()}
    assert policy['rows_sha256']==array_hash(global_rows) and policy['target_sha256']==array_hash(meta['target'])
    out=directory/f'outer_{outer}';receipt=read_json(out/'RESULTS.json')
    assert receipt['status']=='PASS' and receipt['outer_fold']==outer and receipt['scope_policy_sha256']==digest(directory/'POLICY.json')
    assert receipt['outer_validation_never_used_for_inner_training'] and receipt['outer_validation_predictions_absent_from_inner_cache']
    for filename,sha in receipt['files_sha256'].items():
        if digest(out/filename)!=sha:raise RuntimeError('Nested outer cache bytes changed: '+filename)
    train=np.flatnonzero(meta['fold']!=outer);valid=np.flatnonzero(meta['fold']==outer)
    for filename,expected in [('TRAIN_LOCAL_ROWS.npy',train),('VALID_LOCAL_ROWS.npy',valid),
        ('TRAIN_GLOBAL_ROWS.npy',global_rows[train]),('VALID_GLOBAL_ROWS.npy',global_rows[valid])]:
        np.testing.assert_array_equal(np.load(out/filename,allow_pickle=False),expected)
    nested=nested_partitions(meta,train,valid,data.guard)
    gate_train=np.load(out/'GATE_TRAIN.npy',mmap_mode='r',allow_pickle=False)
    gate_valid=np.load(out/'GATE_VALID.npy',mmap_mode='r',allow_pickle=False)
    assert gate_train.shape==(len(train),6) and gate_valid.shape==(len(valid),6)
    lineage={(r['binding']['inner_fold'],r['binding']['role']):r for r in receipt['lineage']}
    sources={str((directory/'POLICY.json').relative_to(ROOT)):digest(directory/'POLICY.json'),
        str((out/'RESULTS.json').relative_to(ROOT)):digest(out/'RESULTS.json')}
    for inner,inner_train,inner_valid in nested:
        for column,role in enumerate(('J102','C205')):
            link=lineage[(inner,role)];path=ROOT/link['receipt'];assert digest(path)==link['sha256']
            record=read_json(path);binding=record['binding'];assert binding==link['binding'] and record['status']=='PASS'
            assert binding['training_rows_sha256']==array_hash(global_rows[inner_train])
            assert binding['validation_rows_sha256']==array_hash(global_rows[inner_valid])
            assert binding['outer_validation_rows_sha256']==array_hash(global_rows[valid])
            assert binding['training_target_sha256']==array_hash(meta['target'][inner_train])
            assert binding['outer_validation_intersection']==0 and binding['inner_train_validation_group_intersection']==0
            for filename,sha in record['files_sha256'].items():
                if digest(path.parent/filename)!=sha:raise RuntimeError('Original nested fit/prediction bytes changed: '+filename)
            np.testing.assert_array_equal(np.load(path.parent/'validation_global_rows.npy',allow_pickle=False),global_rows[inner_valid])
            positions=np.searchsorted(train,inner_valid)
            np.testing.assert_array_equal(gate_train[positions,column],np.load(path.parent/'prediction.npy',allow_pickle=False))
            sources[link['receipt']]=link['sha256']
    frozen,frozen_sources=fold_predictions(data,outer,scope,global_rows[valid])
    np.testing.assert_array_equal(gate_valid[:,0],frozen['J102'])
    np.testing.assert_array_equal(gate_valid[:,1],frozen['C205'])
    data.guard.check_state()
    proof={'scope':scope,'outer_fold':outer,'source_receipts_sha256':sources,'frozen_outer_prediction_sources':frozen_sources,
        'inner_base_fits_reused':2*len(nested),'new_base_fits':0,'outer_validation_in_meta_training':False,
        'training_global_rows_sha256':array_hash(global_rows[train]),'validation_global_rows_sha256':array_hash(global_rows[valid]),
        'training_target_sha256':array_hash(meta['target'][train]),'seal_rows':0,
        'inner_OOF_component_columns_bitwise_verified':True,'outer_component_columns_bitwise_verified':True}
    return np.asarray(gate_train[:,:2]),np.asarray(gate_valid[:,:2]),meta,train,valid,global_rows,proof


def source_hash():
    paths=['src/next4/nested_reuse.py','src/next3/nested.py','src/next3/router.py','src/next3/frozen.py',
        'src/next/candidate_io.py','src/next2/io.py','src/next4/frozen.py']
    return {p:digest(ROOT/p) for p in paths}

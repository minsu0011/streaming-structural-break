"""Unchanged recipes under frozen alternative folds and H-regime exclusion."""
import hashlib
import numpy as np
from src.next5.artifacts import ROOT,BASE,read_json,digest
from src.next5.data import admitted_npz


def heldout_rows(row_ids,assignments,fold):
    ids=np.asarray(row_ids)
    unique,inverse=np.unique(ids,return_inverse=True)
    if any(int(i) not in assignments for i in unique):raise ValueError('Every row must have a frozen group assignment')
    values=np.asarray([assignments[int(i)] for i in unique],dtype=np.int8)[inverse]
    train=np.flatnonzero(values!=fold);valid=np.flatnonzero(values==fold)
    if not len(train) or not len(valid):raise ValueError('Nonempty training and held-out partitions required')
    if set(ids[train])&set(ids[valid]):raise ValueError('Series crosses training and held-out boundary')
    return train,valid


def assignments(split,guard):
    if split in ('ALT_A','ALT_B'):
        values={int(i):int(f) for i,f in read_json(ROOT/'NEXT5_SPLIT_LOCK.json')['mappings'][split].items()}
    elif split=='H_EXCLUSION':
        lock=read_json(BASE/'diagnostics/RESEARCH_DIAGNOSTIC_LOCK.json');path=BASE/'diagnostics/RESEARCH_H_DESCRIPTORS.npz'
        assert digest(path)==lock['files_sha256'][path.name]
        record=admitted_npz(path,guard,'ResearchPool H-only held-out regime assignments')
        values={int(i):int(r) for i,r in zip(record['dataset_id'],record['historical_regime'])}
    else:raise ValueError('Only frozen ALT_A, ALT_B or original H-regime exclusion is allowed')
    if set(values)!=guard.research_ids or set(values.values())!=set(range(5)):raise ValueError('Complete frozen ResearchPool five-way assignment required')
    group_folds={}
    for sid,group in guard.split.groups.items():
        if group in group_folds and group_folds[group]!=values[sid]:raise ValueError('Historical duplicate group crosses held-out assignment')
        group_folds[group]=values[sid]
    return values


def reference_root(kind,split):
    return BASE/'stability/references/H_EXCLUSION'/kind if split=='H_EXCLUSION' else BASE/'references/fits'/split/'full'/kind


def load_reference(kind,split,metadata,guard):
    root=reference_root(kind,split);receipt=read_json(root/'RESULTS.json');path=root/'OOF.npz'
    if receipt['status']!='COMPLETE' or digest(path)!=receipt['OOF_sha256']:raise RuntimeError('Complete immutable held-out reference OOF required')
    arrays=admitted_npz(path,guard,'Matched held-out stability reference')
    rows=arrays['global_row']
    if len(rows)!=len(metadata['dataset_id']) or not np.array_equal(rows,np.arange(len(rows))):raise ValueError('Complete ordered held-out reference rows required')
    np.testing.assert_array_equal(arrays['dataset_id'],metadata['dataset_id'])
    return arrays['prediction'],{'reference':kind,'split':split,'result_sha256':digest(root/'RESULTS.json'),'OOF_sha256':receipt['OOF_sha256']}


def frozen_primary_counts(guard):
    """Reuse the original PRIMARY-stratified real-data draws; no new RNG call."""
    lock=read_json(BASE/'bootstrap/COUNTS_LOCK.json');blocks=[];ids=[];proof=[]
    for row in lock['folds']:
        path=ROOT/row['path'];assert digest(path)==row['sha256']
        data=admitted_npz(path,guard,'Reuse original whole-series resampling multiplicities')
        blocks.append(data['counts']);ids.append(data['dataset_id']);proof.append(row['sha256'])
    all_ids=np.concatenate(ids);order=np.argsort(all_ids,kind='stable')
    if len(np.unique(all_ids))!=len(all_ids) or set(map(int,all_ids))!=guard.research_ids:raise ValueError('Complete unique original resampling membership required')
    return all_ids[order],np.concatenate(blocks,axis=1)[:,order],proof


def training_hash(metadata,train):
    return hashlib.sha256(np.sort(np.unique(metadata['dataset_id'][train])).tobytes()).hexdigest()

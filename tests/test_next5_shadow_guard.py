"""Prospective boundaries fail before real label or observation access."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
from src.next5.guard import ResearchGuard,ShadowAccessError
from src.next5.artifacts import ROOT,read_json
from src.next5.data import admitted_npz


@pytest.fixture(scope='module')
def guard():return ResearchGuard(ROOT)


def test_complete_disjoint_membership_and_group_integrity(guard):
    assert len(guard.research_ids)==6400 and len(guard.shadow_ids)==1600 and len(guard.seal_ids)==2000
    assert guard.research_ids|guard.shadow_ids==guard.base.dev_ids
    assert not guard.research_ids&guard.shadow_ids and not guard.base.dev_ids&guard.seal_ids
    assert not {guard.base.split.groups[i] for i in guard.research_ids}&{guard.base.split.groups[i] for i in guard.shadow_ids}


@pytest.mark.parametrize('scope',['training','validation','feature cache','OOF','candidate selection','bootstrap selection','labels'])
@pytest.mark.parametrize('boundary',['shadow','seal'])
def test_all_research_admission_paths_reject_reserved_ids(guard,scope,boundary):
    ids=guard.shadow_ids if boundary=='shadow' else guard.seal_ids
    with pytest.raises(ShadowAccessError):guard.admit([min(ids)],purpose=scope)


@pytest.mark.parametrize('invalid',[True,1.0,'1',-999999])
def test_noncanonical_or_unknown_identifiers_fail(guard,invalid):
    with pytest.raises(ShadowAccessError):guard.admit([invalid])


def test_default_is_research_only_and_integers_are_supported(guard):
    assert set(guard.admit())==guard.research_ids
    assert guard.admit(np.asarray([min(guard.research_ids)],dtype=np.int32))==[min(guard.research_ids)]


def test_shadow_rejected_before_underlying_raw_scanner(monkeypatch,guard):
    def forbidden(*args,**kwargs):raise AssertionError('Raw scanner called before prospective admission')
    monkeypatch.setattr(guard.base,'scanner',forbidden)
    with pytest.raises(ShadowAccessError):guard.scanner('X_train.parquet',[min(guard.shadow_ids)])
    with pytest.raises(ShadowAccessError):guard.labels([min(guard.shadow_ids)])


def test_cache_labels_not_decoded_before_id_rejection(monkeypatch,guard):
    accessed=[]
    class FakeArchive:
        files=['dataset_id','tau','target']
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,name):
            accessed.append(name)
            if name=='dataset_id':return np.array([min(guard.shadow_ids)],dtype=np.int32)
            raise AssertionError('Forbidden cached labels were decoded')
    monkeypatch.setattr('src.next5.data.np.load',lambda *a,**k:FakeArchive())
    with pytest.raises(ShadowAccessError):admitted_npz(Path('mock.npz'),guard,'cache')
    assert accessed==['dataset_id']


def test_training_validation_overlap_rejected(guard):
    sid=min(guard.research_ids)
    with pytest.raises(ShadowAccessError):guard.partition([sid],[sid])


def test_source_ids_only_reproduce_shadow_lock(guard):
    lock=read_json(ROOT/'NEXT5_SHADOW_LOCK.json');salt=lock['salt']
    groups={guard.base.split.groups[i]:i for i in guard.base.dev_ids}
    assert len(groups)==8000
    ordered=sorted(groups,key=lambda gid:(hashlib.sha256((salt+':shadow:'+str(gid)).encode('ascii')).hexdigest(),gid))
    assert set(ordered[:1600])==set(guard.shadow_ids)


def test_folds_and_screen_stay_inside_research(guard):
    lock=read_json(ROOT/'NEXT5_SPLIT_LOCK.json')
    for partition,mapping in lock['mappings'].items():
        assert set(map(int,mapping))==guard.research_ids
        assert [sum(v==f for v in mapping.values()) for f in range(5)]==[1280]*5
    assert set(lock['screen_ids'])<=guard.research_ids and len(lock['screen_ids'])==2560
    assert lock['screen_folds']==[0,4]


def test_legacy_full_dev_loader_not_imported_in_next5_data():
    import ast
    module=ast.parse((ROOT/'src/next5/data.py').read_text(encoding='utf-8'))
    for node in ast.walk(module):
        if isinstance(node,ast.ImportFrom) and node.module=='src.next.data':
            assert not {'prepare_dev','DevArrays'}&{n.name for n in node.names}

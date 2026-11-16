"""Held-out partition purity and unchanged original resampling coordinates."""
import numpy as np
import pytest
from src.next5.stability import heldout_rows,assignments,frozen_primary_counts
from src.next5.guard import ResearchGuard
from src.next5.bootstrap import PreparedAUC
from src.scoring.ts_auc import ts_auc


def test_every_timeline_stays_together_and_each_series_is_held_out_once():
    ids=np.array([11,11,18,18,18,23,41,41,55]);mapping={11:0,18:1,23:2,41:3,55:4};seen=[]
    for fold in range(5):
        train,valid=heldout_rows(ids,mapping,fold)
        assert not set(ids[train])&set(ids[valid])
        assert np.array_equal(np.sort(np.r_[train,valid]),np.arange(len(ids)))
        seen.extend(valid.tolist())
    assert sorted(seen)==list(range(len(ids)))
    with pytest.raises(ValueError):heldout_rows(ids,{11:0},0)


@pytest.mark.parametrize('split',['ALT_A','ALT_B','H_EXCLUSION'])
def test_actual_frozen_assignments_cover_only_research_and_disjoint_groups(split):
    guard=ResearchGuard();mapping=assignments(split,guard);ids=np.array(sorted(mapping))
    assert not set(mapping)&guard.shadow_ids and not set(mapping)&guard.seal_ids
    for fold in range(5):
        train,valid=heldout_rows(ids,mapping,fold);guard.partition(ids[train],ids[valid])


def test_fixed_primary_draws_are_reused_without_regeneration_or_restratification():
    guard=ResearchGuard();ids,counts,proof=frozen_primary_counts(guard)
    assert len(proof)==5 and counts.shape==(2001,6400)
    np.testing.assert_array_equal(ids,np.array(sorted(guard.research_ids)))
    np.testing.assert_array_equal(counts[0],np.ones(6400))
    np.testing.assert_array_equal(counts.sum(axis=1),np.full(2001,6400))
    # A held-out slice has variable total multiplicity. Preserve it instead of
    # redrawing/normalizing counts, and compare with literal timeline repeats.
    use=np.arange(7);slice_counts=counts[:4,use];target=np.tile([0,1,1],7);times=np.tile([0,1,2],7)
    target[::2]=0;prediction=np.arange(21,dtype=float)/21.;code=np.repeat(np.arange(7),3)
    actual=PreparedAUC.create(target,prediction,times,code,7).evaluate(slice_counts)
    for i,row in enumerate(slice_counts):
        indices=np.repeat(np.arange(len(target)),row[code])
        assert abs(actual[i]-ts_auc(target[indices],prediction[indices],times[indices]))<1e-14

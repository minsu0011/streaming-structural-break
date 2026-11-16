"""All-point target-free audit of stride-eight phase representativeness."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(part):
    denied=install_workspace_boundary()
    import numpy as np
    from src.next9.data import FeatureBank
    from src.next9.observability import finite_window_states
    from src.next9.sampling_phase import phase_counts,eligible_group_frequencies,FAMILIES
    from src.next9.io import save_npz
    root=BASE/'sampling_phase'/f'part_{part}'
    if (root/'DONE.json').exists():raise FileExistsError('Completed phase analysis cannot be rerun')
    sums=np.zeros((5,8,9,153,26));groups=np.zeros((5,8),dtype=np.int64)
    excluded=np.zeros_like(groups);point_totals=np.zeros((5,8,9),dtype=np.int64)
    ids=[];points=0
    with target_free_npz() as accesses:
        bank=FeatureBank();selected=np.arange(len(bank.ids))[part::4]
        for offset in range(0,len(selected),16):
            batch=selected[offset:offset+16]
            with activity(f'All153 stride8 sampling-phase audit part{part} series batch{offset//16}',
                          'SAMPLING_PHASE_SUPPORT','B9_NORMALIZATION_STATE_AVAILABILITY'):
                for s in batch:
                    sid=int(bank.ids[s]);assert bank.guard.split.groups[sid]==sid
                    fold=int(bank.folds[s]);lo,hi=map(int,bank.offsets[s:s+2])
                    values=np.asarray(bank.values[lo:hi]);seed=bank.series['seed_values'][s]
                    times,states=finite_window_states(seed,values,bank.series['epsilon'][s,0],
                                                    bank.series['H_length'][s]>=160,stride=1)
                    np.testing.assert_array_equal(times,bank.times[lo:hi])
                    encoded=np.stack([states[k] for k in FAMILIES],axis=-1)
                    counts,totals=phase_counts(encoded,times)
                    frequencies,eligible=eligible_group_frequencies(counts,totals)
                    sums[fold]+=frequencies;groups[fold]+=eligible
                    excluded[fold]+=(totals[:,8]>0)&~eligible
                    point_totals[fold]+=totals*eligible[:,None]
                    ids.append(sid);points+=len(times)
        save_npz(root/'AGGREGATES.npz',equal_group_frequency_sum=sums,complete_phase_groups=groups,
                 excluded_incomplete_phase_groups=excluded,point_totals=point_totals)
    assert not denied
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),part=part,series_ids=ids,points=points,
        features=153,phases=list(range(8))+['ALL_POINTS'],domains=['ALL']+[f'TIME_{i}' for i in range(7)],
        all_points_not_stride8=True,only_complete_phase_series_compared=True,no_labels_fits_selection_or_threshold_changes=True,
        interpretation='Post hoc covariate sampling diagnosis, not proof of performance causation; no phase optimization.',
        aggregates_sha256=digest(root/'AGGREGATES.npz'),NPZ_members=sorted(set(accesses))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,required=True,choices=range(4));main(p.parse_args().part)

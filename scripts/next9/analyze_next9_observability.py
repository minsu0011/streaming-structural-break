"""All153 descriptive state support and finite-history observability audit."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(part):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OBSERVABILITY_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank
        from src.next9.domains import TIME_BINS,H_BINS
        from src.next9.observability import observed_prefix_reference,finite_window_states
        from src.next9.io import save_npz
        bank=FeatureBank();root=BASE/'observability'/f'part_{part}'
        if (root/'DONE.json').exists():return
        families={'S1_DIRECTION':4,'S2_DIRECTION_RUN':10,'S3_DIRECTION_SLOPE':7,'S4_REVERSAL':5}
        domains=['ALL']+[f'TIME_{k}' for k in range(7)]+[f'H_LENGTH_{k}' for k in range(5)]
        counts={k:np.zeros((5,13,153,n),dtype=np.int64) for k,n in families.items()}
        groups={k:np.zeros_like(v) for k,v in counts.items()}
        totals=np.zeros((5,13),dtype=np.int64);series=np.zeros((5,13),dtype=np.int64)
        event_count=np.zeros((5,13,153,4),dtype=np.int64);event_sum=np.zeros_like(event_count,dtype=float);valid_series=np.zeros((5,13),dtype=np.int64)
        selected=np.arange(len(bank.ids))[part::4];processed=[]
        for offset in range(0,len(selected),32):
            batch=selected[offset:offset+32]
            with activity(f'All153 observability part{part} originalseries batch{offset//32}','OBSERVABILITY_FORENSIC','B9_NORMALIZATION_STATE_AVAILABILITY'):
                for s in batch:
                    sid=int(bank.ids[s]);assert bank.guard.split.groups[sid]==sid
                    fold=int(bank.folds[s]);lo,hi=map(int,bank.offsets[s:s+2]);values=np.asarray(bank.values[lo:hi]);seed=bank.series['seed_values'][s]
                    eps=bank.series['epsilon'][s,0];available=bank.series['H_length'][s]>=160
                    full_run,full_age,full_s4=observed_prefix_reference(seed,values,eps)
                    times,states=finite_window_states(seed,values,eps,available)
                    if available:np.testing.assert_array_equal(states['direction_run_capped15'],np.minimum(full_run[times],15))
                    events=np.stack([full_run[times]>15,full_age[times]>15,
                        states['S4_REVERSAL']!=full_s4[times],states['reversal_age_capped15']!=np.minimum(full_age[times],15)],axis=-1)&available
                    masks=[np.ones(len(times),dtype=bool)]
                    masks.extend((times>=a)&(True if b is None else times<=b) for a,b in TIME_BINS)
                    h=int(bank.series['H_length'][s])
                    masks.extend(np.full(len(times),h>=a and (b is None or h<=b),dtype=bool) for a,b in H_BINS)
                    for d,keep in enumerate(masks):
                        n=int(keep.sum())
                        if not n:continue
                        totals[fold,d]+=n;series[fold,d]+=1;valid_series[fold,d]+=int(available)
                        event_count[fold,d]+=events[keep].sum(axis=0)
                        event_sum[fold,d]+=events[keep].mean(axis=0)
                        for family,width in families.items():
                            state=states[family][keep]
                            for code in range(width):
                                count=(state==code).sum(axis=0)
                                counts[family][fold,d,:,code]+=count;groups[family][fold,d,:,code]+=count>0
                    processed.append(sid)
        arrays={**{'count_'+k:v for k,v in counts.items()},**{'groups_'+k:v for k,v in groups.items()},
            'point_totals':totals,'series_totals':series,'valid_series':valid_series,'event_count':event_count,'event_equal_group_sum':event_sum}
        save_npz(root/'AGGREGATES.npz',**arrays)
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),part=part,series_ids=processed,features=153,
            domains=domains,state_families=families,events=['run_exceeds15','observed_prefix_reversal_age_exceeds15','S4_differs_across_long_flat_bridge','reversal_age_window_differs'],
            reference='Only same16Hseed and observedOnlineprefix. Initial no-reversal age15 is a lower bound, never true event age.',
            interpretation='Finite-window definitions remain frozen; longer-prefix diagnostic never substituted into training or M predictions. Long-FLAT bridge disagreement is an observability limit, not a causal violation.',
            raw_S2_not_TRAIN_sparse_fallback=True,break_labels_or_tau_used=False,NPZ_members=sorted(set(accesses)),
            aggregates_sha256=digest(root/'AGGREGATES.npz'),implementation_sha256=digest(BASE/'OBSERVABILITY_IMPLEMENTATION_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',required=True,type=int,choices=range(4));main(p.parse_args().part)

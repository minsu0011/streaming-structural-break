"""Concise all153 support and finite-window limitation tables."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,csv_write,read_json,digest,activity,utc_now


def main():
    import numpy as np
    with activity('Aggregate all153 originalRP state support and causal finite-window censoring','OBSERVABILITY_SUMMARY','B9_NORMALIZATION_STATE_AVAILABILITY'):
        totals={};ids=[];receipts=[]
        for part in range(4):
            directory=BASE/'observability'/f'part_{part}';r=read_json(directory/'DONE.json');assert r['status']=='COMPLETE'
            assert digest(directory/'AGGREGATES.npz')==r['aggregates_sha256'];ids.extend(r['series_ids']);receipts.append(r)
            with np.load(directory/'AGGREGATES.npz') as z:
                for name in z.files:
                    if name not in totals:totals[name]=z[name].copy()
                    else:totals[name]+=z[name]
        assert len(ids)==len(set(ids))==6400
        schema=receipts[0];support=[];limits=[]
        for fold in range(5):
            for d,domain in enumerate(schema['domains']):
                n=int(totals['point_totals'][fold,d]);groups=int(totals['series_totals'][fold,d]);valid=int(totals['valid_series'][fold,d])
                for j in range(153):
                    for family,width in schema['state_families'].items():
                        assert int(totals['count_'+family][fold,d,j].sum())==n
                        for code in range(width):
                            count=int(totals['count_'+family][fold,d,j,code]);g=int(totals['groups_'+family][fold,d,j,code])
                            support.append(dict(fold=fold,domain=domain,feature=j,state_family=family,state_code=code,
                                points=count,original_groups=g,domain_points=n,domain_groups=groups,
                                point_fraction=count/n if n else None,raw_S2_before_TRAIN_fallback=True,break_labels_used=False))
                    for e,event in enumerate(schema['events']):
                        count=int(totals['event_count'][fold,d,j,e])
                        limits.append(dict(fold=fold,domain=domain,feature=j,event=event,points=count,domain_points=n,
                            point_fraction=count/n if n else None,equal_group_fraction=float(totals['event_equal_group_sum'][fold,d,j,e]/valid) if valid else None,
                            groups_with_available_history=valid,definition_unchanged=True,full_reference_uses_observed_prefix_only=True))
        csv_write('MOVEMENT_STATE_SUPPORT.csv',support);csv_write('STATE_OBSERVABILITY_NEXT9.csv',limits)
        overall=[r for r in limits if r['domain']=='ALL']
        atomic(BASE/'STATE_OBSERVABILITY_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),original_series=6400,features=153,
            seed_reference=schema['reference'],interpretation=schema['interpretation'],events=schema['events'],
            state_support_rows=len(support),observability_rows=len(limits),
            event_points={event:sum(r['points'] for r in overall if r['event']==event) for event in schema['events']},
            unchanged_frozen_state_mapping=True,not_complete_Markov_state_claim=True,
            input_receipts_sha256={str(k):digest(BASE/'observability'/f'part_{k}'/'DONE.json') for k in range(4)}))


if __name__=='__main__':main()

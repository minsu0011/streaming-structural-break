"""Generate only fixed, same-engine matched synthetic counterexamples."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,activity,note
from src.next8.synthetic_controls import generate_block,BACKGROUNDS,H_LENGTHS,SERIES_PER_BACKGROUND


def main(part,parts):
    assert read_json(BASE/'SYNTHETIC_CONTROL_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock=read_json(BASE/'SYNTHETIC_CONTROL_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    total=len(BACKGROUNDS)*len(H_LENGTHS)*SERIES_PER_BACKGROUND
    records=[]
    for block in range(part,total,parts):
        p=BASE/'synthetic_controls'/f'block_{block:04d}'
        if (p/'DONE.json').exists():generate_block(block,p)
        else:
            with activity(f'Matched synthetic feature block {block}/{total}','SYNTHETIC_COUNTEREXAMPLE','B7_NORMALIZATION_AND_LINEAGE'):
                generate_block(block,p)
        records.append({'block':block,'done_sha256':digest(p/'DONE.json')})
    atomic(BASE/'synthetic_controls'/f'PART_{part}_DONE.json',{'status':'COMPLETE','timestamp':utc_now(),
        'part':part,'parts':parts,'records':records,'real_data_read':False,
        'independent_innovation_paths':len(BACKGROUNDS)*SERIES_PER_BACKGROUND,
        'H_and_intervention_variants_are_matched_not_independent':True})
    note('Synthetic matched feature controls complete','part'+str(part)+'; no model refits, no real data','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=(0,1),required=True);p.add_argument('--parts',type=int,choices=(2,),default=2)
    a=p.parse_args();main(a.part,a.parts)

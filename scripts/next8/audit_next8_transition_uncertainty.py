"""Attach actual table probability intervals to the state-transition figure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note
from src.next8.cache import FeatureCache
from src.next8.transition_uncertainty import actual_table_uncertainty


def main():
    lock=read_json(BASE/'TRANSITION_UNCERTAINTY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache()
    with activity('Frozen transition probability uncertainty','TRANSITION_UNCERTAINTY','B2_STATE_OBSERVABILITY'):
        result=actual_table_uncertainty(cache)
        atomic(BASE/'figure_data/TRANSITION_UNCERTAINTY.json',{'timestamp':utc_now(),**result})
    note('Complete TRAIN transition table probability intervals','Separate from heldout proper-loss uncertainty','B2_STATE_OBSERVABILITY')


if __name__=='__main__':main()

"""Select all four prescribed kinds of real heldout explanatory cases."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note
from src.next8.cache import FeatureCache
from src.next8.figure_cases import collect_real_cases


def main():
    lock=read_json(BASE/'FIGURE_CASE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache()
    with activity('Select fixed success/failure/miss/false-alarm illustrations','FIGURE_CASES','B4_MOVEMENT_VS_DISCRIMINATION'):
        result=collect_real_cases(cache)
        atomic(BASE/'figure_data/REAL_CASES.json',{'timestamp':utc_now(),**result})
        atomic(BASE/'figure_data/REAL_CASES_DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'cases_sha256':digest(BASE/'figure_data/REAL_CASES.json')})
    note('Complete fixed real movement illustrations','Every category kept, missing category explicit, no model feedback','B4_MOVEMENT_VS_DISCRIMINATION')


if __name__=='__main__':main()

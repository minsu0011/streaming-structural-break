"""Prepare data-only inputs for standard scientific plots in the figure venv."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,atomic,read_json,digest,csv_write,utc_now,activity,note
from src.next8.cache import FeatureCache
from src.next8.figure_data import same_time_curves,heldout_timeline,transition_illustration


def main():
    lock=read_json(BASE/'FIGURE_DATA_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();directory=BASE/'figure_data';directory.mkdir(parents=True,exist_ok=True)
    with activity('Prepare exact score curves and observed feature/state timeline','FIGURE_DATA','B4_MOVEMENT_VS_DISCRIMINATION'):
        curves,checks=same_time_curves(cache);csv_write('figure_data/SAME_TIME_CURVES.csv',curves)
        atomic(directory/'TIMELINE.json',heldout_timeline(cache))
        atomic(directory/'TRANSITION.json',transition_illustration())
        atomic(directory/'DATA_DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'score_curve_checks':checks,
            'files_sha256':{n:digest(directory/n) for n in ('SAME_TIME_CURVES.csv','TIMELINE.json','TRANSITION.json')},
            'curve_zero_pair_times_preserved_as_missing':True,'case_policy_sha256':digest(BASE/'FIGURE_CASE_POLICY_NEXT8.json'),
            'figures_are_observed_outputs_not_hypothetical_discoveries':True})
    note('Complete canonical plot data','Per-time ranking agrees with frozen score authority; no model changes','B4_MOVEMENT_VS_DISCRIMINATION')


if __name__=='__main__':main()

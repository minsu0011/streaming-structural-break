"""Aggregate fixed A/control and actual state-observability evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note,csv_write
from src.next8.movement_summary import summarize_movement,summarize_observability


def main():
    lock=read_json(BASE/'MOVEMENT_SUMMARY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'movement_summary';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Existing movement summary preserved')
    outputs={};files=[]
    for early in (False,True):
        name='EARLY' if early else 'NORMAL'
        with activity('Aggregate movement factors and controls '+name,'MOVEMENT_SUMMARY','B3_SELF_HISTORY_SUFFICIENCY'):
            result=summarize_movement(early);outputs[name]=result
            atomic(directory/(name+'.json'),{'timestamp':utc_now(),**result});files.append(name+'.json')
            state=summarize_observability(early)
            atomic(directory/(name+'_OBSERVABILITY.json'),{'timestamp':utc_now(),**state});files.append(name+'_OBSERVABILITY.json')
    for key,filename in [('own_dynamics','OWN_DYNAMICS.csv'),('cross_edges','CROSS_EDGES.csv'),('transition_tests','TRANSITIONS.csv'),
                         ('factor_ablations','FACTOR_ABLATIONS.csv'),('fold_metrics','FOLD_METRICS.csv')]:
        csv_write('movement_summary/'+filename,outputs['NORMAL'][key]+outputs['EARLY'][key]);files.append(filename)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'files_sha256':{n:digest(directory/n) for n in files},
        'normal_counts':outputs['NORMAL']['counts'],'early_counts':outputs['EARLY']['counts'],'A_B_C_separate':True})
    note('Complete cross-fold movement and state evidence','All horizons, mechanical and removal controls retained','B3_SELF_HISTORY_SUFFICIENCY')


if __name__=='__main__':main()

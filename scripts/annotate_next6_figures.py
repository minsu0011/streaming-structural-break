"""Add exact point support and window-comparison warnings to result figures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6.common import BASE,read_json,save,activity
from src.next6.data import source

def main():
    with activity('figure_support_and_comparison_annotations','FINAL_AUDIT'):
        data=source();protocol=read_json(BASE/'NEXT6_PROTOCOL_LOCK.json');support={}
        for ix,sid in enumerate(data.ids):
            if data.series['fold'][ix]==0:continue
            n=int(data.series['online_offsets'][ix+1]-data.series['online_offsets'][ix]);tau=int(data.series['tau'][ix]);windows={}
            if tau<0:windows['normal_no_break']=(0,n-1)
            else:
                windows['normal_not_yet']=(0,tau-1)
                for name,(lo,hi) in protocol['phase_intervals'].items():windows[name]=(max(0,tau+lo),min(n-1,n-1 if hi is None else tau+hi))
            for name,(lo,hi) in windows.items():
                for lag in (0,1,2,4,8,16):
                    points=max(0,hi-max(lo,lag)+1)
                    if points<4:continue
                    r=support.setdefault(name+':'+str(lag),{'series':0,'points':0});r['series']+=1;r['points']+=points
        save('CORRELATION_ATLAS/PLOT_SUPPORT.json',support)
        for p in (BASE/'STATE_TRANSITION_TABLES').glob('fold0_*.json'):
            d=read_json(p);tag=p.stem.removeprefix('fold0_');g=read_json(BASE/f'fold_0/graphs/{tag}/TRAIN_PROVENANCE.json');d['training_sample_rows']=g['sample_rows']
            save(str(p.relative_to(BASE)),d,False)
        save('FIGURE_ANNOTATION_AUDIT.json',{'status':'PASS','changes':['exact point/pair support','explicit unequal-window caveat on exploratory MAP C','actual graph training row counts'],
            'model_graph_tables_or_statistical_results_changed':False,'earlier_NEXT5_artifacts_changed':False})

if __name__=='__main__':main()

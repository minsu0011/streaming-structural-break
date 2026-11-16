"""Readable tables for every reachable factorized context, all outer folds."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
from src.next6.common import BASE,activity
from src.next6.nodes import NAMES,KOREAN,LAGS
from src.next6.evaluation import csv_write

def main():
    with activity('all_reachable_factorized_tables','RELATION_EVALUATION'):
        for fold in range(5):
            rows=[];g=joblib.load(BASE/f'fold_{fold}/graphs/outer_full/graph.joblib')
            for phase in (0,1):
                for j in range(16):
                    parents=g.parents[j]
                    for context in range(3**(1+len(parents))):
                        for dest in range(3):
                            r={'fold':fold,'phase':'normal' if phase==0 else 'post','node':j,'feature_name':NAMES[j],'korean':KOREAN[j],
                                'own_previous_state':context%3,'context':context,'next_state':dest,'probability':float(g.tables[phase,j,context,dest]),
                                'independent_training_series':int(g.context_support[phase,j,context]),'own_history_backoff':bool(g.context_support[phase,j,context]<8),
                                'smoothing_alpha':.5,'interpretation':'feature-state transition, not calibrated probability of a break'}
                            for k,(src,lagidx) in enumerate(parents):r.update({f'parent_{k+1}_node':src,f'parent_{k+1}_lag':LAGS[lagidx],f'parent_{k+1}_state':(context//3**(k+1))%3})
                            rows.append(r)
            csv_write(f'STATE_TRANSITION_TABLES/ALL_NODES_fold_{fold}.csv',rows)

if __name__=='__main__':main()

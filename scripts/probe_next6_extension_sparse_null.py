from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.categorical_controls import generated
from src.next6_extension.factorized import fit_pair

def interaction_error(null):
    result=[]
    for own in range(3):
        for a in range(2):
            for b in range(2):
                odds=null[:,own,a,b]*null[:,own,a+1,b+1]/(null[:,own,a,b+1]*null[:,own,a+1,b]);result.append(abs(float(np.log(odds[0])-np.log(odds[1]))))
    return max(result)

if __name__=='__main__':
    rows=[];worst=None
    with activity('sparse_null_coherence_coverage','MATHEMATICAL_AUDIT'):
        for groups in (16,32,64,128):
            for seed in range(32):
                a,b,p,ids=generated(groups,32,[20260913,910,seed],.6,'central80percent','marginal_shift');full,null,support,_=fit_pair(a,b,p,ids,0,4);error=interaction_error(null)
                r={'groups':groups,'seed':seed,'max_cross_phase_log_odds_difference':error,'backoff_contexts':int(np.sum(support<8))};rows.append(r)
                if worst is None or error>worst['max_cross_phase_log_odds_difference']:worst=r|{'support':support.tolist()}
        csv_write('SPARSE_NULL_COHERENCE_COVERAGE.csv',rows);save('SPARSE_NULL_COHERENCE_COVERAGE.json',{'status':'BACKOFF_BREAKS_EXACT_NULL' if worst['max_cross_phase_log_odds_difference']>1e-8 else 'PASS','worst':worst,'tables':len(rows),'actual_RP_selected_graphs_empty':True,'meaning':'Context-specific backoff can break the no-three-way-interaction property of the raw IPF fit; this is a model-coherence audit, not detector score tuning.'})
        print(worst)

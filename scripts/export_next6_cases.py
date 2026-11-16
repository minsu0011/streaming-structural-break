"""Fixed hash-order heldout success/delay/false alarm/miss examples."""
from pathlib import Path
import sys,os,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np
from src.next6.common import BASE,read_json,save,activity
from src.next6.data import arrays
from src.next6.evaluation import frozen_predictions
from src.next6.nodes import NAMES,KOREAN

def main():
    with activity('fixed_heldout_case_export','CASE_ANALYSIS'):
        data,meta,v=arrays();pred,_=frozen_predictions(data,meta);lock=read_json(BASE/'DIAGNOSTIC_LOCK.json');thresholds=read_json(BASE/'synthetic_predictions/fold_0/SUMMARY.json')['thresholds']
        ids=sorted(map(int,data.ids[data.series['fold']==0]),key=lambda i:hashlib.sha256(('NEXT6_CASE:'+str(i)).encode()).hexdigest());found={};counts={k:0 for k in ('success','delayed','false_alarm','miss','prealarm_break')}
        for sid in ids:
            ix=data.id_to_index[sid];a,b=data.series['online_offsets'][ix:ix+2];tau=int(data.series['tau'][ix]);score=pred['M3'][a:b];cross=np.flatnonzero(score>thresholds['M3']);first=int(cross[0]) if len(cross) else None
            category=None
            if tau<0 and first is not None:category='false_alarm'
            elif tau>=0:
                if first is None:category='miss'
                elif first<tau:category='prealarm_break'
                elif first-tau<=31:category='success'
                else:category='delayed'
            if category:counts[category]+=1
            if category in ('success','delayed','false_alarm','miss') and category not in found:
                found[category]={'category':category,'dataset_id':sid,'fold':0,'tau':tau,'first_alarm':first,'points':int(b-a),
                    'scores':{m:p[a:b].tolist() for m,p in pred.items() if m in ('M0','M1','M2','M3')},'thresholds':{m:thresholds[m] for m in ('M0','M1','M2','M3')},
                    'states':np.asarray(v['states'][a:b]).T.tolist(),'normalized_nodes':np.asarray(v['nodes'][a:b]).T.tolist()}
        save('CASE_STUDIES_NEXT6.json',{'scope':'outer fold0 heldout only; illustrative examples, no retuning','selection':lock['case_order'],
            'selection_method':'M3 fixed before outer scores','selection_threshold':'independent stationary synthetic calibration, strict >',
            'category_counts_across_1280_series':counts,'examples':found,'missing_categories':[k for k in ('success','delayed','false_alarm','miss') if k not in found],
            'feature_names':list(NAMES),'korean_meaning':list(KOREAN),'tau_is_plot_annotation_only':True,
            'uncertainty':'single illustrative series per category; no per-point confidence band claimed; overall whole-series uncertainty in paired results'})

if __name__=='__main__':main()

"""Prespecified final-length diagnostic only; length never enters inference."""
from pathlib import Path
import sys,os,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np
from src.next6.common import BASE,save,read_json,utc_now,digest,activity
from src.next6.data import arrays
from src.next6.evaluation import csv_write,frozen_predictions
from src.next5.fractures import pair_mass
from src.scoring.ts_auc import ts_auc

def main():
    p=argparse.ArgumentParser();p.add_argument('--lock-only',action='store_true');args=p.parse_args()
    path=BASE/'LENGTH_DIAGNOSTIC_LOCK.json'
    if not path.exists():
        assert not (BASE/'OUTER_SCORE_RELEASE.json').exists()
        save(path.name,{'created_utc':utc_now(),'bins':[['short',0,255],['middle',256,511],['long',512,None]],'diagnostic_only':True,'model_input':False,'source_sha256':digest(Path(__file__))})
    if args.lock_only:return
    with activity('fixed_final_length_diagnostics','EVALUATION'):
        data,meta,_=arrays();pred,_=frozen_predictions(data,meta);n=np.diff(data.series['online_offsets']);length=n[np.searchsorted(data.ids,meta['dataset_id'])];y=meta['target'];t=meta['time_online'];rows=[]
        for name,lo,hi in read_json(path)['bins']:
            mask=(length>=lo)&(True if hi is None else length<=hi)
            for f in range(5):
                take=mask&(meta['fold']==f);mass=pair_mass(y[take],t[take])
                if not mass:continue
                baseline=ts_auc(y[take],pred['M0'][take],t[take])
                for m,prediction in pred.items():
                    if not np.isfinite(prediction[take]).all():continue
                    value=ts_auc(y[take],prediction[take],t[take]);rows.append({'length_bin':name,'min_length':lo,'max_length':hi,'fold':f,'method':m,
                        'TS_AUC':float(value),'delta_M0':float(value-baseline),'series':len(np.unique(meta['dataset_id'][take])),'rows':int(take.sum()),
                        'positive_negative_time_pairs':mass,'final_length_diagnostic_only':True,'length_used_at_runtime':False})
        csv_write('LENGTH_RESULTS_NEXT6.csv',rows)

if __name__=='__main__':main()

"""Aggregate internal evidence observations without turning them into quality scores."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    out=ROOT/'artifacts/next2/null_evidence';rows=[];sources={};max_rows=[]
    for path in sorted(out.glob('*/*/RESULTS.json')):
        r=json.loads(path.read_text());assert r['status']=='PASS' and r['every_saved_prediction_exact'] and r['cumulative_max_memory_decreases']==0
        for name,digest in r['files_sha256'].items():assert sha256(path.parent/name)==digest
        sources[str(path.relative_to(ROOT))]=sha256(path);rows.append(pd.read_csv(path.parent/'FEATURE_GROWTH.csv',float_precision='round_trip'))
        with np.load(path.parent/'SNAPSHOTS.npz',allow_pickle=False) as arrays:
            maxima=arrays['score_path_max']
        for k,horizon in enumerate((1024,4096,16384)):
            max_rows.append({'candidate':r['candidate'],'background':r['background'],'horizon':horizon,'series':512,
                'mean_max_score':float(maxima[:,k].mean(dtype=np.float64)),
                **{'max_score_quantile_'+str(q):float(v) for q,v in zip((.5,.9,.95,.99),np.quantile(maxima[:,k],[.5,.9,.95,.99]))}})
    assert len(rows)==16
    pd.concat(rows,ignore_index=True).to_csv(out/'FEATURE_GROWTH.csv',index=False)
    pd.DataFrame(max_rows).to_csv(out/'SCORE_MAX_QUANTILES.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'model_background_cohorts':16,
        'prediction_points_rechecked':16*512*16384,'current_and_path_max_separately_recorded':True,
        'cumulative_max_invariant_violations':0,'source_sha256':sources,'script_sha256':sha256(Path(__file__)),
        'files_sha256':{name:sha256(out/name) for name in ('FEATURE_GROWTH.csv','SCORE_MAX_QUANTILES.csv')},
        'new_fits':0,'seal_rows':0,'promotion_authority':False})
    print('NEXT2 NULL INTERNAL EVIDENCE COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()

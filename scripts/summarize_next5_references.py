"""Score fixed reference combinations on the same held-out ResearchPool rows."""
from pathlib import Path
import csv
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next5.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next5.data import prepare_research
from src.next5.references import component
from src.scoring.ts_auc import ts_auc


def main():
    data=prepare_research();metadata=data.rows();predictions={};sources=[]
    for kind in ('J102','C205','B2_augmented_J','D2_expert'):
        predictions[kind],receipt=component(kind,'full','PRIMARY',metadata,data.guard);sources.append(receipt)
    predictions['BL80']=.8*predictions['J102']+.2*predictions['C205']
    predictions['B2']=.8*predictions['B2_augmented_J']+.2*predictions['C205']
    predictions['D2']=.9*predictions['BL80']+.1*predictions['D2_expert']
    rows=[];scores={}
    for name,p in predictions.items():
        scores[name]=[]
        for fold in range(5):
            mask=metadata['fold']==fold;score=float(ts_auc(metadata['target'][mask],p[mask],metadata['time_online'][mask]))
            scores[name].append(score);rows.append({'reference':name,'fold':fold,'TS_AUC':score,'rows':int(mask.sum()),'scope':'ResearchPool only'})
    baseline=np.asarray(scores['BL80']);summary=[]
    for name,values in scores.items():
        values=np.asarray(values);summary.append({'reference':name,'mean':float(values.mean()),'median':float(np.median(values)),
          'worst':float(values.min()),'mean_delta_vs_BL80':float((values-baseline).mean()),'folds':5,'ResearchPool_series':6400,
          'shadow_rows':0,'seal_rows':0,'new_fits_in_this_summary':0})
    out=BASE/'references/comparisons';out.mkdir(exist_ok=False)
    for filename,table in (('FOLD_SCORES.csv',rows),('SUMMARY.csv',summary)):
        with (out/filename).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'COMPLETE','reference_sources':sources,'summary':summary,
      'files_sha256':{name:digest(out/name) for name in ('FOLD_SCORES.csv','SUMMARY.csv')},'new_fits':0,'shadow_rows':0,'seal_rows':0,
      'historical_full_DEV_scores_are_not_matched_comparisons':True},immutable=True)
    for row in summary:print('NEXT5 MATCHED REFERENCE',row['reference'],row['mean'],row['mean_delta_vs_BL80'],flush=True)
    data.guard.check_state();progress('Matched ResearchPool reference scores complete','J102/C205 and unchanged BL80/B2/D2 combinations, all outer-fold training restricted to ResearchPool')


if __name__=='__main__':main()

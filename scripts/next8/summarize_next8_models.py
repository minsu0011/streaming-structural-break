"""Compile matched model scores and domain comparisons without nomination."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note,csv_write
from src.next8.model_summary import summarize_models


def main():
    lock=read_json(BASE/'MODEL_SUMMARY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'model_summary';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Model summary already exists; final audit promotion is a separate receipt')
    with activity('Aggregate paired model and major-domain uncertainty','MODEL_SUMMARY','B6_BL80_REDUNDANCY'):
        normal=summarize_models(False);early=summarize_models(True)
        atomic(directory/'NORMAL.json',{'timestamp':utc_now(),**normal});atomic(directory/'EARLY.json',{'timestamp':utc_now(),**early})
        csv_write('model_summary/NESTED_CV.csv',normal['folds']+early['folds'])
        csv_write('model_summary/PAIRED_UNCERTAINTY.csv',normal['comparisons']+early['comparisons'])
        csv_write('model_summary/DOMAIN_GENERALIZATION.csv',normal['domains']+early['domains'])
        atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'files_sha256':{
            name:digest(directory/name) for name in ('NORMAL.json','EARLY.json','NESTED_CV.csv','PAIRED_UNCERTAINTY.csv','DOMAIN_GENERALIZATION.csv')},
            'canonical_primary_unchanged':True,'promotion_requires_later_complete_audit':True})
    note('Complete paired score summary','Retain original numerical promotion thresholds and all same-input controls','B6_BL80_REDUNDANCY')


if __name__=='__main__':main()

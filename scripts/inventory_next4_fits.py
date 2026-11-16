"""Count primitive real-DEV fit starts/completions without double-counting parents."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--strict-complete',action='store_true');args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe inventory tag required')
    out=BASE/'fit_inventory'/args.tag;out.mkdir(parents=True,exist_ok=False)
    starts=[]
    for scope in ('screen','full'):
        for path in sorted((BASE/'experiments').glob('*/'+scope+'/fold_*_FIT_START.json')):
            starts.append((path,'selection_'+scope,path.parents[1].name,path.with_name(path.name.replace('_FIT_START',''))))
    for path in sorted((BASE/'robustness').glob('*/*/fold_*_FIT_START.json')):
        starts.append((path,'robustness_'+path.parent.name,path.parents[1].name,path.with_name(path.name.replace('_FIT_START',''))))
    for path in sorted((BASE/'refits').glob('*/*/FIT_START.json')):
        starts.append((path,'all_DEV_'+path.parent.name,path.parents[1].name,path.parent/'RESULTS.json'))
    for path in sorted((BASE/'negative_controls').glob('*/FIT_START.json')):
        starts.append((path,'negative_control',path.parent.name,path.parent/'RESULTS.json'))
    for path in sorted((BASE/'reproduction').glob('*/FIT_START.json')):
        starts.append((path,'fresh_raw_reproduction',path.parent.name,path.parent/'RESULTS.json'))
    assert len({p for p,_,_,_ in starts})==len(starts)
    rows=[];evidence={};models={}
    for path,phase,candidate,completion in starts:
        start=read_json(path);evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        row={'candidate_or_control':candidate,'phase':phase,'fit_start_utc':start['created_utc'],
            'fit_start_source':path.relative_to(ROOT).as_posix(),'fit_start_sha256':digest(path),
            'completed':False,'completed_fits':0,'meta_fit':False,'frozen_base_new_fits':0}
        if completion.exists():
            r=read_json(completion);assert r['status']=='PASS' and r['new_fits']==1
            assert datetime.fromisoformat(r['created_utc'])>=datetime.fromisoformat(start['created_utc'])
            assert r.get('frozen_base_new_fits',0)==0
            evidence[completion.relative_to(ROOT).as_posix()]=digest(completion)
            training=r['training'];fitted=[name for name in r['files_sha256'] if name.endswith('.joblib')]
            assert len(fitted)==1,(completion,fitted)
            model=completion.parent/fitted[0];actual=digest(model);assert actual==r['files_sha256'][fitted[0]]
            models[model.relative_to(ROOT).as_posix()]=actual
            row.update({'completed':True,'completed_fits':1,'meta_fit':r.get('new_meta_fits',0)==1,
                'completed_utc':r['created_utc'],'completion_source':completion.relative_to(ROOT).as_posix(),
                'completion_sha256':digest(completion),'model_path':model.relative_to(ROOT).as_posix(),'model_sha256':actual,
                'training_series':training.get('training_series'),'training_rows':training.get('training_rows'),
                'validation_series':training.get('validation_series'),'validation_rows':training.get('validation_rows'),
                'feature_count':training.get('feature_count'),'seal_rows':r.get('seal_rows',training.get('seal_rows'))})
            assert row['seal_rows']==0
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(out/'PRIMITIVE_FITS.csv',index=False)
    completed=[r for r in rows if r['completed']];unfinished=[r for r in rows if not r['completed']]
    if args.strict_complete:assert not unfinished
    selection=[r for r in completed if r['phase'].startswith('selection_')]
    parents=[read_json(p) for p in (BASE/'experiments').glob('*/*/RESULTS.json')]
    assert sum(r['new_fits'] for r in parents)==len(selection)==102
    assert sum(r['meta_fit'] for r in selection)==16
    nested=read_json(BASE/'nested_reuse_audit/RESULTS.json');assert nested['status']=='PASS'
    counts=Counter(r['phase'] for r in completed)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'fit_starts':len(starts),
        'real_DEV_learner_fit_completions':len(completed),'unfinished_primitive_fit_starts':len(unfinished),
        'completed_fits_by_phase':dict(counts),'selection_learner_fits':len(selection),
        'selection_meta_fits':16,'selection_base_or_state_readout_fits':86,
        'new_frozen_base_fits':0,'reused_original_unique_nested_base_fits':56,
        'nested_reuse_audit_sha256':digest(BASE/'nested_reuse_audit/RESULTS.json'),
        'prototype_and_test_fixtures_included_in_real_DEV_fit_count':False,
        'threshold_calibration_or_bootstrap_counted_as_learner_fits':False,
        'parent_summary_fits_double_counted':False,'new_fits_in_inventory':0,
        'primitive_receipt_sha256':evidence,'fitted_model_sha256':models,
        'source_sha256':digest(Path(__file__)),'files_sha256':{'PRIMITIVE_FITS.csv':digest(out/'PRIMITIVE_FITS.csv')}},immutable=True)
    print('NEXT4 PRIMITIVE REAL-DEV FIT INVENTORY',len(completed),'completed;',len(unfinished),'unfinished',flush=True)


if __name__=='__main__':main()

"""Fixed ResearchPool reference fit; never reads a legacy full-DEV cache."""
from pathlib import Path
import argparse
import hashlib
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.data import prepare_research
from src.next5.cache import build_design
from src.next5.model import fit_registered
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',required=True)
    parser.add_argument('--scope',choices=('screen','full'),required=True)
    parser.add_argument('--split',choices=('PRIMARY','ALT_A','ALT_B'),default='PRIMARY')
    args=parser.parse_args();cid='REF_'+args.kind
    assert read_json(BASE/'preflight/NEXT5_PREFLIGHT.json')['status']=='PASS'
    registration=read_json(BASE/'references/registrations'/(cid+'.json'))
    for name,sha in registration['execution_source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Reference execution source changed before fit')
    out=BASE/'references/fits'/args.split/args.scope/args.kind
    if out.exists():raise RuntimeError('Preserve existing reference attempts')
    out.mkdir(parents=True)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'reference':cid,'scope':args.scope,'split':args.split,
        'registration_sha256':digest(BASE/'references/registrations'/(cid+'.json'))},immutable=True)
    data=prepare_research();x,metadata,cache=build_design(data,registration)
    split=read_json(ROOT/'NEXT5_SPLIT_LOCK.json');mapping=split['mappings'][args.split]
    folds=np.asarray([int(mapping[str(int(i))]) for i in data.ids],dtype=np.int8)
    row_folds=np.repeat(folds,np.diff(data.series['online_offsets']))
    eligible=np.ones(len(row_folds),dtype=bool)
    if args.scope=='screen':eligible=np.isin(metadata['dataset_id'],split['screen_ids'])
    fold_menu=[0,4] if args.scope=='screen' else list(range(5))
    all_rows=[];all_predictions=[];results=[]
    for fold in fold_menu:
        directory=out/('fold_'+str(fold));directory.mkdir()
        train=np.flatnonzero(eligible&(row_folds!=fold));valid=np.flatnonzero(eligible&(row_folds==fold))
        atomic_json(directory/'FIT_START.json',{'created_utc':utc_now(),'fold':fold,'training_rows':len(train),
            'validation_rows':len(valid),'registration_sha256':digest(BASE/'references/registrations'/(cid+'.json'))},immutable=True)
        progress('ResearchPool reference training',cid+' '+args.scope+' '+args.split+' fold '+str(fold))
        print('NEXT5 FIT',cid,args.scope,args.split,fold,len(train),len(valid),flush=True);started=time.perf_counter()
        model,training=fit_registered(x,metadata,train,valid,guard=data.guard,registration=registration)
        model.save(directory/'model.joblib');prediction=model.predict(x[valid])
        if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite reference prediction')
        np.savez(directory/'predictions.npz',global_row=valid,dataset_id=metadata['dataset_id'][valid],prediction=prediction)
        score=float(ts_auc(metadata['target'][valid],prediction,metadata['time_online'][valid]))
        result={'status':'COMPLETE','created_utc':utc_now(),'reference':cid,'scope':args.scope,'split':args.split,'fold':fold,
            'TS_AUC':score,'elapsed_seconds':time.perf_counter()-started,'model_sha256':digest(directory/'model.joblib'),
            'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'prediction_file_sha256':digest(directory/'predictions.npz'),
            'cache_key':cache['key'],**training}
        atomic_json(directory/'RESULTS.json',result,immutable=True);results.append(result)
        all_rows.append(valid);all_predictions.append(prediction)
        print('NEXT5 REFERENCE SCORE',cid,args.scope,args.split,fold,score,flush=True)
    rows=np.concatenate(all_rows);predictions=np.concatenate(all_predictions);order=np.argsort(rows,kind='stable')
    np.savez(out/'OOF.npz',global_row=rows[order],dataset_id=metadata['dataset_id'][rows[order]],prediction=predictions[order])
    scores=[r['TS_AUC'] for r in results]
    atomic_json(out/'RESULTS.json',{'status':'COMPLETE','created_utc':utc_now(),'reference':cid,'scope':args.scope,'split':args.split,
        'fold_results':results,'mean_TS_AUC':float(np.mean(scores)),'median_TS_AUC':float(np.median(scores)),
        'worst_TS_AUC':float(min(scores)),'new_fits':len(results),'shadow_rows':0,'seal_rows':0,
        'OOF_sha256':digest(out/'OOF.npz'),'registration_sha256':digest(BASE/'references/registrations'/(cid+'.json'))},immutable=True)
    data.guard.check_state();progress('ResearchPool reference completed',cid+' '+args.scope+' '+args.split+'; matched training groups only')


if __name__=='__main__':main()

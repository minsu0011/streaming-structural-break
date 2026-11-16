"""Count actual training events from model-producing receipts, never from candidate names."""
from pathlib import Path
import sys
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    out=ROOT/'artifacts/next2/training_inventory';out.mkdir(parents=True,exist_ok=True)
    records=[];skipped=[]
    def add(category,candidate,model,receipt,detail=''):
        model=Path(model);receipt=Path(receipt)
        if not model.exists() or not receipt.exists():raise RuntimeError('A claimed training event has no model/receipt')
        records.append({'category':category,'candidate':candidate,'model_path':str(model.relative_to(ROOT)),
            'model_sha256':sha256(model),'receipt_path':str(receipt.relative_to(ROOT)),'receipt_sha256':sha256(receipt),'detail':detail,'fit_count':1})
    for path in sorted((ROOT/'artifacts/next2/experiments').glob('*/RESULTS.json')):
        report=json.loads(path.read_text());c=report['candidate'];eid=c['experiment_id']
        if c.get('blend') or c.get('parity_reference'):
            skipped.append({'candidate':eid,'reason':'fixed blend assembly' if c.get('blend') else 'exact prior experiment parity reuse','new_fits':0})
            continue
        for scope in ('screen','full'):
            for model in sorted((path.parent/scope).glob('fold_?.joblib')):
                receipt=model.with_suffix('.json');r=json.loads(receipt.read_text());assert r['status']=='PASS'
                assert r['files_sha256'][model.name]==sha256(model)
                add('ORIGINAL_'+scope.upper(),eid,model,receipt,report['status'])
    for model in sorted((ROOT/'artifacts/next2/robustness').glob('*/*/fold_?.joblib')):
        eid=model.parent.parent.name
        if eid.startswith('BL2_'):continue
        receipt=model.with_suffix('.json');r=json.loads(receipt.read_text());assert r['status']=='PASS'
        assert r['files_sha256'][model.name]==sha256(model)
        add('ALTERNATE_OR_H_HOLDOUT',eid,model,receipt,model.parent.name)
    for model in sorted((ROOT/'artifacts/next2/alarm_calibration').glob('fold_*/*.joblib')):
        receipt=model.with_name(model.stem+'_FIT.json')
        add('NESTED_ALARM_CALIBRATION',model.stem,model,receipt,'Smaller train split; calibration and outer evaluation excluded from fitting')
    for receipt in sorted((ROOT/'artifacts/next2/refits/C205_DEPENDENCE').glob('*/TRAINING.json')):
        r=json.loads(receipt.read_text());assert r['status']=='PASS'
        assert sha256(receipt.parent/'model.joblib')==r['model_sha256']
        add('ALL_DEV_RESEARCH_REFIT','C205_DEPENDENCE',receipt.parent/'model.joblib',receipt,receipt.parent.name)
    for receipt in sorted((ROOT/'artifacts/next2/entry_reproduction').glob('*/*/training_components/*/TRAINING.json')):
        r=json.loads(receipt.read_text());assert r['status']=='PASS'
        assert sha256(receipt.parent/'model.joblib')==r['model_sha256']
        add('ACTUAL_JOBLIB_ENTRY_REPRODUCTION',r['candidate']['experiment_id'],receipt.parent/'model.joblib',receipt,
            'Count completed fit even when the outer strict-byte reproduction failed and required canonical serialization')
    frame=pd.DataFrame(records)
    assert not frame.model_path.duplicated().any()
    frame.to_csv(out/'TRAINING_EVENTS.csv',index=False)
    counts=frame.groupby('category').fit_count.sum().to_dict()
    write_json(out/'RESULTS.json',{'status':'RESEARCH_INVENTORY','created_utc':utc_now(),'script_sha256':sha256(Path(__file__)),
        'completed_training_events':int(frame.fit_count.sum()),'by_category':{k:int(v) for k,v in counts.items()},
        'zero_fit_experiment_records':skipped,'table_sha256':sha256(out/'TRAINING_EVENTS.csv'),
        'counts_distinct_from_named_experiments':True,'final_independent_canonical_training_not_yet_included':True,
        'scope':'Only original NEXT2 model-producing runs; prior-session reused fits, fixed model assemblies, repackaging, runtime replay and diagnostic scoring are not counted as new fits.'})
    print('NEXT2 ACTUAL TRAINING EVENTS',int(frame.fit_count.sum()),counts,flush=True)


if __name__=='__main__':main()

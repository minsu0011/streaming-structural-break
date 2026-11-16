"""Fixed training-fit sensitivity of primary/reference/complement counterexamples."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path=ROOT/'configs/next2/FINALIST_FIT_SENSITIVITY_LOCK.json'
    if path.exists():
        print('NEXT2 FIT SENSITIVITY LOCK RETAINED',flush=True)
        return
    models=[]
    for fold in range(5):
        for eid in ('J102_SERIAL_VARIANCE_512','Y001_LONG_VARIANCE_ARCH1','B104_LONG_VARIANCE_NO_GLR_MEMORY','C205_DEPENDENCE'):
            folder=ROOT/('artifacts/next2/experiments' if eid=='C205_DEPENDENCE' else 'artifacts/next/experiments')/eid
            r=json.loads((folder/'RESULTS.json').read_text())
            model=folder/'full'/f'fold_{fold}.joblib'
            item={'experiment_id':eid,'fold':fold,'candidate':r['candidate'],
                'model_path':model.relative_to(ROOT).as_posix(),'model_sha256':sha256(model),
                'candidate_receipt_sha256':sha256(folder/'RESULTS.json'),'full_cv_status':r['status']}
            if fold==0 and eid!='C205_DEPENDENCE':
                existing=ROOT/'artifacts/next2/component_stress_cv0'/eid
                proof=json.loads((existing/'RESULTS.json').read_text())
                if proof['status']!='PASS' or proof['identity']['model_sha256']!=item['model_sha256']:
                    raise RuntimeError('Existing CV0 stress is not the same fitted model')
                item['reuse_directory']=existing.relative_to(ROOT).as_posix()
                item['reuse_receipt_sha256']=sha256(existing/'RESULTS.json')
            models.append(item)
    write_json(path,{'status':'FROZEN_FIT_SENSITIVITY_DIAGNOSTIC','created_utc':utc_now(),
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),'models':models,
        'new_model_fits':0,'new_predictive_components':17,'reused_complete_component_studies':3,
        'fixed_blend':{'primary':'J102_SERIAL_VARIANCE_512','complement':'C205_DEPENDENCE','weight':.8},
        'purpose':'Separate conditional DGP sampling uncertainty from variation across the five existing overlapping training fits.',
        'scope':'The same nine null-calibration cohorts, eighteen counterexamples and four long-null cohorts. Every existing fold is included.',
        'reporting':'Report each fitted model separately plus ranges across fits. Do not treat five overlapping training sets as five independent datasets, or average their predictions into an unregistered ensemble.',
        'nomination_context':'All-DEV finalist improves dependence and worsens variance decrease/outlier alarms. Check whether these directions persist across existing training fits.',
        'selection_authority':False,'gate_changes':False,'seal_rows':0,'reduced_new_usage':0})
    print('NEXT2 FIT SENSITIVITY LOCKED',len(models),'models, 17 new prediction studies, zero fits',flush=True)


if __name__=='__main__':
    main()

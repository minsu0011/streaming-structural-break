"""Fit the time-matched early BL80 reference in the new namespace only."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,read_json,atomic,digest,utc_now,activity,note
from src.next8.data import source,outer_scope,rows_for
from src.next8.nested_auxiliary import early_scope
from src.next8.cache import immutable_npz
from src.next8.early_reference import fit_early_reference
from src.next6.data import reference_matrix
from src.next5.model import FeatureModel


def main(fold):
    assert read_json(BASE/'EARLY_REFERENCE_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock=read_json(BASE/'EARLY_REFERENCE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    data=source();metadata=data.rows();scope=early_scope(outer_scope(fold,data),True)
    directory=BASE/'early_baseline'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Completed early reference cannot be silently rerun')
    rows=rows_for(metadata,scope.validation_ids);rows=rows[metadata['time_online'][rows]>=400]
    scope.admit_validation(np.unique(metadata['dataset_id'][rows]));predictions=[];receipts=[]
    for kind in ('J102','C205'):
        p=directory/kind;p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json')
            assert digest(p/'model.joblib')==done['model_sha256'] and digest(p/'PREDICTIONS.npz')==done['prediction_sha256']
            with np.load(p/'PREDICTIONS.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['global_row'],rows);predictions.append(a['prediction'].copy())
            receipts.append(done);continue
        if any(p.iterdir()):raise RuntimeError('Partial early baseline preserved for explicit repair')
        with activity(f'Early-only baseline {scope.name} {kind}','EARLY_REFERENCE_FIT','B5_DOMAIN_GENERALIZATION'):
            x=reference_matrix(kind,data)
            reg_path=ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json'
            registration=read_json(reg_path)
            assert digest(reg_path)==lock['registration_sha256'][kind]
            model=fit_early_reference(x,metadata,scope,registration,data.guard)
            model.save(p/'model.joblib');prediction=np.empty(len(rows),dtype=np.float64)
            for start in range(0,len(rows),50000):prediction[start:start+50000]=model.predict(x[rows[start:start+50000]])
            immutable_npz(p/'PREDICTIONS.npz',global_row=rows,dataset_id=metadata['dataset_id'][rows],prediction=prediction)
            done={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'kind':kind,'training':model.training,
                'model_sha256':digest(p/'model.joblib'),'prediction_sha256':digest(p/'PREDICTIONS.npz'),
                'registration_sha256':digest(reg_path),'early_to_late_scored':False,'canonical_files_mutated':False}
            atomic(p/'DONE.json',done);receipts.append(done);predictions.append(prediction)
            del x,model
    immutable_npz(directory/'BL80_PREDICTIONS.npz',global_row=rows,dataset_id=metadata['dataset_id'][rows],
        prediction=.8*predictions[0]+.2*predictions[1])
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'prediction_sha256':digest(directory/'BL80_PREDICTIONS.npz'),'components':receipts,
        'baseline_name':'M0_EARLY_T199_REFIT','heldout_min_time':400,'same_training_cutoff_as_new_early_models':True,
        'training_sampling':'Original baseline recipe uses all permitted t<=199 rows; new model menu uses preregistered stride8',
        'score_released':False,'reference_must_not_be_confused_with_existing_alltime_M0':True})
    note('Early-to-late reference predictions frozen',scope.name+'; all training labels<=199; new canonical role unchanged','B5_DOMAIN_GENERALIZATION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    main(p.parse_args().fold)

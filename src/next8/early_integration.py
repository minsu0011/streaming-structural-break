"""Early-only integration qualification, including excluded-group inputs."""
import numpy as np
from src.next8.data import rows_for
from src.next8.features import DETAILED,UNKNOWN
from src.next8.joint_models import joint_design,model_design


def early_prediction_rows(cache,scope):
    rows=rows_for(cache.meta,scope.validation_ids)
    rows=rows[cache.meta['time_online'][rows]<=199]
    scope.admit_validation(np.unique(cache.meta['dataset_id'][rows]))
    return rows


def predict_early_head(cache,scope,selection,model,actual_method,rows,cross):
    rows=np.asarray(rows,dtype=np.int64)
    if model.training.get('early_training_cutoff')!=199:raise ValueError('Early-only fitted head required')
    if selection['scope']!=scope.provenance() or model.training['training_ids_sha256']!=scope.provenance()['training_ids_sha256']:
        raise ValueError('Exact early excluded-group selection and model required')
    scope.admit_validation(np.unique(cache.meta['dataset_id'][rows]))
    if np.any(cache.meta['time_online'][rows]>199):raise ValueError('Late row cannot qualify early integration')
    if cross.shape!=(len(rows),len(DETAILED)):raise ValueError('Early state row order must match')
    output=np.empty(len(rows),dtype=np.float32)
    for start in range(0,len(rows),50000):
        sl=slice(start,start+50000)
        design=joint_design(cache,rows[sl],selection,actual_method,cross[sl])
        output[sl]=model.predict_proba(model_design(model,design))[:,1]
    return output


def normal_integration_relevant(receipts):
    if len(receipts)!=5 or any(r.get('status')!='COMPLETE' for r in receipts):
        raise ValueError('All five frozen normal integration qualifications required')
    return any(r['eligible'] for r in receipts)

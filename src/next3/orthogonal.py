"""A single train-only low-complexity residual complement, without tuning."""
import numpy as np
from src.next.fitting import weights_from_training

KIND='O1_LINEAR_RESIDUAL'
RIDGE=.001
RESIDUAL_WEIGHT=.2


def predictors(features,center,scale):
    values=np.asarray(features,dtype=np.float64)[:,[0,2]]
    z=np.clip((values-center)/scale,-8.,8.)
    return np.column_stack([np.ones(len(z)),z])


def fit_residual(features,metadata):
    values=np.asarray(features,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=6 or not np.isfinite(values).all():
        raise ValueError('Six finite nested causal inputs required')
    raw=values[:,[0,2]]
    center=raw.mean(axis=0);scale=np.maximum(raw.std(axis=0),1e-6)
    x=predictors(values,center,scale)
    weight=weights_from_training(metadata['target'],metadata['time_online'],metadata['dataset_id'],'W3')
    norm=float(weight.sum())
    moment=x.T@(weight[:,None]*x)/norm
    response=x.T@(weight*values[:,1])/norm
    penalty=np.diag([0.,RIDGE,RIDGE])
    coefficients=np.linalg.solve(moment+penalty,response)
    if not np.isfinite(coefficients).all():raise RuntimeError('Non-finite residual coefficients')
    return {'format':'NEXT3_RESIDUAL_JSON_V1','kind':KIND,'center':center.tolist(),'scale':scale.tolist(),
        'coefficients':coefficients.tolist(),'ridge':RIDGE,'residual_weight':RESIDUAL_WEIGHT,
        'explanatory_inputs':['J102_inner_OOF_score','current_scale_strength'],
        'response':'C205_inner_OOF_score; no mechanism label',
        'training_rows':len(values),'weighting':'W3','input_z_clip':8.,
        'normal_equation_max_abs_error':float(np.max(np.abs((moment+penalty)@coefficients-response)))}


def residual_values(descriptor,features):
    if descriptor.get('format')!='NEXT3_RESIDUAL_JSON_V1' or descriptor.get('kind')!=KIND:
        raise ValueError('Unknown residual descriptor')
    if descriptor['ridge']!=RIDGE or descriptor['residual_weight']!=RESIDUAL_WEIGHT:
        raise ValueError('Frozen residual recipe changed')
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all():raise ValueError('Six finite features required')
    explained=predictors(x,np.asarray(descriptor['center']),np.asarray(descriptor['scale']))@np.asarray(descriptor['coefficients'])
    return x[:,1].astype(np.float64)-explained


def predict_residual(descriptor,features):
    primary=np.asarray(features,dtype=np.float32)[:,0].astype(np.float64)
    return np.asarray(np.clip(primary+RESIDUAL_WEIGHT*residual_values(descriptor,features),0.,1.),dtype=np.float32)

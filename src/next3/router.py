"""Small causal specialist gates; no identifiers, labels or future inputs."""
import numpy as np

HEURISTICS = ('R1_SCORE_GAP', 'R1_EVIDENCE_RATIO')
GATE_NAMES = ('J102_score', 'C205_score', 'scale_strength', 'dependence_strength',
              'raw_energy_current', 'forecast_current')


def gate_design(primary, complement, primary_design, complement_design, primary_names, complement_names):
    p = np.asarray(primary, dtype=np.float32)
    c = np.asarray(complement, dtype=np.float32)
    left = np.asarray(primary_design)
    right = np.asarray(complement_design)
    if p.ndim != 1 or c.shape != p.shape or len(left) != len(p) or len(right) != len(p):
        raise ValueError('Aligned current-point feature rows required')
    scale_columns = [primary_names.index('serial_variance__'+channel+'_'+stat)
        for channel in ('raw_energy','log_variance_forecast') for stat in ('ewma8','ewma32')]
    dependence_columns = [complement_names.index('next2_complement__dependence_'+str(lag)+'_'+stat)
        for lag in range(4) for stat in ('ewma8','ewma32')]
    raw = primary_names.index('serial_variance__raw_energy_current')
    forecast = primary_names.index('serial_variance__log_variance_forecast_current')
    result = np.column_stack([p,c,np.max(np.abs(left[:,scale_columns]),axis=1),
        np.max(np.abs(right[:,dependence_columns]),axis=1),left[:,raw],left[:,forecast]]).astype(np.float32)
    if not np.isfinite(result).all() or ((p<0)|(p>1)|(c<0)|(c>1)).any():
        raise ValueError('Finite gate features and bounded component scores required')
    return result


def heuristic_weight(features, kind):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(GATE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Exact finite causal gate input contract required')
    if kind == 'R1_SCORE_GAP':
        gap = x[:,1]-x[:,0]
        return np.where(gap>.1,.3,np.where(gap>0,.2,np.where(gap>-.1,.1,0.)))
    if kind == 'R1_EVIDENCE_RATIO':
        ratio = x[:,3]/(1.+x[:,2])
        return np.where(x[:,3]<1.,0.,np.where(ratio>4.,.3,np.where(ratio>2.,.2,np.where(ratio>1.,.1,0.))))
    raise ValueError('Unregistered heuristic')


def blend_with_weight(primary, complement, weight):
    p = np.asarray(primary,dtype=np.float32).astype(np.float64)
    c = np.asarray(complement,dtype=np.float32).astype(np.float64)
    w = np.asarray(weight,dtype=np.float64)
    if p.shape != c.shape or w.shape != p.shape or not np.isfinite(w).all() or ((w<0)|(w>.3)).any():
        raise ValueError('Aligned component predictions and weight in [0,.3] required')
    if not np.isfinite(p).all() or not np.isfinite(c).all() or ((p<0)|(p>1)|(c<0)|(c>1)).any():
        raise ValueError('Invalid component score')
    return np.asarray((1.-w)*p+w*c,dtype=np.float32)


def heuristic_predict(features, kind):
    weight = heuristic_weight(features,kind)
    return blend_with_weight(features[:,0],features[:,1],weight),weight

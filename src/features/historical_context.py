"""Historical-only context beside the unchanged 51 AR-calibrated ABCD features."""
import hashlib,json
from pathlib import Path
import numpy as np
from src.features.config import CONFIG
from src.features.streaming import feature_names
from src.features.ar_residual_input import ARResidualCalibratedState,historical_innovations,implementation_hash as ar_hash
from src.features.historical_calibration import CalibrationPolicy

POLICY=json.loads((Path(__file__).resolve().parents[2]/'configs/historical_context.json').read_text())
CONTEXT_NAMES=tuple(POLICY['context_columns'])


def implementation_hash(policy,config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes()+json.dumps(POLICY,sort_keys=True).encode()+ar_hash(policy,config).encode()).hexdigest()


def names(config=CONFIG):return feature_names(config)[:51]+CONTEXT_NAMES


def model_families(families):
    if any(f not in 'ABCDG' for f in families):raise ValueError('Context representation exposes only ABCD and historical G')
    return families.replace('G','EFQ')


def historical_context(historical,reference,coefficients,config=CONFIG):
    raw=np.asarray(historical,dtype=np.float32).astype(np.float64)
    finite=raw[np.isfinite(raw)];median=float(np.median(finite)) if len(finite) else 0.
    h=np.where(np.isfinite(raw),raw,median)
    if not len(h):h=np.zeros(1,dtype=np.float64)
    std=max(float(h.std(ddof=1)) if len(h)>1 else 1.,config.scale_floor)
    mad=max(float(np.median(abs(h-median)))*config.mad_consistency_factor,std*config.mad_min_std_fraction,config.scale_floor)
    z=np.clip((np.where(np.isfinite(raw),raw,reference[0])-reference[0])/reference[1],-config.clip_z,config.clip_z)
    if not len(z):z=np.zeros(1,dtype=np.float64)
    centered=z-z.mean();variance=max(float(np.mean(centered*centered)),config.moment_floor)
    acf=float(np.mean(centered[1:]*centered[:-1])/variance) if len(z)>1 else 0.
    residual=historical_innovations(z,coefficients).astype(np.float64) if len(z)>len(coefficients) else z
    ratio=max(float(np.mean(residual*residual)),config.moment_floor)/max(float(np.mean(z*z)),config.moment_floor)
    kurtosis=max(0.,float(np.mean(centered**4)/variance**2)-3.)
    result=np.array([np.log1p(len(raw)),np.log(np.clip(mad/std,1e-4,1e4)),np.clip(acf,-1,1),
        np.log1p(np.linalg.norm(coefficients)),np.log(np.clip(ratio,1e-8,1e8)),np.log1p(min(kurtosis,1e8))],dtype=np.float32)
    if not np.isfinite(result).all():raise RuntimeError('Nonfinite historical context')
    return result


class HistoricalContextState(ARResidualCalibratedState):
    def __init__(self,historical,*,config=CONFIG,policy=CalibrationPolicy(method='historical_median_mad')):
        super().__init__(historical,config=config,policy=policy)
        self.context=historical_context(historical,self.reference,self.coefficients,config)
        self.output=np.zeros(57,dtype=np.float32)
    def update_and_get(self,point):
        self.output[:51]=super().update_and_get(point)[:51];self.output[51:]=self.context
        return self.output
    @property
    def state_array_bytes(self):return super().state_array_bytes+self.context.nbytes+self.output.nbytes

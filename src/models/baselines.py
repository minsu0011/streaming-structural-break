"""Independent statistical baseline ladder. No fitted parameters."""
import math
import numpy as np
from src.features.streaming import StreamingFeatureState

BASELINES = ("B0_constant", "B1_official_ewma", "B2_multiscale_mean", "B3_cusum", "B4_scale", "B5_tail", "B6_dependence", "B7_manual")

class OfficialEWMA:
    """Exact finite-input arithmetic from the pinned official notebook."""
    def __init__(self, historical):
        h = np.asarray(historical, dtype=np.float64)
        self.mean = float(h.mean()) if len(h) else 0.0
        self.sd = max(float(h.std(ddof=1)) if len(h)>1 else 1.0, 1e-8)
        self.ewma = self.mean
        self.n_eff = 0.0

    def predict_one(self, point):
        self.ewma = 0.95*self.ewma + 0.05*float(point)
        self.n_eff = 0.95*self.n_eff + 1.0
        se = self.sd/math.sqrt(max(self.n_eff,1.0))
        z = (self.ewma-self.mean)/max(se,1e-8)
        return math.tanh(abs(z)/3.0)

def from_features(name, f):
    if name == "B0_constant":
        return 0.5
    mean = max(float(f[7]),float(f[22]),float(f[37]))/3
    cusum = max(float(f[3]),float(f[4]))/4
    scale = max(abs(float(f[9])),abs(float(f[24])),abs(float(f[39])))/2
    tail = max(abs(float(f[11])),abs(float(f[26])),abs(float(f[41])),float(f[14]),float(f[29]),float(f[44]))/4
    dependence = max(abs(float(f[15])),abs(float(f[30])),abs(float(f[45])),abs(float(f[20])),abs(float(f[35])),abs(float(f[50])))/4
    evidence = {"B2_multiscale_mean":mean,"B3_cusum":cusum,"B4_scale":scale,"B5_tail":tail,"B6_dependence":dependence,"B7_manual":(mean+scale+tail+dependence)/4}
    if name not in evidence:
        raise ValueError(name)
    value = evidence[name]
    # Rational squash retains rank resolution longer than tanh for large evidence.
    return value/(1.0+value)

class BaselineDetector:
    def __init__(self, name, historical):
        if name not in BASELINES:
            raise ValueError(name)
        self.name = name
        self.state = OfficialEWMA(historical) if name == "B1_official_ewma" else (None if name == "B0_constant" else StreamingFeatureState(historical))

    def predict_one(self, point):
        if self.name == "B0_constant":
            return 0.5
        if self.name == "B1_official_ewma":
            return self.state.predict_one(point)
        return from_features(self.name,self.state.update_and_get(point))

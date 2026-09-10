"""Authoritative feature configuration; all defaults live in configs/features.json."""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / 'configs/features.json'

@dataclass(frozen=True)
class FeatureConfig:
    schema_version: int
    scales: tuple
    alpha_rule: str
    lags: tuple
    tail_thresholds: tuple
    quantile_probabilities: tuple
    clip_z: float
    scale_floor: float
    moment_floor: float
    rate_variance_floor: float
    cusum_drift: float
    memory_decay: float
    ar_phi_limit: float
    mad_consistency_factor: float
    mad_min_std_fraction: float
    normalization: str
    finite_value_bound: float

    def __post_init__(self):
        if len(self.scales) != 3 or any(s <= 1 for s in self.scales) or tuple(sorted(self.scales)) != self.scales:
            raise ValueError('Three ordered scales greater than one required')
        if len(self.lags) != 4 or self.lags[0] != 1 or any(int(l) != l or l < 1 for l in self.lags):
            raise ValueError('Four positive integer lags starting at one required')
        if len(self.tail_thresholds) != 2 or len(self.quantile_probabilities) != 4:
            raise ValueError('Two tail thresholds and four quantile boundaries required')
        if self.alpha_rule != '1/scale' or self.normalization not in ('mean_std', 'median_mad'):
            raise ValueError('Unknown alpha or normalization policy')
        if not 0 < self.memory_decay <= 1 or self.scale_floor <= 0 or self.clip_z <= 0:
            raise ValueError('Invalid numeric policy')

    @property
    def alphas(self):
        return tuple(1.0 / scale for scale in self.scales)

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def variant(self, **changes):
        return replace(self, **changes)

def load_config(path=CONFIG_PATH):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('scales', 'lags', 'tail_thresholds', 'quantile_probabilities'):
        values[key] = tuple(values[key])
    return FeatureConfig(**values)

CONFIG = load_config()

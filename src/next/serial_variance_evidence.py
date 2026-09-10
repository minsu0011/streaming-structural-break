"""Same H variance channels, with an H-only serial correction to GLR/Bayes."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical, innovation_update
from src.next.normalization import fit_normalization
from src.next.variance_evidence import CHANNELS, variance_channel_update, historical_variance_channels
from src.next.variance_evidence import implementation_hash as parent_hash
from src.next.channel_evidence import make_channel_args, CHANNEL_STATS
from src.next.serial_channel_evidence import historical_correlations, sum_variances, serial_channel_step, source_hash as serial_hash


@dataclass(frozen=True)
class SerialVarianceConfig:
    order: int = 8
    normalization: str = 'arch1'
    max_age: int = 128


def _config(settings):
    config = SerialVarianceConfig(**settings)
    if config.order != 8 or config.normalization != 'arch1' or config.max_age not in (128, 512):
        raise ValueError('Unregistered serial-variance correction recipe')
    return config


@njit(cache=True)
def serial_variance_step(point, filter_args, parameters, conditional, mode, calibration, work, evidence_args, age_variances):
    residual = innovation_update(point, *filter_args)
    variance_channel_update(residual, parameters, conditional, mode, work)
    for j in range(len(work)):
        work[j] = (work[j]-calibration[0, j])/calibration[1, j]
    return serial_channel_step(work, evidence_args, age_variances)


@njit(cache=True)
def replay_serial_variance(points, filter_args, parameters, conditional, mode, calibration, work, evidence_args, age_variances):
    result = np.empty((len(points), len(evidence_args[-1])), dtype=np.float32)
    for j in range(len(points)):
        result[j] = serial_variance_step(points[j], filter_args, parameters, conditional, mode, calibration, work, evidence_args, age_variances)
    return result


class SerialVarianceState:
    def __init__(self, historical, config=SerialVarianceConfig()):
        config = _config(asdict(config))
        self.config = config
        fit = fit_historical(historical, config.order)
        profile = fit_normalization(fit['historical_innovations'], config.normalization)
        self.parameters, self.mode = profile['parameters'], profile['mode']
        h_variance = self.parameters[2]/max(1.-self.parameters[3]-self.parameters[4], .05)
        conditional = np.full(3, h_variance)
        channels = historical_variance_channels(fit['historical_innovations'], self.parameters, conditional, self.mode)
        np.testing.assert_allclose(conditional, profile['conditional'], rtol=1e-14, atol=1e-14)
        channels = channels[min(160, len(channels)//4):]
        self.calibration = np.stack([channels.mean(axis=0), np.maximum(channels.std(axis=0), 1e-6)])
        self.correlations = historical_correlations((channels-self.calibration[0])/self.calibration[1])
        self.conditional = profile['conditional'].copy()
        self.filter_args = tuple(fit[key] for key in ('reference', 'coefficients', 'ring', 'counter'))
        self.work = np.empty(len(CHANNELS))
        self.evidence_args = make_channel_args(len(CHANNELS), config.max_age)
        self.age_variances = sum_variances(self.evidence_args[0], self.correlations)

    def update(self, point):
        return serial_variance_step(float(point), self.filter_args, self.parameters, self.conditional, self.mode, self.calibration, self.work, self.evidence_args, self.age_variances)

    def replay(self, points):
        return replay_serial_variance(np.asarray(points, dtype=np.float32), self.filter_args, self.parameters, self.conditional, self.mode, self.calibration, self.work, self.evidence_args, self.age_variances)

    @property
    def state_array_bytes(self):
        return sum(value.nbytes for value in self.filter_args+self.evidence_args)+self.parameters.nbytes+self.conditional.nbytes+self.calibration.nbytes+self.work.nbytes+self.correlations.nbytes+self.age_variances.nbytes


def feature_names(settings):
    _config(settings)
    return tuple(channel+'_'+stat for channel in CHANNELS for stat in CHANNEL_STATS)


def feature_groups(settings):
    _config(settings)
    return tuple('VARIANCE_GLR' if stat.startswith('glr') else 'VARIANCE_BAYES' if stat.startswith('bayes') else 'VARIANCE_LOCAL' for channel in CHANNELS for stat in CHANNEL_STATS)


def make_state(historical, settings):
    return SerialVarianceState(historical, _config(settings))


def implementation_hash(settings):
    config = asdict(_config(settings))
    return hashlib.sha256(Path(__file__).read_bytes()+parent_hash(config).encode()+serial_hash().encode()+json.dumps(config, sort_keys=True).encode()).hexdigest()

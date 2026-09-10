"""Fixed-memory O(H+T) historical-reference features used by train and infer."""
import hashlib
from pathlib import Path
import numpy as np
from numba import njit

from src.features.config import CONFIG, FeatureConfig
SCALES = np.array(CONFIG.scales, dtype=np.float64)
CLIP_Z = CONFIG.clip_z
CUSUM_DRIFT = CONFIG.cusum_drift
MEMORY_DECAY = CONFIG.memory_decay
CATALOG = [
    ("current_z", "A"), ("current_abs_z", "A"), ("current_robust_z", "C"),
    ("cusum_positive", "A"), ("cusum_negative", "A"), ("running_mean", "A")]
for scale in CONFIG.scales:
    CATALOG += [(f"{name}_{scale}", family) for name, family in [
        ("mean_z", "A"), ("abs_mean_z", "A"),
        ("log_second_ratio", "B"), ("log_variance_ratio", "B"), ("absolute_moment_delta", "B"),
        ("tail2_delta", "C"), ("tail3_delta", "C"), ("sign_delta", "C"), ("quantile_l1", "C"),
        ("lag1_delta", "D"), ("lag2_delta", "D"), ("lag4_delta", "D"), ("lag8_delta", "D"),
        ("ar_mean_z", "D"), ("ar_log_second_ratio", "D")]]
CATALOG += [("fast_slow_mean", "E"), ("fast_slow_scale", "E"), ("fast_slow_dependence", "E"),
            ("evidence_max", "F"), ("evidence_decayed_max", "F"), ("current_missing", "Q")]
FEATURE_NAMES = tuple(name for name, _ in CATALOG)
FEATURE_FAMILIES = tuple(family for _, family in CATALOG)

def feature_names(config=CONFIG):
    names=list(FEATURE_NAMES)
    for s,scale in enumerate(config.scales):
        for j in range(15):
            name=names[6+s*15+j].rsplit('_',1)[0]
            if 9 <= j <= 12:
                name=f'lag{config.lags[j-9]}_delta'
            names[6+s*15+j]=f'{name}_{scale}'
    return tuple(names)

def feature_version(config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes() + (Path(__file__).parent/'config.py').read_bytes() + config.digest().encode()).hexdigest()

@njit(cache=False)
def _update(point, reference, baseline, thresholds, ring, ew, counters, output, parameters, alphas, lags):
    # No labels, IDs, future suffix or final length enter this kernel.
    missing = not np.isfinite(point)
    x = reference[0] if missing else point
    clip_z, drift, decay, moment_floor, rate_floor, tail_low, tail_high = parameters
    z = min(max((x - reference[0]) / reference[1], -clip_z), clip_z)
    robust = min(max((x - reference[2]) / reference[3], -clip_z), clip_z)
    t = int(counters[0]) + 1
    counters[0] = t
    counters[1] = max(0.0, counters[1] + z - drift)
    counters[2] = max(0.0, counters[2] - z - drift)
    counters[3] += (z - counters[3]) / t
    output[0:6] = (z, abs(z), robust, np.log1p(counters[1]), np.log1p(counters[2]), counters[3])
    stats = np.empty(17, dtype=np.float64)
    stats[0] = z
    stats[1] = z*z
    stats[2] = abs(z)
    stats[3] = 1.0 if abs(z) > tail_low else 0.0
    stats[4] = 1.0 if abs(z) > tail_high else 0.0
    stats[5] = 1.0 if z > 0.0 else 0.0
    for j in range(4):
        stats[6+j] = z * ring[(int(counters[4]) - lags[j]) % len(ring)]
    err = z - reference[4] * ring[(int(counters[4])-1) % len(ring)]
    stats[10], stats[11] = err, err*err
    bin_id = 0
    for threshold in thresholds:
        bin_id += int(z > threshold)
    for j in range(5):
        stats[12+j] = 1.0 if bin_id == j else 0.0
    evidence = 0.0
    for s in range(3):
        alpha = alphas[s]
        for j in range(17):
            ew[s,j] += alpha * (stats[j] - ew[s,j])
        counters[7+s] = (1-alpha)*counters[7+s] + 1.0
        root_n = np.sqrt(counters[7+s])
        offset = 6 + 15*s
        mean_z = (ew[s,0]-baseline[0])*root_n
        log_second = np.log(max(ew[s,1],moment_floor)/max(baseline[1],moment_floor))
        log_var = np.log(max(ew[s,1]-ew[s,0]**2,moment_floor)/max(baseline[1]-baseline[0]**2,moment_floor))
        output[offset] = mean_z
        output[offset+1] = abs(mean_z)
        output[offset+2] = log_second
        output[offset+3] = log_var
        output[offset+4] = (ew[s,2]-baseline[2])*root_n
        for j in range(3):
            rate = baseline[3+j]
            output[offset+5+j] = (ew[s,3+j]-rate)*root_n/np.sqrt(max(rate*(1-rate),rate_floor))
        output[offset+8] = np.sum(np.abs(ew[s,12:17]-baseline[12:17]))*root_n
        for j in range(4):
            output[offset+9+j] = (ew[s,6+j]-baseline[6+j])*root_n
        output[offset+13] = (ew[s,10]-baseline[10])*root_n/np.sqrt(max(baseline[11],moment_floor))
        output[offset+14] = np.log(max(ew[s,11],moment_floor)/max(baseline[11],moment_floor))
        # Explicit float64 avoids NumPy 2 scalar promotion (float32 / Python int)
        # differing from the compiled kernel at a float32 rounding boundary.
        evidence = max(evidence, abs(mean_z)/3, abs(log_var), abs(np.float64(output[offset+9]))/3.0)
    output[51] = ew[0,0] - ew[2,0]
    output[52] = np.log(max(ew[0,1],moment_floor)/max(ew[2,1],moment_floor))
    output[53] = ew[0,6] - ew[2,6]
    counters[5] = max(counters[5], evidence)
    counters[6] = max(counters[6]*decay, evidence)
    output[54], output[55], output[56] = counters[5], counters[6], 1.0 if missing else 0.0
    ring[int(counters[4]) % len(ring)] = z
    counters[4] += 1

class StreamingFeatureState:
    """Output is a reusable float32 buffer; copy it if retaining a row.

    Nonfinite historical values are replaced by finite historical median; a
    nonfinite online point is replaced by historical mean and flagged. Finite
    values clip at 20 historical SD to bound outlier leverage. Historical-only
    scale floor avoids division by zero. This is an engineering policy, not a
    claim that missingness occurs in the official data.
    """
    def __init__(self, historical, *, config=CONFIG):
        self.config = config
        raw = np.asarray(historical, dtype=np.float64)
        if raw.ndim != 1:
            raise ValueError("Historical must be one-dimensional")
        finite = raw[np.isfinite(raw)]
        # Bound pathological finite floats before moment arithmetic can overflow.
        finite = np.clip(finite, -config.finite_value_bound, config.finite_value_bound)
        median = float(np.median(finite)) if finite.size else 0.0
        h = np.clip(np.where(np.isfinite(raw), raw, median), -config.finite_value_bound, config.finite_value_bound)
        if not h.size:
            h = np.array([0.0])
        mean = float(h.mean())
        std = max(float(h.std(ddof=1)) if h.size > 1 else 1.0, config.scale_floor)
        mad = max(float(np.median(np.abs(h-median)))*config.mad_consistency_factor, std*config.mad_min_std_fraction, config.scale_floor)
        if config.normalization == 'median_mad':
            mean, std = median, mad
        z = np.clip((h-mean)/std, -config.clip_z, config.clip_z)
        thresholds = np.quantile(z, config.quantile_probabilities)
        baseline = np.zeros(17, dtype=np.float64)
        baseline[:6] = [z.mean(), np.mean(z*z), np.mean(abs(z)), np.mean(abs(z)>config.tail_thresholds[0]), np.mean(abs(z)>config.tail_thresholds[1]), np.mean(z>0)]
        for j, lag in enumerate(config.lags):
            baseline[6+j] = np.mean(z[lag:]*z[:-lag]) if len(z)>lag else 0.0
        phi = np.clip(baseline[6]/max(baseline[1],config.moment_floor), -config.ar_phi_limit,config.ar_phi_limit)
        errors = z[1:]-phi*z[:-1] if len(z)>1 else z
        baseline[10:12] = [errors.mean(), np.mean(errors*errors)]
        bins = np.searchsorted(thresholds,z,side="left")
        baseline[12:17] = np.bincount(bins,minlength=5)/len(z)
        self.reference = np.array([mean,std,median,mad,phi],dtype=np.float64)
        self.baseline = baseline
        self.thresholds = thresholds
        ring_size = max(config.lags)
        self.ring = np.zeros(ring_size,dtype=np.float64)
        self.ring[-min(ring_size,len(z)):] = z[-ring_size:]
        self.ew = np.tile(baseline,(3,1))
        self.counters = np.zeros(10,dtype=np.float64)
        self.counters[4] = ring_size
        self.output = np.zeros(len(FEATURE_NAMES),dtype=np.float32)
        self.parameters = np.array([config.clip_z,config.cusum_drift,config.memory_decay,config.moment_floor,config.rate_variance_floor,*config.tail_thresholds],dtype=np.float64)
        self.alphas = np.asarray(config.alphas,dtype=np.float64)
        self.lags = np.asarray(config.lags,dtype=np.int64)

    def update_and_get(self, point):
        _update(float(point), self.reference, self.baseline, self.thresholds,
                self.ring, self.ew, self.counters, self.output, self.parameters, self.alphas, self.lags)
        return self.output

    @property
    def state_array_bytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))

def replay(historical, online, *, config=CONFIG):
    state = StreamingFeatureState(historical, config=config)
    for point in online:
        yield state.update_and_get(point).copy()

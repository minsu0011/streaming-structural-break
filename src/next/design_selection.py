"""Explicit frozen-family selection for the unchanged ARCH8 reference base."""
from pathlib import Path
import hashlib
import numpy as np
from src.features.config import CONFIG
from src.features.streaming import feature_names, FEATURE_FAMILIES


def selection_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def select_reference_families(design, names, families='ABCD'):
    if families != 'ABCD':
        raise ValueError('Only the original frozen ABCD base is admitted')
    config = CONFIG.variant(normalization='median_mad',scales=(5,20,160))
    group = dict(zip(['arch8__'+name for name in feature_names(config)],FEATURE_FAMILIES))
    columns = [j for j,name in enumerate(names) if not name.startswith('arch8__') or group[name] in families]
    selected = [names[j] for j in columns]
    if sum(name.startswith('arch8__') for name in selected) != 51:
        raise RuntimeError('Frozen ARCH8 ABCD base must have 51 model columns')
    return np.asarray(design[:,columns],dtype=np.float32),selected

"""Preserve individual historical-normalized AR score directions for trees."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.score_evidence import ARScoreState, ScoreConfig, SCORE_NAMES, score_step, implementation_hash as base_hash
from src.next.engine import ALPHAS


@njit(cache=True)
def coordinate_step(point, base_args, output):
    base = score_step(point, base_args)
    output[:len(base)] = base
    ewma, t = base_args[7], base_args[11][0]
    p = ewma.shape[1]
    cursor = len(base)
    for half in range(3):
        alpha = ALPHAS[half]
        scale = np.sqrt(max(alpha / (2. - alpha) * (1. - (1. - alpha)**(2*t)), 1e-12))
        for j in range(p):
            output[cursor] = ewma[half, j] / scale
            cursor += 1
    return output


@njit(cache=True)
def replay_coordinates(points, base_args, output):
    result = np.empty((len(points), len(output)), dtype=np.float32)
    for i in range(len(points)):
        result[i] = coordinate_step(points[i], base_args, output)
    return result


class ScoreCoordinateState:
    def __init__(self, historical, config=ScoreConfig()):
        self.base = ARScoreState(historical, config)
        self.output = np.empty(len(SCORE_NAMES) + 3*config.order, dtype=np.float32)

    def update(self, point):
        return coordinate_step(float(point), self.base.args, self.output)

    def replay(self, points):
        return replay_coordinates(np.asarray(points, dtype=np.float32), self.base.args, self.output)

    @property
    def state_array_bytes(self):
        return self.base.state_array_bytes + self.output.nbytes


def feature_names(settings):
    config = ScoreConfig(**settings)
    return SCORE_NAMES + tuple(f'score_coordinate_lag{j+1}_{half}' for half in (8,32,128) for j in range(config.order))


def feature_groups(settings):
    return ('SCORE',) * len(SCORE_NAMES) + ('COORDINATE',) * (3*ScoreConfig(**settings).order)


def make_state(historical, settings):
    return ScoreCoordinateState(historical, ScoreConfig(**settings))


def implementation_hash(settings):
    return hashlib.sha256(Path(__file__).read_bytes() + base_hash(settings).encode() + json.dumps(asdict(ScoreConfig(**settings)), sort_keys=True).encode()).hexdigest()

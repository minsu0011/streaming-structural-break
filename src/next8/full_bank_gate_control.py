"""One artificial153-coordinate joint screen; no competition data or tuning."""
from contextlib import contextmanager
import hashlib
import numpy as np
from src.next8.data import FitScope
from src.next8.features import DETAILED
from src.next8.gate_reachability import controlled_cache, coordinate, BOUND

SIGNALS = {0: 'MONOTONE', 6: 'FIXED_U', 9: 'OWN_STATE_REVERSAL'}
TARGET_SEED = 9101913
COORDINATE_SEED = 9101941


def coordinate_values(j, labels):
    assert 0 <= j < 153
    sequence = np.random.SeedSequence(COORDINATE_SEED).spawn(153)[j]
    noise_rng, sign_rng = [np.random.default_rng(s) for s in sequence.spawn(2)]
    series, steps = labels.shape
    # Time-first drawing makes a shorter simulation preserve the same prefix.
    noise = noise_rng.normal(size=(steps, series)).T
    signs = np.where(sign_rng.integers(0, 2, size=(steps, series)).T, 1, -1)
    return coordinate(noise, labels, signs, SIGNALS.get(j, 'NULL'))


def full_bank_fixture(*, series=5120, length=512):
    fixture = controlled_cache('NULL', 0., TARGET_SEED, series=series, length=length)
    cache = fixture.cache
    ids = np.arange(800000, 800000 + series, dtype=np.int64)
    cache.series['dataset_id'] = ids
    cache.meta['dataset_id'] = np.repeat(ids, length)
    labels = np.column_stack((np.zeros((series, 16), dtype=np.uint8),
                              cache.meta['target'].reshape(series, length)))
    values = np.empty((series * length, 153), dtype=np.float32, order='F')
    seeds = np.empty((series, 16, 153), dtype=np.float32)
    hashes = {}
    for j in range(153):
        column = coordinate_values(j, labels)
        values[:, j] = column[:, 16:].reshape(-1)
        seeds[:, :, j] = column[:, :16]
        hashes[str(j)] = hashlib.sha256(column.tobytes()).hexdigest()
    values.flags.writeable = seeds.flags.writeable = False
    cache.values = values
    cache.series.update(seed_values=seeds, low=np.full((series, 153), -BOUND),
                        high=np.full((series, 153), BOUND), valid=np.ones((series, 153), dtype=bool))
    sentinel = frozenset((999999999,))
    name = 'ARTIFICIAL_FULL153_JOINT_SCREEN'
    fixture.parent = FitScope(frozenset(map(int, ids)), sentinel, sentinel, name)
    fixture.scopes = []
    for k in range(3):
        valid = frozenset(map(int, ids[fixture.assignment == k]))
        fixture.scopes.append(FitScope(fixture.parent.training_ids - valid, valid, sentinel, name + '_inner' + str(k)))
    fixture.coordinate_sha256 = hashes
    return fixture


@contextmanager
def full_bank_names():
    import src.next8.screening as screening
    before = screening.NAMES, screening.DETAILED
    screening.NAMES = tuple('ARTIFICIAL_F' + f'{j:03d}_' + SIGNALS.get(j, 'NULL') for j in range(153))
    screening.DETAILED = DETAILED
    try:
        yield screening
    finally:
        screening.NAMES, screening.DETAILED = before

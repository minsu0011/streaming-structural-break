"""Matched synthetic raw laws through the unchanged full H feature engine."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from src.next8.features import FeatureState, N_FEATURES, NORMALIZATION, source_hash
from src.next8.cache import FeatureCache, immutable_npz
from src.next8.data import FitScope
from src.next8.common import atomic, read_json, digest, utc_now, activity

BANKS = ('NULL', 'MEAN_PLUS2', 'SCALE_SD2')
SEED = 9102057
H_LENGTHS = (1000, 1500, 2000, 3000, 4500)


def sample_plan(series=5120):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))
    positive = rng.permutation(series)[:series // 2]
    tau = np.full(series, 513, dtype=np.int32)
    early = len(positive) // 10
    tau[positive[:early]] = 0
    tau[positive[early:]] = rng.integers(1, 400, len(positive) - early)
    h = np.asarray(H_LENGTHS, dtype=np.int32)[rng.permutation(series) % 5]
    assignment = np.empty(series, dtype=np.int8)
    assignment[rng.permutation(series)] = np.arange(series) % 3
    return {'dataset_id': np.arange(900000, 900000 + series, dtype=np.int64),
            'H_length': h, 'tau': tau, 'inner_assignment': assignment}


def raw_case(index, H_length, tau, *, length=512):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, int(index)]))
    historical = rng.normal(size=int(H_length)).astype(np.float32)
    online = rng.normal(size=length).astype(np.float32)
    mask = np.arange(length) >= tau
    mean = online.copy()
    mean[mask] += np.float32(2.)
    scale = online.copy()
    scale[mask] *= np.float32(2.)
    return historical, dict(zip(BANKS, (online, mean, scale)))


def extract_case(historical, variants):
    initial = FeatureState(historical)
    assert initial.normalizer_fit_sha256 == initial.calibration_reference_sha256 == initial.online_reference_sha256
    output = {}
    for bank in BANKS:
        state = FeatureState.restore(initial)
        output[bank] = state.replay(variants[bank])[0]
    return initial, output


def generate_cache(directory, *, series=5120, length=512):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assert not any(directory.iterdir()), 'Partial cache retained; no silent replay'
    plan = sample_plan(series)
    values = {b: np.lib.format.open_memmap(directory / (b + '.npy'), mode='w+', dtype=np.float32,
               shape=(series * length, N_FEATURES)) for b in BANKS}
    fields = {'seed_values': np.empty((series, 16, N_FEATURES), dtype=np.float32),
        'low': np.empty((series, N_FEATURES)), 'high': np.empty((series, N_FEATURES)),
        'valid': np.empty((series, N_FEATURES), dtype=bool), 'epsilon': np.empty((series, 3, N_FEATURES)),
        'context': np.empty((series, 6), dtype=np.float32), 'center': np.empty((series, N_FEATURES)),
        'spread': np.empty((series, N_FEATURES))}
    clip_counts = {b: np.zeros(N_FEATURES, dtype=np.int64) for b in BANKS}
    raw_hashes, reference_hashes = [], []
    for start in range(0, series, 64):
        with activity('Known raw H-engine control series' + str(start), 'SYNTHETIC_GATE_CONTROL', 'B7_NORMALIZATION_AND_LINEAGE'):
            for i in range(start, min(start + 64, series)):
                historical, variants = raw_case(i, plan['H_length'][i], plan['tau'][i], length=length)
                initial, feature = extract_case(historical, variants)
                raw_hashes.append({'group_index': i, 'H_sha256': hashlib.sha256(historical.tobytes()).hexdigest(),
                    'Online_sha256': {b: hashlib.sha256(variants[b].tobytes()).hexdigest() for b in BANKS}})
                reference_hashes.append(initial.normalizer_fit_sha256)
                for name in fields:
                    fields[name][i] = initial.history if name == 'seed_values' else getattr(initial, name)
                for bank in BANKS:
                    values[bank][i * length:(i + 1) * length] = feature[bank]
                    clip_counts[bank] += (np.abs(feature[bank]) >= 12).sum(axis=0)
    for array in values.values():
        array.flush()
    values.clear()
    del array
    target = (np.arange(length)[None, :] >= plan['tau'][:, None]).astype(np.uint8)
    immutable_npz(directory / 'SERIES.npz', **fields, **plan, target=target)
    atomic(directory / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(), 'series': series,
        'online_length': length, 'banks': list(BANKS), 'normalization': NORMALIZATION,
        'feature_source_sha256': source_hash(), 'files_sha256': {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
        'raw_hashes': raw_hashes, 'H_reference_sha256': reference_hashes,
        'clip_counts': {b: x.tolist() for b, x in clip_counts.items()},
        'invalid_H_counts_per_feature': np.sum(~fields['valid'], axis=0).tolist(),
        'matched_H_fit_reused_across_banks': True, 'real_RP_values_used': False})


def load_fixture(directory, bank):
    directory = Path(directory)
    assert bank in BANKS
    done = read_json(directory / 'DONE.json')
    assert done['status'] == 'COMPLETE' and not done['real_RP_values_used']
    assert done['feature_source_sha256'] == source_hash()
    for name in ('SERIES.npz', bank + '.npy'):
        assert digest(directory / name) == done['files_sha256'][name]
    with np.load(directory / 'SERIES.npz', allow_pickle=False) as stream:
        data = {name: stream[name] for name in stream.files}
    cache = FeatureCache.__new__(FeatureCache)
    series, length = done['series'], done['online_length']
    ids = data['dataset_id']
    cache.values = np.load(directory / (bank + '.npy'), mmap_mode='r', allow_pickle=False)
    cache.series_index = np.repeat(np.arange(series, dtype=np.int32), length)
    cache.series = {name: value for name, value in data.items() if name not in ('target', 'tau', 'inner_assignment')}
    cache.meta = {'dataset_id': np.repeat(ids, length), 'time_online': np.tile(np.arange(length, dtype=np.int32), series),
                  'target': data['target'].reshape(-1)}
    sentinel = frozenset((999999999,))
    parent = FitScope(frozenset(map(int, ids)), sentinel, sentinel, 'RAW_ENGINE_CONTROL_' + bank)
    scopes = []
    for k in range(3):
        valid = frozenset(map(int, ids[data['inner_assignment'] == k]))
        scopes.append(FitScope(parent.training_ids - valid, valid, sentinel, parent.name + '_inner' + str(k)))
    return SimpleNamespace(cache=cache, parent=parent, scopes=scopes)

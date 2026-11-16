"""Fixed fitted-fold classifier stress helpers, isolated from all old writers."""
import csv
import hashlib
import joblib
import numpy as np
from numba import njit
from src.next7.common import ROOT, BASE, read_json, digest, ids_hash
from src.next5.model import FeatureModel
from src.next5.runtime import SharedResearchBaseline, baseline_step
from src.next6.runtime import Prepared as PreparedM3
from src.next7.model import Correction
from src.next7.fast_state import FastQualifiedState
from src.next7.features import from_state
from src.next7.null_growth import qualified_features
from src.next7.calibration import load_calibration, cutoff
from src.next6_extension.synthetic import metrics, first_alarm

METHODS = ('M0', 'M3', 'AMPLITUDE_0_9_M0', 'M_EVENT', 'G2_EVENT',
    'I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL')
METRICS = ('threshold', 'observed_break', 'first_alarm', 'any_alarm', 'prebreak_alarm',
    'postbreak_first_detection', 'postbreak_any_detection', 'first_detection_delay',
    'first_postbreak_alarm_delay', 'score_max', 'paired_prechange_max_error', 'tau',
    'recovery_delay', 'recovery_censored', 'recovery_tolerance', 'paired_effect_after_128_max',
    'paired_excess_alarm_after_128', 'control_any_alarm', 'control_first_alarm',
    'treatment_alarm_control_never', 'restricted_first_detection_delay', 'amplitude_control_shared_M0_cut_any_alarm')
BOOL_FIELDS = ('observed_break', 'any_alarm', 'prebreak_alarm', 'postbreak_first_detection',
    'postbreak_any_detection', 'recovery_censored', 'paired_excess_alarm_after_128',
    'control_any_alarm', 'treatment_alarm_control_never', 'amplitude_control_shared_M0_cut_any_alarm')


@njit(cache=False)
def baseline_replay(points, args):
    out = np.empty(len(points))
    for t in range(len(points)):
        out[t] = baseline_step(float(points[t]), args)
    return out


def load_models(guard, *, include_m3=False):
    result, provenance = [], []
    for fold in range(5):
        contract = read_json(BASE / f'nested_base/fold_{fold}/CONTRACT.json')
        # NEXT5 frozen row metadata stores dataset_id as int32; byte hashing is
        # dtype-sensitive even though the integer membership is identical.
        train = np.array(contract['training_ids'], dtype=np.int32)
        assert train.tolist() == contract['training_ids']
        valid = contract['outer_validation_ids']
        guard.partition(train, valid)
        assert all(guard.split.assignment(int(i))[1] != fold for i in train)
        assert all(guard.split.assignment(int(i))[1] == fold for i in valid)
        refs, hashes = [], {}
        for kind in ('J102', 'C205'):
            parent = ROOT / f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}'
            receipt = read_json(parent / 'RESULTS.json')
            assert digest(parent / 'model.joblib') == receipt['model_sha256']
            model = FeatureModel.load(parent / 'model.joblib')
            assert model.training['training_ids_sha256'] == hashlib.sha256(np.sort(train).tobytes()).hexdigest()
            refs.append(model)
            hashes[kind] = receipt['model_sha256']
        heads = {}
        for method, relative in [('I4_EVENT_SCALE_TREE', f'g1/fold_{fold}/I4_EVENT_SCALE_TREE'),
                                 ('G2_FULL', f'g2/fold_{fold}/G2_FULL'),
                                 ('G3_PAIRWISE_FULL', f'g3/fold_{fold}')]:
            parent = BASE / relative
            receipt = read_json(parent / 'DONE.json')
            assert digest(parent / 'model.joblib') == receipt['model_sha256']
            heads[method] = Correction.load(parent / 'model.joblib')
            assert heads[method].training['training_ids_sha256'] == ids_hash(train)
            hashes[method] = receipt['model_sha256']
        parent = ROOT / f'artifacts/next6/fold_{fold}/outer/M3'
        receipt = read_json(parent / 'DONE.json')
        assert receipt['training_ids_sha256'] == ids_hash(train)
        assert digest(parent / 'model.joblib') == receipt['model_sha256']
        hashes['M3'] = receipt['model_sha256']
        m3 = PreparedM3(joblib.load(parent / 'model.joblib'), resident=False) if include_m3 else None
        result.append({'baseline': SharedResearchBaseline(*refs), 'heads': heads, 'M3': m3})
        provenance.append({'fold': fold, 'training_ids_sha256': ids_hash(train), 'model_sha256': hashes})
    return result, provenance


def predict_all(models, h, o, *, include_m3=False):
    cal = load_calibration()
    state = FastQualifiedState(h, core_cut=cal['core_cut'], scale_cut=cutoff(cal, len(h))).replay(o)
    raw, qualified = from_state(state[:, :30]), qualified_features(state)
    output = np.full((5, len(o), len(METHODS)), np.nan)
    for fold, model in enumerate(models):
        baseline = baseline_replay(o, model['baseline'].arguments(h))
        output[fold, :, 0] = baseline
        if include_m3:
            output[fold, :, 1] = model['M3'].new_state(h).replay(o)
        output[fold, :, 2] = .9 * baseline
        output[fold, :, 3] = .9 * baseline + .1 * raw[:, 1]
        output[fold, :, 4] = .9 * baseline + .1 * qualified[:, 1]
        for position, method in enumerate(METHODS[5:], start=5):
            features = raw if method == 'I4_EVENT_SCALE_TREE' else qualified
            output[fold, :, position] = model['heads'][method].predict(baseline, features)
    selected = output if include_m3 else output[:, :, [0, 2, 3, 4, 5, 6, 7]]
    assert np.isfinite(selected).all() and np.all((selected >= 0) & (selected <= 1))
    return output


def old_case_rows(case, protocol):
    records = {}
    for fold in range(5):
        path = ROOT / f'artifacts/next6_extension/extended_synthetic/fold_{fold}/{case}_SERIES.csv'
        assert digest(path) == protocol['old_evaluation_receipts'][f'{fold}/{case}']
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                records[fold, int(row['seed']), row['method'], int(row['horizon'])] = row
    return records


def numeric_row(record, horizon):
    values = np.full(len(METRICS), np.nan)
    for index, name in enumerate(METRICS):
        value = record.get(name)
        if value is None or value == '':
            continue
        if isinstance(value, str) and name in BOOL_FIELDS:
            assert value in ('True', 'False')
            value = value == 'True'
        values[index] = float(value)
    if values[METRICS.index('observed_break')] > .5:
        tau, first = values[METRICS.index('tau')], values[METRICS.index('first_alarm')]
        values[METRICS.index('restricted_first_detection_delay')] = first - tau if first >= tau else horizon - tau
    return values


def curve_metrics(p, c, cut, tau, info, horizon, *, amplitude_M0_cut=None):
    record = metrics(p, c, threshold=cut, tau=tau, info=info, horizon=horizon, units='raw detector probability')
    record['control_first_alarm'] = first_alarm(c[:horizon], cut)
    record['control_any_alarm'] = record['control_first_alarm'] >= 0
    record['treatment_alarm_control_never'] = record['any_alarm'] and not record['control_any_alarm']
    if amplitude_M0_cut is not None:
        record['amplitude_control_shared_M0_cut_any_alarm'] = bool(np.any(p[:horizon] > amplitude_M0_cut))
    return numeric_row(record, horizon)


def assert_old_parity(actual, old, horizon):
    expected = numeric_row(old, horizon)
    known = np.isfinite(expected)
    np.testing.assert_allclose(actual[known], expected[known], rtol=0., atol=1e-12)
    # Undefined conditional detection/recovery fields must remain undefined.
    for name in ('first_detection_delay', 'first_postbreak_alarm_delay', 'recovery_delay'):
        j = METRICS.index(name)
        assert np.isnan(actual[j]) == np.isnan(expected[j])

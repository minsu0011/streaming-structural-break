"""Benchmark-only loading and checkpoint adapters for unchanged scalar APIs."""
import copy
import numpy as np
from src.next7.common import ROOT, BASE, read_json, digest


CLASSIFIERS = ('M0', 'M3', 'M_EVENT', 'G2_EVENT', 'I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL')
STATES = ('FRESH_CORE20', 'OBSERVABLE_FUSED27', 'G1_READABLE_STATE30', 'G2_FUSED_STATE45')


class Factory:
    def __init__(self, method):
        import joblib
        self.method = method
        self.calibration = None
        self.prepared = None
        if method in CLASSIFIERS and method != 'M3':
            from src.next5.model import FeatureModel
            from src.next5.runtime import SharedResearchBaseline
            from src.next7.integrated_runtime import Prepared
            from src.next7.model import Correction
            directory = BASE / 'research_bundles/fold_0'
            manifest = read_json(directory / 'MANIFEST.json')
            assert manifest['role'] == 'RESEARCH_FOLD_ONLY_NOT_PROMOTED' and manifest['default_method'] == 'M0'
            assert manifest['fold'] == 0
            names = ['J102.joblib', 'C205.joblib']
            if method not in ('M0', 'M_EVENT', 'G2_EVENT'):
                names.append(method + '.joblib')
            names.append('SCALE_CALIBRATION.json')
            for name in names:
                assert digest(directory / name) == manifest['files_sha256'][name]
            for name, sha in manifest['source_sha256'].items():
                assert digest(ROOT / name) == sha
            references = [FeatureModel.load(directory / name) for name in names[:2]]
            assert all(m.training['training_ids_sha256'] == manifest['base_training_int32_sha256'] for m in references)
            self.calibration = read_json(directory / 'SCALE_CALIBRATION.json')
            if method == 'M0':
                self.prepared = SharedResearchBaseline(*references)
            else:
                head = None if method in ('M_EVENT', 'G2_EVENT') else Correction.load(directory / (method + '.joblib'))
                if head is not None:
                    assert head.training['fold'] == 0 and head.training['training_ids_sha256'] == manifest['training_ids_sha256']
                self.prepared = Prepared(*references, head, self.calibration,
                    qualified=method in ('G2_EVENT', 'G2_FULL', 'G3_PAIRWISE_FULL'))
            return
        if method == 'M3':
            from src.next6.runtime import Prepared
            directory = ROOT / 'artifacts/next6/fold_0/outer/M3'
            assert digest(directory / 'model.joblib') == read_json(directory / 'DONE.json')['model_sha256']
            self.prepared = Prepared(joblib.load(directory / 'model.joblib'))
            return
        if method in ('FRESH_CORE20', 'OBSERVABLE_FUSED27'):
            from src.next6_extension.observer import load
            from src.next6_extension.validated_runtime import Prepared
            self.observable = load(ROOT / 'artifacts/next6_extension/observable_models/fold_0/model.joblib')
            if method == 'OBSERVABLE_FUSED27':
                self.prepared = Prepared(self.observable)
            return
        if method in ('G1_READABLE_STATE30', 'G2_FUSED_STATE45'):
            from src.next7.calibration import load_calibration
            self.calibration = load_calibration()
            return
        raise ValueError('Unknown frozen benchmark method')

    def new_state(self, historical):
        if self.prepared is not None:
            return self.prepared.new_state(historical)
        if self.method == 'FRESH_CORE20':
            from src.next6_extension.fresh_runtime import SafeState
            return SafeState(historical, threshold=self.observable.threshold, graph=self.observable.graph)
        from src.next7.calibration import cutoff
        from src.next7.event import CombinedState
        from src.next7.fast_state import FastQualifiedState
        cls = CombinedState if self.method == 'G1_READABLE_STATE30' else FastQualifiedState
        return cls(historical, core_cut=self.calibration['core_cut'], scale_cut=cutoff(self.calibration, len(historical)))


def memory(method, state):
    from src.next.fused_runtime import unique_array_bytes
    if method == 'M0':
        return int(state.state_array_bytes), None
    if method == 'M3':
        return int(unique_array_bytes(state.args[:12] + (state.args[16], state.args[17], state.args[18], state.args[20]))), None
    if method in CLASSIFIERS:
        return int(state.instance_array_bytes_excluding_models), int(state.added_bounded_mutable_bytes)
    if method in ('FRESH_CORE20', 'OBSERVABLE_FUSED27'):
        feature = state if method == 'FRESH_CORE20' else state.features
        arrays = feature.args[:11] + (feature.args[14], feature.H_block_z, feature.H_previous_state)
        if method == 'OBSERVABLE_FUSED27':
            arrays += (state.args[3], state.args[4])
        return int(unique_array_bytes(arrays)), int(state.bounded_mutable_state_bytes)
    raw = state if method == 'G1_READABLE_STATE30' else state.reference.raw
    arrays = (raw.core.args[:11], raw.core.args[14], raw.core.H_block_z, raw.core.H_previous_state,
        raw.scale.args, raw.args)
    if method == 'G2_FUSED_STATE45':
        arrays += (state.reference.args,)
    return int(unique_array_bytes(arrays)), int(state.bounded_mutable_state_bytes)


def checkpoint(method, state):
    if method not in CLASSIFIERS:
        raise ValueError('Only classifier throughput has persisted checkpoints')
    if method in ('M0', 'M3'):
        return {'method': method, 'args': copy.deepcopy(state.args)}
    snapshot = state.__getstate__().copy()
    snapshot.pop('prepared')
    return {'method': method, 'attributes': copy.deepcopy(snapshot)}


def restore(factory, checkpoint):
    method = factory.method
    if checkpoint['method'] != method:
        raise ValueError('Checkpoint recipe mismatch')
    if method == 'M0':
        from src.next5.runtime import ScalarDetector, baseline_step
        return ScalarDetector(baseline_step, checkpoint['args'], factory.prepared)
    if method == 'M3':
        from numba import typeof
        from src.next5.runtime import resident_class
        from src.next6.runtime import State, kernel
        state = State.__new__(State)
        state.args = checkpoint['args']
        state.resident = resident_class(kernel, typeof(state.args))(state.args)
        return state
    from src.next7.integrated_runtime import IntegratedState
    state = IntegratedState.__new__(IntegratedState)
    state.__setstate__({**checkpoint['attributes'], 'prepared': factory.prepared})
    return state

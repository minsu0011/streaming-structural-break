"""Direct current-point runtime for the unpromoted B2 diagnostic model."""
from pathlib import Path
import hashlib
import numpy as np
from src.next4.artifacts import ROOT,read_json,digest
from src.next4.model import FeatureModel
from src.next2.blend import combine
from src.next2.dependence_runtime import PreparedDependencePredictor
from src.next.fused_runtime import unique_array_bytes


class PreparedB2:
    def __init__(self,model,complement):
        model.validate()
        if model.module_name!='src.next4.scale_asymmetry' or model.settings!={'version':'separate_energy_branches_v1','kind':'log_semitail'}:
            raise ValueError('Exact unchanged B2 diagnostic model required')
        self.model=model;self.complement=PreparedDependencePredictor(complement,resident=False)

    def new_state(self,historical):return B2State(self,historical)


class B2State:
    def __init__(self,prepared,historical):
        self.prepared=prepared;self.primary=prepared.model.make_state(historical)
        self.complement=prepared.complement.new_state(historical)

    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value):raise ValueError('Finite current observation required')
        left=self.prepared.model.predict_one(self.primary.update(value))
        right=self.complement.predict_one(value)
        return float(combine(left,right,.8))

    @property
    def state_array_bytes(self):
        return unique_array_bytes((self.primary.args,self.complement.call_args[:-1]))


def load_prepared(directory):
    from src.next2.shared_inference import load_artifact
    directory=Path(directory);receipt=read_json(directory/'RESULTS.json')
    if receipt.get('status')!='PASS' or receipt.get('candidate')!='B2_LOG_ENERGY_SEMITAIL' or receipt.get('promotion') is not False:
        raise ValueError('Explicit B2 diagnostic all-DEV receipt required')
    if digest(directory/'model.joblib')!=receipt['files_sha256']['model.joblib']:raise RuntimeError('B2 model bytes changed')
    model=FeatureModel.load(directory/'model.joblib')
    frozen,_=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    return PreparedB2(model,frozen.complement)


def iterate_current(datasets,prepared_factory):
    yield None
    prepared=prepared_factory()
    for historical,online in datasets:
        state=prepared.new_state(historical)
        for point in online:yield state.predict_one(point)


def infer(datasets,model_directory_path):
    yield from iterate_current(datasets,lambda:load_prepared(model_directory_path))


def source_hash():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

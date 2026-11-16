"""Append four negative-direction summaries while preserving all frozen J inputs."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from numba import njit
from src.next3.stage_trace import directional_bayes
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig,serial_variance_step
from src.next.extensions import ExtendedFeatureState,extension_hash
from src.next.registered_extensions import register_extension
from src.next.predictor import NextBundle

NAMES=tuple('next3_negative_bayes__'+channel+'_'+prior for channel in ('raw_energy','log_variance_forecast') for prior in ('bayes025','bayes1'))


def source_hash():
    here=Path(__file__)
    return hashlib.sha256(here.read_bytes()+here.with_name('stage_trace.py').read_bytes()).hexdigest()


@njit(cache=False)
def negative_values(evidence_args,age_variances):
    split=directional_bayes(evidence_args,age_variances)
    result=np.empty(4,dtype=np.float32)
    for channel in range(2):
        for prior in range(2):result[channel*2+prior]=max(split[channel*4+prior],-80.)
    return result


@njit(cache=False)
def replay_negative(points,filter_args,parameters,conditional,mode,calibration,work,evidence_args,age_variances):
    result=np.empty((len(points),4),dtype=np.float32)
    for t in range(len(points)):
        serial_variance_step(float(points[t]),filter_args,parameters,conditional,mode,calibration,work,evidence_args,age_variances)
        result[t]=negative_values(evidence_args,age_variances)
    return result


def negative_feature_replay(historical,online):
    state=SerialVarianceState(historical,SerialVarianceConfig(order=8,normalization='arch1',max_age=512))
    return replay_negative(np.asarray(online,dtype=np.float32),state.filter_args,state.parameters,state.conditional,
        state.mode,state.calibration,state.work,state.evidence_args,state.age_variances)


class NegativeBayesState:
    def __init__(self,historical,names,extension,registration):
        register_extension(extension,registration)
        if tuple(names[-4:])!=NAMES:raise ValueError('Exact four-feature directional contract required')
        self.base=ExtendedFeatureState(historical,tuple(names[:-4]),extension)
        self.output=np.empty(len(names),dtype=np.float32)

    def update(self,point):
        self.output[:-4]=self.base.update(point)
        self.output[-4:]=negative_values(self.base.new.evidence_args,self.base.new.age_variances)
        return self.output


@dataclass
class NegativeBayesBundle(NextBundle):
    primary_extension:dict=None
    primary_registration:dict=None
    primary_extension_sha256:str=''
    negative_bayes_sha256:str=''

    def validate(self):
        register_extension(self.primary_extension,self.primary_registration)
        if self.negative_bayes_sha256!=source_hash() or self.primary_extension_sha256!=extension_hash(self.primary_extension):
            raise RuntimeError('Negative Bayes model feature contract changed')

    def make_state(self,historical):
        self.validate()
        return NegativeBayesState(historical,self.names,self.primary_extension,self.primary_registration)

    @classmethod
    def wrap(cls,model,primary):
        return cls(**vars(model),primary_extension=primary['extension'],primary_registration=primary['extension_registration'],
            primary_extension_sha256=primary['extension_frozen_hash'],negative_bayes_sha256=source_hash())

    @classmethod
    def load(cls,path):
        model=NextBundle.load(path)
        if not isinstance(model,cls):raise ValueError('Wrong scale model type')
        model.validate()
        return model

"""Guarded partial-training experiments; never replaces primary CV authority."""
from pathlib import Path
import hashlib
import numpy as np
from src.next.candidate_io import candidate_design,fit_candidate,model_class,load_full_oof
from src.next.references import REFERENCES,load_reference_fold
from src.next.fitting import weights_from_training
from src.features.config import CONFIG
from src.features.cross_order import feature_names as cross_names
from src.features.streaming import feature_names as old_names
from src.features.historical_calibration import CalibrationPolicy
from src.models.cross_order import fit_cross_order,CrossOrderBlendBundle
from src.models.learners import fit_candidate as fit_old
from src.models.arch_calibrated import ARCHCalibratedBundle
from src.utils.artifacts import sha256


OLD_CONFIG = CONFIG.variant(normalization='median_mad',scales=(5,20,160))


def implementation_hash():
    return sha256(Path(__file__))


def ranked_training_groups(guard,outer_fold,seed,purpose):
    groups = {guard.split.groups[sid] for sid in guard.dev_ids if guard.split.assignment(sid)[1]!=outer_fold}
    return sorted(groups,key=lambda group:hashlib.sha256(f'next-{purpose}:{seed}:{outer_fold}:{group}'.encode('ascii')).digest())


def training_ids_for_groups(guard,outer_fold,groups):
    selected = set(groups)
    ids = [sid for sid in guard.dev_ids if guard.split.assignment(sid)[1]!=outer_fold and guard.split.groups[sid] in selected]
    return guard.admit(ids,purpose='fixed train-only historical-group subset')


def ordered_rows(metadata,ids):
    mask = np.isin(metadata['dataset_id'],ids)
    return np.concatenate([np.flatnonzero(mask & (metadata['fold']==fold)) for fold in range(5)])


class DiagnosticTraining:
    def __init__(self,data,eid):
        self.data,self.eid = data,eid
        self.metadata = data.rows()
        self.candidate = None
        if eid in ('PRIMARY','HIGHEST_MEAN'):
            self.names = cross_names(OLD_CONFIG) if eid=='PRIMARY' else old_names(OLD_CONFIG)
            self.design = np.empty((len(self.metadata['target']),len(self.names)),dtype=np.float32)
            self.prediction = np.empty(len(self.design),dtype=np.float32)
            self.sources = {}
            for fold in range(5):
                frame,prediction,_ = load_reference_fold(data.guard.root,data.guard,eid,fold,features=True)
                rows = np.flatnonzero(self.metadata['fold']==fold)
                for name in ('dataset_id','time_online','target'):
                    np.testing.assert_array_equal(frame[name].to_numpy(),self.metadata[name][rows])
                self.design[rows] = frame.loc[:,self.names].to_numpy(dtype=np.float32)
                self.prediction[rows] = prediction
                directory = data.guard.root/'data/processed/oof'/REFERENCES[eid]
                self.sources[f'fold_{fold}_oof_sha256'] = sha256(directory/f'fold_{fold}.npy')
            self.sources['reference_oof_manifest_sha256'] = sha256(directory/'MANIFEST.json')
        else:
            self.candidate,self.original,metadata,self.prediction,self.sources = load_full_oof(data,eid)
            self.design,self.metadata,self.names,feature_sources,rows = candidate_design(data,self.candidate)
            np.testing.assert_array_equal(rows,np.arange(len(rows)))
            self.sources = {'oof':self.sources,'features':feature_sources}
            for key in metadata:
                np.testing.assert_array_equal(metadata[key],self.metadata[key])

    def fit(self,train,valid):
        data,metadata = self.data,self.metadata
        data.guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
        if self.candidate is not None:
            return fit_candidate(data,self.candidate,self.design,metadata,self.names,train,valid)
        x = np.asarray(self.design[train],dtype=np.float32)
        y = metadata['target'][train]
        weights = weights_from_training(y,metadata['time_online'][train],metadata['dataset_id'][train],'W3')
        if self.eid=='PRIMARY':
            model = fit_cross_order(x,y,sample_weight=weights,config=OLD_CONFIG,families='ABCD',params={})
        else:
            excluded = tuple(np.flatnonzero(np.ptp(x,axis=0)<=1e-12))
            base = fit_old('M4_lightgbm',x,y,families='ABCD',sample_weight=weights,params={},config=OLD_CONFIG,excluded_columns=excluded)
            model = ARCHCalibratedBundle(base,CalibrationPolicy(method='historical_median_mad'),OLD_CONFIG)
        prediction = np.asarray(model.predict(self.design[valid]),dtype=np.float32)
        return model,prediction,{'training_rows':len(train),'validation_rows':len(valid),'weighting':'W3 exactly equal to old S3','constant_removal':'TRAIN only'}

    def load(self,path):
        cls = model_class(self.candidate) if self.candidate is not None else CrossOrderBlendBundle if self.eid=='PRIMARY' else ARCHCalibratedBundle
        return cls.load(path)

    def validate_saved_predictions(self,model,path,valid,prediction):
        model.save(path)
        loaded = self.load(path)
        np.testing.assert_array_equal(loaded.predict(self.design[valid[:4096]]).astype(np.float32),prediction[:4096])
        sid = int(self.metadata['dataset_id'][valid[0]])
        h,o,_ = self.data.get(sid)
        state = loaded.make_state(h)
        update = state.update if self.candidate is not None else state.update_and_get
        stream = np.asarray([loaded.predict_one(update(point)) for point in o],dtype=np.float32)
        np.testing.assert_array_equal(stream,prediction[self.metadata['dataset_id'][valid]==sid])
        return {'serialization_probe_bitwise_equal':True,'one_complete_heldout_series_streaming_bitwise_equal':True}

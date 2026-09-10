"""Small, deterministic supervised candidates and a low-overhead linear predictor."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from src.features.streaming import FEATURE_FAMILIES,feature_version
from src.features.config import CONFIG
from src.models.compact_trees import export_histgb,export_lightgbm

MODEL_NAMES=('M1_logistic','M2_ridge','M3_histgb','M4_lightgbm','M5_xgboost')

def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'compact_trees.py').read_bytes()).hexdigest()

def make_estimator(name,params=None):
    if params:
        estimator=make_estimator(name)
        return estimator.set_params(**params)
    if name=='M1_logistic': return LogisticRegression(C=0.1,max_iter=500,random_state=20260908)
    if name=='M2_ridge': return RidgeClassifier(alpha=100.0)
    if name=='M3_histgb': return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,max_depth=4,min_samples_leaf=30,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20260908)
    if name=='M4_lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100,num_leaves=15,max_depth=4,min_child_samples=30,reg_lambda=10,learning_rate=.05,n_jobs=4,random_state=20260908,deterministic=True,force_col_wise=True,verbosity=-1)
    if name=='M5_xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100,max_depth=3,min_child_weight=30,reg_lambda=10,learning_rate=.05,n_jobs=4,random_state=20260908,tree_method='hist',device='cpu')
    raise ValueError(name)

@dataclass
class ModelBundle:
    name: str
    feature_hash: str
    columns: np.ndarray
    estimator: object
    scaler: object = None
    linear_weight: object = None
    linear_intercept: float = 0.0
    compact: object = None
    predictor_hash: str = ''
    feature_config: object = CONFIG

    def predict(self, features):
        x=np.asarray(features,dtype=np.float32)[:,self.columns]
        if self.linear_weight is not None:
            return expit(x@self.linear_weight+self.linear_intercept)
        if self.compact is not None:
            return self.compact.predict(x)
        with threadpool_limits(limits=1):
            if self.name=='M4_lightgbm':
                return self.estimator.booster_.predict(x,num_threads=1)
            if self.name=='M5_xgboost':
                return self.estimator.get_booster().inplace_predict(x)
            return self.estimator.predict_proba(x)[:,1]

    def predict_one(self, feature):
        x=feature[self.columns]
        if self.linear_weight is not None:
            return float(expit(np.dot(x,self.linear_weight)+self.linear_intercept))
        if self.compact is not None:
            return self.compact.predict_one(x)
        if self.name=='M4_lightgbm':
            return float(self.estimator.booster_.predict(x.reshape(1,-1),num_threads=1)[0])
        if self.name=='M5_xgboost':
            return float(self.estimator.get_booster().inplace_predict(x.reshape(1,-1))[0])
        return float(self.estimator.predict_proba(x.reshape(1,-1))[0,1])

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(self,path)

    @staticmethod
    def load(path):
        bundle=joblib.load(path)
        if bundle.feature_hash!=feature_version(bundle.feature_config):
            raise RuntimeError('Model feature contract hash mismatch; retrain required')
        if bundle.predictor_hash!=implementation_hash():
            raise RuntimeError('Predictor implementation changed; retrain and verify required')
        return bundle

def fit_candidate(name,features,target,*,families='ABCD',sample_weight=None,params=None,config=CONFIG,excluded_columns=()):
    columns=np.array([j for j,f in enumerate(FEATURE_FAMILIES) if f in families and j not in excluded_columns],dtype=np.int32)
    if not len(columns): raise ValueError('Empty feature family selection')
    x=np.asarray(features,dtype=np.float32)[:,columns]
    y=np.asarray(target,dtype=np.uint8)
    if not np.isfinite(x).all() or set(np.unique(y))!={0,1}: raise ValueError('Finite features and both classes required for training')
    estimator=make_estimator(name,params)
    scaler=None; weight=None; intercept=0.0
    with threadpool_limits(limits=4):
        if name in ('M1_logistic','M2_ridge'):
            scaler=StandardScaler().fit(x,sample_weight=sample_weight)
            estimator.fit(scaler.transform(x),y,sample_weight=sample_weight)
            coefficients=np.asarray(estimator.coef_).reshape(-1)
            if coefficients.shape != scaler.scale_.shape:
                raise ValueError('Unexpected binary linear coefficient shape')
            weight=coefficients/scaler.scale_
            intercept=float(np.asarray(estimator.intercept_).reshape(-1)[0]-np.dot(scaler.mean_,weight))
        else:
            estimator.fit(x,y,sample_weight=sample_weight)
    compact=export_histgb(estimator) if name=='M3_histgb' else (export_lightgbm(estimator) if name=='M4_lightgbm' else None)
    if compact is not None:
        # Fitting does not accept an exporter whose predictions differ from native.
        probe=x[:min(2048,len(x))]
        with threadpool_limits(limits=1):
            native=estimator.booster_.predict(probe,num_threads=1) if name=='M4_lightgbm' else estimator.predict_proba(probe)[:,1]
        np.testing.assert_allclose(compact.predict(probe),native,rtol=0,atol=1e-12)
    return ModelBundle(name,feature_version(config),columns,estimator,scaler,weight,intercept,compact,implementation_hash(),config)

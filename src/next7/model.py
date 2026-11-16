"""Small fixed causal corrections to a frozen BL80 probability score."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from src.models.compact_trees import _arrays,tree_margins,tree_margin
from src.next7.features import CURRENT_EVENT,FULL,EVENT_SCALE_TREE


def implementation_hash():
    root=Path(__file__).resolve().parents[2]
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in (
        'src/next7/model.py','src/next7/features.py','src/models/compact_trees.py'))).hexdigest()


def logit(values):
    p=np.clip(np.asarray(values,dtype=np.float64),1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


def export_regression(estimator):
    dump=estimator.booster_.dump_model()
    if dump['objective']!='regression' or dump['average_output'] or dump['num_tree_per_iteration']!=1:
        raise ValueError('Exact single-output L2 regression trees required')
    roots=[];features=[];thresholds=[];left=[];right=[];values=[];missing=[]
    def add(node):
        index=len(features)
        features.append(-1);thresholds.append(0.);left.append(-1);right.append(-1);values.append(0.);missing.append(False)
        if 'leaf_value' in node:
            values[index]=float(node['leaf_value'])
            return index
        if node['decision_type']!='<=' or node['missing_type'] not in ('None','NaN'):
            raise ValueError('Unsupported regression split/missing policy')
        features[index]=int(node['split_feature']);thresholds[index]=float(node['threshold'])
        missing[index]=bool(node['default_left'])
        left[index]=add(node['left_child']);right[index]=add(node['right_child'])
        return index
    for tree in dump['tree_info']:
        roots.append(add(tree['tree_structure']))
    return _arrays(roots,features,thresholds,left,right,values,missing,0.)


@dataclass
class Correction:
    method:str
    columns:tuple
    center:np.ndarray
    spread:np.ndarray
    beta:object
    tree:object
    training:dict
    source_sha256:str

    def validate(self):
        if self.source_sha256!=implementation_hash():
            raise RuntimeError('Fitted NEXT7 correction source changed')

    def predict(self,baseline,features):
        b=np.asarray(baseline,dtype=np.float64)
        x=np.asarray(features,dtype=np.float64)[:,self.columns]
        if self.tree is None:
            z=np.clip((x-self.center)/self.spread,-8.,8.)
            return expit(logit(b)+z@self.beta)
        residual=tree_margins(x.astype(np.float32),*self.tree.args())
        return np.clip(b+residual,1e-6,1-1e-6)

    def predict_one(self,baseline,features):
        x=np.asarray(features,dtype=np.float64)[list(self.columns)]
        if self.tree is None:
            z=np.clip((x-self.center)/self.spread,-8.,8.)
            return float(expit(float(logit(baseline))+z@self.beta))
        return float(np.clip(float(baseline)+tree_margin(x.astype(np.float32),*self.tree.args()),1e-6,1-1e-6))

    def save(self,path):
        path=Path(path)
        if path.exists():
            raise FileExistsError(str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(self,path)

    @classmethod
    def load(cls,path):
        obj=joblib.load(path)
        if not isinstance(obj,cls):
            raise TypeError('NEXT7 correction model required')
        obj.validate()
        return obj


def fit_correction(method,features,baseline,target,weights,provenance,tree_parameters=None):
    columns=CURRENT_EVENT if method=='I3_CURRENT_EVENT' else FULL if method=='M_FULL' else EVENT_SCALE_TREE
    if method not in ('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE'):
        raise ValueError('Registered correction method required')
    if not provenance.get('strict_nested_OOF') or not provenance.get('outer_validation_excluded'):
        raise RuntimeError('LEAKAGE_FAIL: verified nested OOF provenance required')
    x=np.asarray(features,dtype=np.float64)[:,columns]
    y=np.asarray(target,dtype=np.float64)
    b=np.asarray(baseline,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if not (len(x)==len(y)==len(b)==len(w)) or not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError('Finite aligned training inputs required')
    if set(np.unique(y))!={0.,1.} or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError('Binary labels and finite nonnegative weights with positive total required')
    w=w/w.mean()
    training=dict(provenance)
    if method!='I4_EVENT_SCALE_TREE':
        center=x.mean(axis=0)
        spread=np.maximum(x.std(axis=0),1e-4)
        z=np.ascontiguousarray(np.clip((x-center)/spread,-8.,8.))
        offset=logit(b)
        penalty=.01
        def objective(beta):
            eta=offset+z@beta
            loss=float(np.mean(w*(np.logaddexp(0.,eta)-y*eta))+.5*penalty*(beta@beta))
            gradient=(z.T@(w*(expit(eta)-y)))/len(y)+penalty*beta
            return loss,gradient
        with threadpool_limits(limits=1):
            result=minimize(objective,np.zeros(len(columns)),method='L-BFGS-B',jac=True,
                options={'maxiter':300,'ftol':1e-12,'gtol':1e-8})
        if not result.success:
            raise RuntimeError('Fixed correction optimizer did not converge: '+str(result.message))
        training.update({'optimizer_success':bool(result.success),'iterations':int(result.nit),
            'objective':float(result.fun),'gradient_max_abs':float(np.max(np.abs(result.jac))),
            'penalty':penalty,'intercept':False,'baseline_offset_fixed_coefficient':1.})
        return Correction(method,columns,center,spread,result.x,None,training,implementation_hash())
    from lightgbm import LGBMRegressor
    if tree_parameters is None:
        raise ValueError('Preregistered residual tree parameters required')
    allowed=('n_estimators','num_leaves','max_depth','min_child_samples','reg_lambda','learning_rate',
        'random_state','n_jobs','deterministic','force_col_wise','verbosity')
    params={k:tree_parameters[k] for k in allowed}
    estimator=LGBMRegressor(objective='regression',**params)
    design=x.astype(np.float32)
    with threadpool_limits(limits=4):
        estimator.fit(design,y-b,sample_weight=w)
    compact=export_regression(estimator)
    with threadpool_limits(limits=1):
        native=estimator.booster_.predict(design[:4096],num_threads=1)
    error=float(np.max(np.abs(tree_margins(design[:4096],*compact.args())-native)))
    if error>1e-12:
        raise RuntimeError('Residual tree export parity failed')
    training.update({'native_compact_max_error':error,'tree_count':len(compact.roots),
        'target':'y minus inner-OOF BL80','parameters':params})
    return Correction(method,columns,np.zeros(len(columns)),np.ones(len(columns)),None,compact,training,implementation_hash())

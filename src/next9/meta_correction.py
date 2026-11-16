"""One fixed offset-logistic correction fitted only from honest inner OOF."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from src.next9.movement_screen import group_weights


def validate_oof(outer,parts,expected_rows):
    """Each part is a full excluded-group procedure, never an in-sample head."""
    if len(parts)!=3:raise ValueError('Exactly three original inner procedures required')
    rows=[];predictions=[];seen=set()
    for part in parts:
        scope=part['scope'];ids=np.asarray(part['dataset_id'])
        scope.admit_validation(ids)
        outer.admit_training(ids)
        if not scope.training_ids<=outer.training_ids or scope.training_ids & scope.validation_ids:
            raise ValueError('Inner procedure must exclude its own meta rows and outer holdout')
        provenance=part['training_provenance']
        if provenance['training_ids_sha256']!=scope.provenance()['training_ids_sha256']:
            raise ValueError('In-sample or wrong-scope meta model')
        if not part['complete_selection_excluded']:
            raise ValueError('Candidate/state/orientation/transition selection must also exclude meta groups')
        groups=set(map(int,ids))
        if seen & groups:raise ValueError('Meta validation group repeated')
        seen |= groups
        r=np.asarray(part['global_row']);p=np.asarray(part['prediction'])
        if r.shape!=ids.shape or p.shape!=r.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Aligned finite OOF row probabilities required')
        rows.append(r);predictions.append(p)
    if seen!=outer.training_ids:raise ValueError('OOF must cover every outer TRAIN group')
    rows=np.concatenate(rows);predictions=np.concatenate(predictions)
    order=np.argsort(rows);rows=rows[order];predictions=predictions[order]
    if not np.array_equal(rows,np.asarray(expected_rows)):
        raise ValueError('Missing, duplicate, or unexpected meta OOF rows')
    return predictions


def fixed_diagnostic(baseline,model):
    b,m=np.asarray(baseline,dtype=float),np.asarray(model,dtype=float)
    if b.shape!=m.shape or not np.isfinite(b).all() or not np.isfinite(m).all() or np.any((b<0)|(b>1)|(m<0)|(m>1)):
        raise ValueError('Matched probabilities required')
    return .95*b+.05*m


def eligible(result):
    return bool(result.get('status')=='SUPPORTED' and result.get('ci_low',0)>0 and result.get('positive_folds',0)>=2 and len(result.get('fold_deltas',[]))==3)


@dataclass(frozen=True)
class OffsetCorrection:
    center: float
    scale: float
    coefficient: float
    intercept: float
    training: dict

    def predict(self,baseline,model):
        b,m=np.asarray(baseline),np.asarray(model)
        if b.shape!=m.shape or not np.isfinite(b).all() or not np.isfinite(m).all() or np.any((b<0)|(b>1)|(m<0)|(m>1)):
            raise ValueError('Matched fixed baseline and full-procedure head probabilities required')
        x=(logit(np.clip(m,1e-6,1-1e-6))-self.center)/self.scale
        return expit(logit(np.clip(b,1e-6,1-1e-6))+self.intercept+self.coefficient*x)


def fit_correction(baseline,model,y,ids,groups,scope,qualification):
    scope.admit_training(ids)
    if not eligible(qualification):raise ValueError('Honest fixed diagnostic must qualify before learned correction')
    b,m,y=np.asarray(baseline),np.asarray(model),np.asarray(y)
    fixed_diagnostic(b,m)
    if y.shape!=b.shape or len(ids)!=len(y) or not np.isin(y,[0,1]).all():
        raise ValueError('Aligned TRAIN labels and honest inner OOF only')
    w=group_weights(groups)
    x=logit(np.clip(m,1e-6,1-1e-6));offset=logit(np.clip(b,1e-6,1-1e-6))
    center=float(np.average(x,weights=w));scale=max(float(np.sqrt(np.average((x-center)**2,weights=w))),1e-6)
    x=(x-center)/scale
    def objective(theta):
        z=offset+theta[0]+theta[1]*x
        residual=w*(expit(z)-y)
        loss=np.sum(w*(np.logaddexp(0,z)-y*z))+5.*theta[1]**2
        return float(loss),np.array([residual.sum(),residual@x+10.*theta[1]])
    result=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',
        options=dict(maxiter=250,ftol=1e-12,gtol=1e-6))
    if not result.success:raise RuntimeError('Offset logistic solver did not converge: '+str(result.message))
    return OffsetCorrection(center,scale,float(result.x[1]),float(result.x[0]),dict(**scope.provenance(),
        C=.1,baseline_logit_offset_coefficient=1.,rows=len(y),iterations=int(result.nit),
        estimator_fit_calls=1,training_inputs='Full-procedure inner excluded OOF M5 and excluded BL80 only',
        diagnostic_qualifies_fixed_five_percent_not_learned_M7=True,
        learned_M7_quality_not_evaluated_on_its_meta_training_labels=True))

"""At most eight precommitted cross proposals, conditioned on self support."""
from dataclasses import dataclass
import numpy as np
from src.next8.features import mechanical_controls
from src.next9.transitions import design as self_design,summaries
from src.next9.learning import fit_classifier
from src.next9.movement_screen import group_weights


def proposals(candidates,rule_specs,law_features):
    result=[]
    for destination in candidates:
        if destination not in rule_specs or destination not in law_features:continue
        blocked=set(mechanical_controls(destination))
        parents=[j for j in candidates if j not in blocked][:2]
        for parent in parents:
            if len(result)>=8:return result
            result.append((destination,parent))
    return result


def design(bank,rows,destination,parents):
    if len(parents)>2 or len(set(parents))!=len(parents):raise ValueError('At most two distinct frozen parents')
    if set(parents)&set(mechanical_controls(destination)):raise ValueError('Mechanical or self parent excluded')
    state=bank.pre_state(rows,destination)
    x=self_design(state['S1_DIRECTION'],state['direction_run_capped15'],state['slope8'],bank.series['context'][bank.series_index[rows]])
    parts=[x]
    for parent in parents:parts.append(bank.history(rows,parent)[:,[-1,-4,-16]])
    return np.column_stack(parts).astype(np.float32)


@dataclass
class CrossLaw:
    destination: int
    parents: tuple
    classifier: object
    prior: np.ndarray
    training: dict

    def probabilities(self,bank,rows):
        return self.classifier.predict_proba(design(bank,rows,self.destination,self.parents))


def fit_law(bank,rows,destination,parents,target,ids,groups,scope):
    x=design(bank,rows,destination,parents)
    model=fit_classifier(x,target,ids,groups,scope,classes=4)
    counts=np.bincount(np.asarray(target,dtype=int),weights=group_weights(groups),minlength=4)+.5
    return CrossLaw(destination,tuple(parents),model,counts/counts.sum(),dict(**scope.provenance(),
        destination=destination,parents=list(parents),variant='T2_FIXED_PARENT_EXTENSION',
        current_F_and_current_label_excluded=True,estimator_fit_calls=model.training['estimator_fit_calls']))


def path_features(bank,rows,law):
    rows=np.asarray(rows,dtype=np.int64)
    current=law.probabilities(bank,rows)
    prior=np.tile(law.prior,(len(rows),1));keep=bank.times[rows]>=1
    prior[keep]=law.probabilities(bank,rows[keep]-1)
    states=bank.pre_state(rows,law.destination)
    epsilon=bank.series['epsilon'][bank.series_index[rows],0,law.destination]
    x=summaries(states['S1_DIRECTION'],states['direction_run_capped15'],states['reversal_age_capped15'],states['slope8'],epsilon,prior,current)
    x[~keep,2]=0.
    return x[:,[2,3,4,5]]  # Existing surprise/reversal/continuation/entropy types.


def qualified_horizon(records):
    if len(records)!=3:raise ValueError('All three original internal panels required')
    class_good=sum(r['next_state_contrast']['groups']>=64 and r['next_state_contrast']['ci_low']>0 for r in records)>=2
    if not class_good:return None
    for h in [1,4,16]:
        contrasts=[]
        for record in records:
            r=next(v for v in record['forecast_contrasts'] if v['horizon']==h)
            contrasts.append(r)
        if (all(v[ref]['groups']>=64 and v[ref]['mean_gain']>=0 for v in contrasts for ref in ['D3','D5'])
                and sum(all(v[ref]['ci_low']>0 for ref in ['D3','D5']) for v in contrasts)>=2):
            return h
    return None
